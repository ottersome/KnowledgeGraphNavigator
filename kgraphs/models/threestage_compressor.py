from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from torch.nn import functional as F
from torch.nn.modules.transformer import _detect_is_causal_mask, _get_seq_len

from kgraphs.utils.logging import create_logger

from .transformer import PositionalEncoding, generate_srcsequence_masks_fortorch


class NonParallelDecoder(TransformerDecoder):
    # def __init__(self, decoder_layer, num_layers):
    #     super().__init__(decoder_layer, num_layers)
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: Optional[bool] = None,
        memory_is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        seq_len = _get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)

        for mod in self.layers:
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=tgt_is_causal,
                memory_is_causal=memory_is_causal,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output[:, -1, :].squeeze(1)


class ThreeStageCompressor(nn.Module):
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1
    ACTIVATION = "relu"
    LAYER_NORM_EPS = 1e-5
    BATCH_FIRST = True
    NORM_FIRST = False
    BIAS = True
    MAX_SEQ_LENGTH = 1024
    TERMINATION_THRESHOLD = 1
    MAX_DOCENC_LENGTH = 128

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        vocab_size: int,
        pretrained_embedding: nn.Module,
        padding_id: int,
        eos_id: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.logger = create_logger(__class__.__name__)
        # Used Pretrained embeddings
        self.encoder_embedding = pretrained_embedding
        self.decoder_embedding = pretrained_embedding
        self.padding_id = padding_id
        self.eos_id = eos_id

        self.logger.info(f"Running a three stage compressor with {d_model} dimensions")

        # TODO: pretrain this one
        self.positional_encoding = PositionalEncoding(d_model, self.MAX_SEQ_LENGTH)
        self.d_model = d_model

        self.encoder_layer = TransformerEncoderLayer(
            d_model,
            nhead,
            self.DIM_FEEDFORWARD,
            self.DROPOUT,
            self.ACTIVATION,
            self.LAYER_NORM_EPS,
            self.BATCH_FIRST,
            self.NORM_FIRST,
            self.BIAS,
        )
        self.decoder_layer = TransformerDecoderLayer(
            d_model,
            nhead,
            self.DIM_FEEDFORWARD,
            self.DROPOUT,
            self.ACTIVATION,
            self.LAYER_NORM_EPS,
            self.BATCH_FIRST,
            self.NORM_FIRST,
            self.BIAS,
        )
        self.st1 = TransformerEncoder(self.encoder_layer, num_layers)
        self.st2 = NonParallelDecoder(self.decoder_layer, num_layers)
        # TODO: Consider if we want a new layer for the final decoder
        self.st3 = TransformerDecoder(self.decoder_layer, num_layers)
        self.vocab_layer = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(self.DROPOUT)

    def forward(self, src_tokens: Tensor):
        # TODO: weight have to return encoded_text too.
        num_heads = self.encoder_layer.self_attn.num_heads
        src_mask = generate_srcsequence_masks_fortorch(
            src_tokens, self.padding_id, num_heads
        )
        self.logger.debug(
            f"src_mask is of shape {src_mask.shape} with device {src_mask.device}"
        )
        enc_embedding = self.encoder_embedding(src_tokens)
        self.logger.debug(f"Embedded src is of shape {enc_embedding.shape}")
        # CHECK: Do we need dropout here?
        inp_embedding = self.dropout(self.positional_encoding(enc_embedding))
        self.logger.debug(
            f"Inp embedding is of shape {inp_embedding.shape} with device {inp_embedding.device}"
        )
        # Lets ensure the embeddings are okay
        self.logger.debug(f"embeddings are of shape {enc_embedding.shape}")
        self.logger.debug(f"embeddings content: {enc_embedding}")

        # What about droput here?
        self.logger.debug(f"Inp embedding is of shape {inp_embedding.shape}")
        self.logger.debug(f"Inp embedding content: {inp_embedding}")

        # TODO: Ensure the masks are correct
        self.logger.debug(f"src_mask is of shape {src_mask.shape}")
        self.logger.debug(f"src_mask is of values {src_mask[0,0,1]}")
        context_embeddings = self.st1(inp_embedding, src_mask)

        self.logger.debug(f"context embeddings is of shape {context_embeddings.shape}")
        self.logger.debug(f"context embeddings is of values {context_embeddings[0,0,1]}")
        # Set (0,0,...,0) to EOS
        # TODO: Enssure we get the right dimension here
        tgt = (
            torch.zeros(self.d_model)
            .unsqueeze(0)
            .unsqueeze(1)
            .repeat(context_embeddings.shape[0], 1, 1)
        ).to(context_embeddings.device)
        self.logger.debug(f"context_embeddings is of shape {context_embeddings.shape}")
        self.logger.debug(f"context_embeddings content: {context_embeddings}")
        results = [[] for i in range(context_embeddings.shape[0])]
        finished = [None] * context_embeddings.shape[0]
        eos_point = torch.zeros(self.d_model).to(context_embeddings.device)
        current_ids_to_work = set(range(context_embeddings.shape[0]))

        # TODO: Enforce use of EOS token with some sort of regularization
        # Non Parallel Decoder
        batchmax_length = 0
        while (
            len(current_ids_to_work) > 0 and batchmax_length < self.MAX_DOCENC_LENGTH
        ):  # Check length means amount of items
            self.logger.debug(
                f"Current tgt is of shape {tgt.shape} shape of memory is {context_embeddings.shape}"
            )
            # CHECK: Must we place something in between?

            # Get the computation
            compressed_enc = self.st2(tgt=tgt, memory=context_embeddings).squeeze()
            self.logger.debug(f"Compressed enc is of shape {compressed_enc.shape}")

            # Check distance of output to eos
            dist_to_eos = torch.abs(
                compressed_enc
                - eos_point.unsqueeze(0)
                .unsqueeze(1)
                .repeat(compressed_enc.shape[0], 1, 1)
            ).sum()
            # CHECK: Adding is correspondent
            for i, ciw in enumerate(current_ids_to_work):
                results[ciw].append(compressed_enc[i])

            # Once added check if we have finished
            idxs_to_remove = torch.where(dist_to_eos < self.TERMINATION_THRESHOLD)[0]
            for trm in idxs_to_remove:
                self.logger.debug(f"Removing {trm}")
                if trm in current_ids_to_work:
                    current_ids_to_work.remove(trm)

            self.logger.debug(
                f"At length {batchmax_length} Current ids to work are {current_ids_to_work}"
            )
            batchmax_length += 1

        self.logger.debug(f"Out of the loop with a max length of {batchmax_length}")
        # Create a padded tensor from results
        compdoc_padded = torch.zeros((len(results), batchmax_length, self.d_model)).to(
            context_embeddings.device
        )
        compdoc_padded_mask = torch.zeros((len(results), batchmax_length)).to(
            context_embeddings.device
        )
        self.logger.debug(
            f"Compdoc padded is of shape {compdoc_padded.shape} and compdoc_padded_mask is of shape {compdoc_padded_mask.shape}\n"
            f"With devices {compdoc_padded.device} and {compdoc_padded_mask.device}"
        )
        # CHECK: That the tensors are maintaining computation graph
        for i, res in enumerate(results):
            compdoc_padded[i, : len(res)] = torch.stack(res)
            compdoc_padded_mask[i, : len(res)] = 1

        # We are reconstructing so we are taking the src tokens as the tgt
        # CHECK: Do we need to provide the mask for source?
        self.logger.debug(
            f"The dimensions for tgt are {src_tokens.shape}, compdoc_padded are {compdoc_padded.shape}"
            f" and compdoc_padded_mask are {compdoc_padded_mask.shape}"
        )
        # Just create a causal mask for inp_embedding
        # tgt_mask = generate_srcsequence_masks_fortorch(
        #     inp_embedding, self.padding_id, num_heads
        # )
        decoded_text = self.st3(
            tgt=inp_embedding,
            tgt_mask=src_mask,  # Self attention
            memory=compdoc_padded,
            # memory_mask=compdoc_padded_mask,  # Cross Attention
        )
        self.logger.debug(f"Decoded text is of shape {decoded_text.shape}")
        final_text = self.vocab_layer(decoded_text).squeeze()
        self.logger.debug(f"Final text is of shape {final_text.shape}")
        return final_text
