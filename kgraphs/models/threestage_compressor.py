from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
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

    # Temporary:
    COMPRESSION_LENGTH = 20

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
        num_heads = self.encoder_layer.self_attn.num_heads
        src_mask = generate_srcsequence_masks_fortorch(
            src_tokens, self.padding_id, num_heads
        )
        enc_embedding = self.encoder_embedding(src_tokens)
        inp_embedding = self.dropout(self.positional_encoding(enc_embedding))

        # TODO: Ensure the masks are correct
        context_embeddings = self.st1(inp_embedding, src_mask)
        tgt = (
            torch.zeros(self.d_model)
            .unsqueeze(0)
            .unsqueeze(1)
            # DEBUG: Lets start with just 10 outputs.
            # Then we change when we canthink of information maximization
            .repeat(context_embeddings.shape[0], self.COMPRESSION_LENGTH, 1)
        ).to(context_embeddings.device)
        initial_tgt = (
            torch.zeros(self.d_model)
            .unsqueeze(0)
            .unsqueeze(1)
            .repeat(context_embeddings.shape[0], self.COMPRESSION_LENGTH, 1)
            .to(context_embeddings.device)
        )

        results = [[] for i in range(context_embeddings.shape[0])]
        ## Non Parallel Decoder
        # TODO: IMPORTANT: Enforce use of EOS token with some sort of regularization
        batchmax_length = 0
        # DEBUG: Remove the self.COMPRESSION_LENGTH and change it to be dynamic
        results = [initial_tgt]
        for i in range(self.COMPRESSION_LENGTH):
            targeto = torch.cat(results, dim=1)
            compressed_enc = self.st2(tgt=targeto, memory=context_embeddings).view(
                targeto.shape[0], -1, targeto.shape[-1]
            )
            results.append(compressed_enc[:, -1, :][:, None, :])

        # At the very end we calculate tgt based on the results
        tgt = torch.cat(results, dim=1)

        # DEBUG: Amount of VRam being Used
        # TODO: This will be used later when we move to dynamic length encoding 
        # Create a padded tensor from results
        compdoc_padded = torch.zeros((len(results), batchmax_length, self.d_model)).to(
            context_embeddings.device
        )
        compdoc_padded_mask = torch.zeros((len(results), batchmax_length)).to(
            context_embeddings.device
        )
        compdoc_padded = tgt
        # CHECK: Do we need to provide the mask for source?

        decoded_text = self.st3(
            tgt=inp_embedding,
            tgt_mask=src_mask,  # Self attention
            memory=tgt,
            # memory_mask=compdoc_padded_mask,  # Cross Attention
        )
        final_text = self.vocab_layer(decoded_text).squeeze()

        return final_text
