from typing import Tuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import BartTokenizer

from ..utils.logging import create_logger


class BaseCompressor(L.LightningModule):
    def __init__(
        self,
        compressor_model: nn.Module,
        tokenizer: BartTokenizer,
        masking_percentage: float = 0.15,
        lr: float = 0.0001,
        # For debugging:
    ):
        super().__init__()
        self.model = compressor_model
        # self.criterium = torch.nn.CrossEntropyLoss()
        self.tokenizer = tokenizer
        self.masking_percentage = masking_percentage
        self.my_logger = create_logger(__class__.__name__)
        # TODO: Ensure criterium enforeces sparseness of compression
        self.criterium = torch.nn.CrossEntropyLoss()
        self.lr = lr

    def training_step(self, batch, batch_idx):
        target = batch
        target = target.to(torch.long)
        target_flat = target.flatten()

        recovered_text = self.model(target)
        recovered_text_flat = recovered_text.view(-1, recovered_text.shape[2])

        # TODO: Ensure that the loss tries to minimize the number of embeddings from the first decoder
        self.my_logger.debug(
            f"Target flattened is of shape {target_flat.shape}"
            f" and recovered text flattened is of shape {recovered_text_flat.shape}"
        )

        loss = self.criterium(recovered_text_flat, target_flat)
        self.my_logger.debug(f"Non avg loss shape is {loss.shape}")
        loss_avg = loss.mean()
        self.my_logger.debug(f"Loss is {loss_avg.item()}")
        self.log("train_loss", loss_avg.item())
        return loss_avg

    def validation_step(self, batch, batch_idx):
        self.my_logger.info(f"Performing validation for batch {batch_idx}")
        target = batch
        target = target.to(torch.long)
        # target_flat = target.flatten()
        recovered_text = self.model(target)
        argmaxed_target = torch.argmax(target,dim=-1) 
        detokenized = self.tokenizer.batch_decode(argmaxed_target)
        # recovered_text_flat = recovered_text.view(-1, recovered_text.shape[2])

        for b in range(recovered_text.shape[0]):
            self.my_logger.debug(f"Detokenized looks like {detokenized}\n")
            self.my_logger.debug(f"Target is {target[b]}")


        self.log("val_loss", 0)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
