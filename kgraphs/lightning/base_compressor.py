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
        self.my_logger.info(f"Tryign to get somewhere within this")
        target = batch
        target = target.to(torch.long)

        recovered_text = self.model(target)

        # TODO: Ensure that the loss tries to minimize the number of embeddings from the first decoder
        loss = self.criterium(recovered_text, target)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
