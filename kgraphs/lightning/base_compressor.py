import lightning as L
import torch
import torch.nn as nn
from transformers import BartTokenizer
import torch.autograd

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

        loss = self.criterium(recovered_text_flat, target_flat)
        loss_avg = loss.mean()
        loss_avg_item = loss_avg.item()
        amnt_vram = torch.cuda.memory_allocated(loss_avg.device) / 1e9
        self.my_logger.debug(f"Using {amnt_vram: .2f} GB of VRam after the loss is calculated")
        self.my_logger.info(f"Loss is {loss_avg_item}")
        self.log("train_loss", loss_avg_item)
        return loss_avg

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.my_logger.info(f"VALIDATION STEP BOIII: ENTER")
            self.my_logger.info(f"Performing validation for batch {batch_idx}")
            target = batch
            target = target.to(torch.long)
            # target_flat = target.flatten()
            inf_text = self.model(target)
            argmaxed_inf = torch.argmax(inf_text,dim=-1) 
            detokenized_inf = self.tokenizer.batch_decode(argmaxed_inf)
            detokeniized_target = self.tokenizer.batch_decode(target)
            # recovered_text_flat = recovered_text.view(-1, recovered_text.shape[2])

            for b in range(target.shape[0]):
                self.my_logger.debug(f"Target is:\n{detokeniized_target[b]}\n")
                self.my_logger.debug(f"Inference is:\n{detokenized_inf[b]}\n")

            self.my_logger.info(f"VALIDATION STEP BOIII: EXIT")
            self.log("val_loss", 0)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
