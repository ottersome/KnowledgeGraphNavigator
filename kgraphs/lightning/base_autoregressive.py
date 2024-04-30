from typing import Tuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import BartTokenizer

from ..utils.logging import create_logger


class BaseAutoregressive(L.LightningModule):
    def __init__(
        self,
        autoregressive_model: nn.Module,
        tokenizer: BartTokenizer,
        masking_percentage: float = 0.15,
    ):
        super().__init__()
        self.model = autoregressive_model
        self.criterium = torch.nn.CrossEntropyLoss()
        self.tokenizer = tokenizer
        self.masking_percentage = masking_percentage
        self.my_logger = create_logger(__class__.__name__)

    def training_step(self, batch, batch_idx):
        target = batch
        mlmd_tensor, mask_tensor = get_mlm_tensor(  # type: ignore
            target, self.tokenizer, float(self.masking_percentage)
        )
        mlmd_tensor.to(torch.long)
        mask_tensor.to(torch.bool)

        result = self.model(mlmd_tensor, target.clone())

        # TODO: Ensure this loss is only activiated for MLM tokens
        masked_result = result[mask_tensor]
        masked_target = target[mask_tensor]

        loss = self.criterium(masked_result, masked_target)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        target = batch
        mlmd_tensor, mask_tensor = get_mlm_tensor(  # type: ignore
            target, self.tokenizer, float(self.masking_percentage)
        )
        mlmd_tensor.to(torch.long)
        mask_tensor.to(torch.bool)
        extended_mask = get_idx_around_mask(mask_tensor, ctx_win_radius=3)

        model_device = next(self.model.parameters()).device
        result = self.model(mlmd_tensor, target.clone())
        masked_result = result[extended_mask]
        masked_target = target[extended_mask]
        softies = F.softmax(masked_result, dim=-1)
        loss = F.cross_entropy(masked_result, masked_target)
        chosen_ids = torch.argmax(softies, dim=-1)

        self.my_logger.debug(f"Estimated Ids: {chosen_ids}")
        self.my_logger.debug(f"True Targets: {masked_target}")
        self.my_logger.debug(f"Final criterium: {loss.mean().item()}")

        self.log("val_loss", loss.mean().item())

        # Log mlmd_tensor[:20] vs result[:20] vs token_list_tensor[:20] textually to see examples of guesses
        # corrupted_translated = self.tokenizer.batch_decode(mlmd_tensor[:, :20])
        # denosied_translated = tokenizer.batch_decode(chosen_ids[:, :20])
        # true_translated = tokenizer.batch_decode(token_list_tensor[:, :20])
        # logger.debug(f"MLMD: {corrupted_translated}")
        # logger.debug(f"Result: {denosied_translated}")
        # logger.debug(f"True: {true_translated}")
        #
        # loss = criterium(
        #     result.view(-1, result.shape[-1]),
        #     token_list_tensor.view(-1),
        # )

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.0001)


def get_idx_around_mask(mask_tensor: Tensor, ctx_win_radius: int = 3):
    """
    Increase tensor True values around True Values
    """
    mask_tensor = mask_tensor.to(torch.bool)
    mask_copy = mask_tensor.clone()
    for i in range(1, ctx_win_radius + 1):
        mask_copy |= torch.roll(mask_tensor, shifts=i, dims=1)
        mask_copy[:, :i] = False
        mask_copy |= torch.roll(mask_tensor, shifts=-i, dims=1)
        mask_copy[:, -i:] = False

    mask_tensor |= mask_copy
    return mask_tensor


def get_mlm_tensor(
    tokens: torch.Tensor, tokenizer: BartTokenizer, masking_percentage: float
) -> Tuple[Tensor, Tensor]:
    """
    Take a list of a batch_size  x model_tokenwindow_size
    and mask using masking_percentage on each model
    """
    new_list = tokens.clone()
    # Iterate over each element
    # OPTIM: vectorize
    # Create a mask vector with masking_percentage set to true
    mask_vector = torch.rand(new_list.shape, device=tokens.device) < masking_percentage

    if tokenizer.mask_token_id is not None:
        new_list[mask_vector] = tokenizer.mask_token_id
    else:
        raise ValueError("Mask token not found in tokenizer")

    return new_list, mask_vector
