"""
Main Idea of this script is to create a seq2seq model that will provide me with maximal amount of info
while keeping the encoded sequences minimal.
"""

import argparse
import ast
import os
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
import wandb
from lightning import Callback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from rich import traceback
from torch.utils.data import DataLoader
from transformers import BartModel, BartTokenizer

from kgraphs.lightning.base_compressor import BaseCompressor
from kgraphs.models.threestage_compressor import ThreeStageCompressor
from kgraphs.utils.logging import close_loggers, create_logger

torch.autograd.set_detect_anomaly(True)
traceback.install()


def argies():
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--epochs", type=int, default=2)
    ap.add_argument("-b", "--batch_size", type=int, default=8)
    ap.add_argument(
        "-d", "--dataset", default="./data/wikitext_dump.csv", help="Dataset Location"
    )
    ap.add_argument(
        "-c",
        "--chkpnt_loc",
        default="./checkpoints",
        help="Wehre to store the model's checkpoint",
    )
    ap.add_argument("--max_docsize", default=1024, help="Maximum size of the document")
    ap.add_argument("--min_docsize", default=128, help="Maximum size of the document")
    ap.add_argument(
        "--cachedata_loc",
        default=".cache/ds_encoded.csv",
        help="Where to store the cached data",
    )
    ap.add_argument("--lr", default=0.00001, type=float)
    # Model Architecture
    ap.add_argument("--num_layers", default=3, type=int)
    ap.add_argument("--d_model", default=768, type=int)
    ap.add_argument("--nhead", default=8, type=int)
    # Wandb Stuff
    ap.add_argument("-w", "--wandb", action="store_true")
    ap.add_argument(
        "--wandb_project_name",
        default="Document Compressor",
        help="Project name",
        type=str,
    )
    ap.add_argument("--wr_name", help="Wand Run Name", type=str)
    ap.add_argument("--wr_notes", help="Wand Run Notes", type=str)

    if not os.path.exists(".cache"):
        os.makedirs(".cache")

    return ap.parse_args()


# class DataLoader(torch.utils.data.DataLoader):
#     def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
#                  num_workers=0, collate_fn=default_collate, pin_memory=False, drop_last=False,
#                  timeout=0, worker_init_fn=None):
#         super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn)


def wrap_data_collator(pad_token_id: int):
    def data_collator(batch):
        # Ensure that given batch is right paddes to the longest sequence
        longest = max(len(x) for x in batch)
        padded_batch = torch.full((len(batch), longest), pad_token_id, dtype=torch.long)
        for i, seq in enumerate(batch):
            padded_batch[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        return padded_batch

    return data_collator


def data_collator(batch):
    # Ensure that given batch is right paddes to the longest sequence
    longest = max(len(x) for x in batch)
    padded_batch = torch.zeros((len(batch), longest), dtype=torch.long)
    for i, seq in enumerate(batch):
        padded_batch[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    padded_batch = padded_batch.to(torch.long)
    return padded_batch


class DataModule(L.LightningDataModule):
    """
    We assume the data is already sampled uniformly.
    That way we can just take the last 20% of the data to make it repeatable
    """

    def __init__(
        self,
        rawdata_location: str,
        cachedata_loc: str,
        batch_size: int,
        tokenizer: BartTokenizer,
        max_docsize: int,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.rawdata_location = rawdata_location
        self.cachedata_loc = cachedata_loc
        # CHECK: strange that we need this now and not before
        self.prepare_data_per_node = False
        self.max_docsize = max_docsize

    # Single process stuff
    def prepare_data(self):
        if not os.path.exists(self.cachedata_loc):
            logger.info("No cache found, preparing data")
            data = pd.read_csv(self.rawdata_location)
            # Tokenize it
            # Add new column for tokenized text
            logger.info("Tokenizing the data")
            # TODO: Check that the documents are not too big.
            new_col = data.apply(
                lambda row: self.tokenizer.encode(row["text"]), axis=1
            ).tolist()
            logger.info("COlumn created")
            data["tokenized_text"] = new_col
            # Drop those whose encoded length is greater than max_docsize
            old_amnt = len(data)
            data = data.loc[data["tokenized_text"].apply(len) <= self.max_docsize]
            logger.info(f"Dropped {old_amnt - len(data)} samples")

            data.to_csv(self.cachedata_loc, index=False)
        else:
            logger.info("No need to prepare data its already cached")
            logger.info(f"Loading cached data from {self.cachedata_loc}")

    # Per GPU stuff
    def setup(self, stage):
        logger.info(f"Loading cached data {self.cachedata_loc}")
        self.data = pd.read_csv(self.cachedata_loc)
        logger.info(
            f"Read the csv with length {len(self.data)} and columns {self.data.columns}"
        )
        # Use At to convert his column containing a string representation of a list to a list
        logger.info(f"Getting the stuff for dat of type {type(self.data)}")
        trn_val_border = int(0.8 * len(self.data))
        self.data_train = (
            self.data.loc[:trn_val_border, "tokenized_text"]
            .apply(lambda item: ast.literal_eval(item))
            .tolist()
        )
        self.data_val = (
            self.data.loc[trn_val_border:, "tokenized_text"]
            .apply(lambda item: ast.literal_eval(item))
            .tolist()
        )

    def train_dataloader(self):
        assert (
            self.tokenizer.pad_token_id is not None
        ), "Tokenizer is expected to have a pad token"
        dataloader = DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            collate_fn=wrap_data_collator(self.tokenizer.pad_token_id),
        )
        logger.info(f"Using dataloader with length {len(dataloader)}")
        return dataloader

    def val_dataloader(self):
        assert (
            self.tokenizer.pad_token_id is not None
        ), "Tokenizer is expected to have a pad token"
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            collate_fn=wrap_data_collator(self.tokenizer.pad_token_id),
        )


class BigBrother(Callback):
    def __init__(self, wandb_logger: WandbLogger):
        self.wandb_logger = wandb_logger
        self.writer = create_logger(__class__.__name__)
        self.params_store = []

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs,
        batch: Any,
        batch_idx: int,
    ):
        assert isinstance(pl_module, BaseCompressor)
        first_encoder = pl_module.model.st1
        all_weights = []
        for p in first_encoder.parameters():
            all_weights.append(p.detach().cpu().flatten())
            # Histogram of weights:
        final_weights = torch.cat(all_weights, dim=0).flatten().numpy()
        if self.wandb_logger is not None:
            self.wandb_logger.experiment.log(
                {"first_encoder": wandb.Histogram(final_weights, num_bins=100)}
            )
        if len(self.params_store) > 0:
            difference = np.sum(np.abs(self.params_store - final_weights))
            self.writer.debug(f"Difference is {difference}")
        self.params_store = final_weights

        self.writer.debug(f"Params of first encoder is {len(final_weights)}")


def main():
    args = argies()

    # For now we go with default tokenizer
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    model_itself = BartModel.from_pretrained("facebook/bart-base")
    pretrained_embedding = model_itself.get_input_embeddings()  # type: ignore

    datamodule = DataModule(
        rawdata_location=args.dataset,
        cachedata_loc=args.cachedata_loc,
        batch_size=args.batch_size,
        tokenizer=tokenizer,
        max_docsize=args.max_docsize,
    )
    # Initialize Wandb Logger
    wandb_logger = None
    args_dict_str = str(vars(args))
    if args.wandb:
        notes = "The following are the training parameters\n"
        notes += args_dict_str
        logger.info("🪄 Instantiating WandB")
        wandb_logger = WandbLogger(
            project=args.wandb_project_name, name=args.wr_name, notes=notes
        )

    ### Lightning Implemebtations
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.chkpnt_loc, save_top_k=3, monitor="val_loss"
    )
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        logger=wandb_logger,
        accumulate_grad_batches=4,
        max_epochs=args.epochs,
        val_check_interval=0.05,
        log_every_n_steps=1,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback, BigBrother(wandb_logger)],
    )

    # TODO: its having some problems right now
    # tuner = Tuner(trainer)
    # tuner.scale_batch_size(
    #     lightning_module,
    #     train_dataloaders=train_dl,
    #     val_dataloaders=val_dl,
    #     mode="binsearch",
    # )

    model = ThreeStageCompressor(
        args.d_model,
        args.nhead,
        args.num_layers,
        tokenizer.vocab_size,
        pretrained_embedding=pretrained_embedding,
        padding_id=tokenizer.pad_token_id,
        eos_id=tokenizer.eos_token_id,
    )
    amnt_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"The amount of parameters in our three stage compressor {amnt_params}")
    lightning_module = BaseCompressor(model, tokenizer, args.lr)

    logger.info("Starting to train the model")
    trainer.fit(
        lightning_module,
        datamodule=datamodule,
        # ckpt_path="./checkpoints/epoch=0-step=2330.ckpt",
    )


if __name__ == "__main__":
    logger = create_logger("__MAIN__")
    main()
    close_loggers(logger)
