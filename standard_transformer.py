# %% Important imports
import argparse
from copy import deepcopy
from math import floor
from time import time
from typing import List, Tuple

import lightning as L
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner.tuning import Tuner
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, BartModel, BartTokenizer
from transformers.tokenization_utils_fast import TokenizerFast

from kgraphs.dataprocessing.gutenberg_data import BasicDataset, DatasetFactory
from kgraphs.lightning.base_autoregressive import BaseAutoregressive
from kgraphs.models.models import Transformer
from kgraphs.utils.logging import create_logger, time_to_largest_unit

# %% Some global initalization
logger = create_logger("MAIN")


def argsies():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", default="manu/project_gutenberg")
    # Hyperparameters
    ap.add_argument("--epochs", default=10)
    ap.add_argument("--batch_size", default=16, type=int)
    ap.add_argument("--model_name", default="facebook/bart-base")
    ap.add_argument("--model_tokenwindow_size", default=1024)
    ap.add_argument("--token_count_cap", default=10000)
    ap.add_argument("--model_dimension", default=768)
    ap.add_argument("--model_dimension_ff", default=3072)
    ap.add_argument("--num_layers", default=3)  # This is on the smaller side
    ap.add_argument("--num_heads", default=8)
    ap.add_argument("--dropout_rate", default=0.1)
    ap.add_argument("--masking_percentage", default=0.1)
    ap.add_argument("--raw_ds_location", default="./data/raw/")

    ap.add_argument("--chkpnt_loc", default="./checkpoints", type=str)

    ap.add_argument("-w", "--wandb", action="store_true")
    ap.add_argument("--wandb_project_name", help="Project name", type=str)
    ap.add_argument("--wr_name", help="Wand Run Name", type=str)
    ap.add_argument("--wr_notes", help="Wand Run Notes", type=str)

    return ap.parse_args()


class TextDataSet(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data.iloc[idx].tolist()).to(torch.long)


# %% Main Functions
if __name__ == "__main__":
    start_time = time()
    args = argsies()

    # Initialize wandb
    if args.wandb:
        wandb.init(project="kgraphs")

    # Load the Tokenizer
    tokenizer: BartTokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model_itself = BartModel.from_pretrained(args.model_name)
    # Get only the embedding layer of this model
    embedding_layer = model_itself.get_input_embeddings()  # type: ignore
    for param in embedding_layer.parameters():
        param.requires_grad = False

    dataset = DatasetFactory(
        dataset_name=args.dataset_name,
        ptrnd_tknzr=tokenizer,
        window_size=args.model_tokenwindow_size,
        amnt_tkns_for_training=args.token_count_cap,
        ds_location=args.raw_ds_location,
    )

    # data_module = DataModule()
    ds: Tuple[pd.DataFrame, ...] = dataset.load_split()
    train_df, val_df, test_df = ds
    logger.info(f"Size of train_ds {train_df.shape}")
    train_ds = TextDataSet(train_df)
    val_ds = TextDataSet(val_df)
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=1,
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch_size, drop_last=True, num_workers=1
    )

    test_it = next(iter(train_dl))

    logger.info(f"Loadad Train Dataset with {len(train_ds)} samples")
    logger.info(f"Loadad Val Dataset with {len(val_ds)} samples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the model
    model = Transformer(
        args.model_dimension,
        args.num_heads,
        args.num_layers,
        args.model_dimension_ff,
        args.model_tokenwindow_size,
        args.dropout_rate,
        tokenizer.pad_token_id,
        embedding_layer,  # type: ignore
    ).to(device)
    # Wrap it in the lightning Module
    lightning_module = BaseAutoregressive(model, tokenizer)

    # Get All arguments into a dictionary
    args_dict_str = str(deepcopy(vars(args)))
    notes = "The following are the training parameters\n"
    notes += args_dict_str

    # Initialize Wandb Logger
    wandb_logger = None
    if args.wandb:
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
        callbacks=[checkpoint_callback],
    )
    # TODO: its having some problems right now
    # tuner = Tuner(trainer)
    # tuner.scale_batch_size(
    #     lightning_module,
    #     train_dataloaders=train_dl,
    #     val_dataloaders=val_dl,
    #     mode="binsearch",
    # )

    logger.info("Starting to train the model")
    trainer.fit(lightning_module, train_dl, val_dl)

    exit()

    ### Old Stuff
    end_time = time()
    time, unit = time_to_largest_unit(end_time - start_time)
    logger.info(f"Loading data took {time:.2f} {unit} to run.")

    logger.info(f"Using device {device}")

    # Once the dataset is build we can load the model and train it on the stream
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterium = torch.nn.CrossEntropyLoss()

    num_batches = floor(len(train_ds) / args.batch_size)
    losses_epoch = []
    # Inner and outer bars
    e_bar = tqdm(range(args.epochs), desc="Epochs")
    b_bar = tqdm(range(num_batches), desc="Batches")
    for e in range(args.epochs):
        # Extract the batch
        losses_batch = []

        b_bar.reset()
        for b in range(num_batches):
            # Get the batch
            batch = train_ds.iloc[
                b * args.batch_size : (b + 1) * args.batch_size
            ].values.tolist()

            # Dont deal with small batches
            if len(batch) < args.batch_size:
                continue
            # Send data
            target = torch.Tensor(batch).to(torch.long).to(device)
            token_list_tensor = torch.tensor(batch).to(device)
            mlmd_tensor = mask_tensor(  # type: ignore
                token_list_tensor, tokenizer, float(args.masking_percentage)
            ).to(torch.long)
            # Multiply source by mask

            result = model(mlmd_tensor, target)

            loss = criterium(
                result.view(-1, result.shape[-1]),
                target.view(-1),
            ).mean()
            losses_batch.append(loss.item())
            if len(losses_batch) > 5 and args.wandb:
                wandb.log({"loss": sum(losses_batch[-5:]) / 5, "batch": b})

            # Get the backpropagation details
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            b_bar.update(1)

            # TODO: perhaps save the model environment eventually
        losses_epoch.append(sum(losses_batch) / len(losses_batch))
        # Report locally and to wandb
        logger.info(f"Epoch {e} has loss {losses_epoch[-1]}")
        if args.wandb:
            wandb.log({"loss": losses_epoch[-1], "epoch": e})

        # Validate on a per epoch basis
        e_bar.set_description("Evaluating")
        model.eval()
        num_eval_batches = len(val_ds) // args.batch_size
        eval_losses = []
        b_bar.set_description("Evaluation Batch")
        for i in range(num_eval_batches):
            batch = val_ds.iloc[
                i * args.batch_size : (i + 1) * args.batch_size
            ].values.tolist()

            token_list_tensor = torch.Tensor(batch).to(torch.long).to(device)

            mlmd_tensor = mask_tensor(  # type: ignore
                token_list_tensor, tokenizer, float(args.masking_percentage)
            ).to(torch.long)

            result = model(mlmd_tensor, token_list_tensor)
            softies = F.softmax(result, dim=-1)
            chosen_ids = torch.argmax(softies, dim=-1)

            # Log mlmd_tensor[:20] vs result[:20] vs token_list_tensor[:20] textually to see examples of guesses
            corrupted_translated = tokenizer.batch_decode(mlmd_tensor[:, :20])
            denosied_translated = tokenizer.batch_decode(chosen_ids[:, :20])
            true_translated = tokenizer.batch_decode(token_list_tensor[:, :20])
            logger.debug(f"MLMD: {corrupted_translated}")
            logger.debug(f"Result: {denosied_translated}")
            logger.debug(f"True: {true_translated}")

            loss = criterium(
                result.view(-1, result.shape[-1]),
                token_list_tensor.view(-1),
            )
            eval_losses.append(loss.item())
            b_bar.update(1)

        # Present the validation

        e_bar.update(1)
