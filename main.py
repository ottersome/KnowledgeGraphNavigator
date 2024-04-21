# %% Important imports
import argparse
from copy import deepcopy
from math import floor
from time import time
from typing import List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from transformers import AutoTokenizer, BartModel, BartTokenizer
from transformers.tokenization_utils_fast import TokenizerFast

from kgraphs.dataprocessing.data import BasicDataset, DatasetFactory
from kgraphs.models.models import Transformer
from kgraphs.utils.logging import create_logger, time_to_largest_unit

# %% Some global initalization
logger = create_logger("MAIN")


def argsies():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", default="manu/project_gutenberg")
    # Hyperparameters
    ap.add_argument("--epochs", default=10)
    ap.add_argument("--batch_size", default=4)
    ap.add_argument("--model_name", default="facebook/bart-base")
    ap.add_argument("--model_tokenwindow_size", default=1024)
    ap.add_argument("--token_count_cap", default=10000)
    ap.add_argument("--model_dimension", default=768)
    ap.add_argument("--model_dimension_ff", default=3072)
    ap.add_argument("--num_layers", default=3)  # This is on the smaller side
    ap.add_argument("--num_heads", default=12)
    ap.add_argument("--dropout_rate", default=0.1)
    ap.add_argument("--masking_percentage", default=0.1)
    ap.add_argument("--raw_ds_location", default="./data/raw/")

    ap.add_argument("-w", "--wandb", action="store_true")

    return ap.parse_args()


def mask_tensor(
    tokens: torch.Tensor, tokenizer: BartTokenizer, masking_percentage: float
) -> torch.Tensor:
    """
    Take a list of a batch_size  x model_tokenwindow_size
    and mask using masking_percentage on each model
    """
    new_list = deepcopy(tokens)
    # Iterate over each element
    # OPTIM: vectorize
    for i in range(new_list.shape[0]):
        for j in range(new_list.shape[1]):
            if torch.rand(1) < masking_percentage:
                new_list[i, j] = tokenizer.mask_token_id  # type: ignore
    # CHECK: Log these bois
    return new_list


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

    ds: Tuple[pd.DataFrame, ...] = dataset.load_split()
    train_ds, val_ds, test_ds = ds
    logger.info(f"Loadad Train Dataset with {len(train_ds)} samples")
    logger.info(f"Loadad Val Dataset with {len(val_ds)} samples")
    logger.info(f"Loadad Test Dataset with {len(test_ds)} samples")
    end_time = time()
    time, unit = time_to_largest_unit(end_time - start_time)
    logger.info(f"Loading data took {time:.2f} {unit} to run.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device {device}")

    # Once the dataset is build we can load the model and train it on the stream
    model = Transformer(
        args.model_dimension,
        args.num_heads,
        args.num_layers,
        args.model_dimension_ff,
        args.model_tokenwindow_size,
        args.dropout_rate,
        embedding_layer,  # type: ignore
    ).to(device)
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
