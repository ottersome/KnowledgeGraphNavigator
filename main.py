# %% Important imports
import argparse
from copy import deepcopy
from math import floor
from time import time
from typing import List, Tuple

import pandas as pd
import torch
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

    return ap.parse_args()


def masked_tensor(
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
    return new_list


# %% Main Functions
if __name__ == "__main__":
    start_time = time()
    args = argsies()

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
    losses = []
    for e in tqdm(range(args.epochs)):
        # Extract the batch
        logger.debug(f"Entering epoch number {e}")
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
            masked_tensor = masked_tensor(  # type: ignore
                token_list_tensor, tokenizer, float(args.masking_percentage)
            ).to(torch.long)
            # Multiply source by mask

            result = model(masked_tensor, target)

            logger.info(
                f"Result from model is of shape {result.shape} and target is of shape {target.shape}"
            )
            loss = criterium(result.view(-1, ), target.squeeze(0)).mean()
            losses.append(loss.item())

            # Get the backpropagation details
            optimizer.zero_grad()
            logger.info("Just passed the optimizer ")ZZ
            loss.backward()
            logger.info("Just did backpropagation")
            optimizer.step()
            logger.info("Just did optimizer step")

            # TODO: perhaps save the model environment eventually
