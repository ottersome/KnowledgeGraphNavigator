# %% Important imports
import argparse
import os
from time import time

from transformers import AutoTokenizer

from kgraphs.dataprocessing.data import BasicDataset, DatasetFactory
from kgraphs.models.models import MultiHeadAttention
from kgraphs.utils.logging import create_logger, time_to_largest_unit

# %% Some global initalization
logger = create_logger("MAIN")


def argsies():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", default="manu/project_gutenberg")
    ap.add_argument("--tokenizer_name", default="facebook/bart-base")
    ap.add_argument("--model_tokenwindow_size", default=1024)
    ap.add_argument("--token_count_cap", default=10000)
    ap.add_argument("--model_dimension", default=1024)
    ap.add_argument("--num_heads", default=16)
    ap.add_argument("--raw_ds_location", default="./data/raw/")

    return ap.parse_args()


# %% Main Functions
if __name__ == "__main__":
    start_time = time()
    args = argsies()

    # Load the Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    dataset = DatasetFactory(
        dataset_name=args.dataset_name,
        ptrnd_tknzr=tokenizer,
        window_size=args.model_tokenwindow_size,
        amnt_tkns_for_training=args.token_count_cap,
        ds_location=args.raw_ds_location,
    )

    train_ds, val_ds, test_ds = dataset.load_split()
    logger.info(f"Loadad Train Dataset with {len(train_ds)} samples")
    logger.info(f"Loadad Val Dataset with {len(val_ds)} samples")
    logger.info(f"Loadad Test Dataset with {len(test_ds)} samples")

    # Once the dataset is build we can load the model and train it on the stream
    model = MultiHeadAttention(args.model_dimension, args.num_heads)
    end_time = time()
    time, unit = time_to_largest_unit(end_time - start_time)
    logger.info(f"Script took {time:.2f} {unit} to run.")
