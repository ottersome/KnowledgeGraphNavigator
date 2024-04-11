# %% Important imports
import argparse
import os

from transformers import AutoTokenizer

from kgraphs.dataprocessing.data import BasicDataset, DatasetFactory
from kgraphs.utils.logging import create_logger

# %% Some global initalization
logger = create_logger("MAIN")


def argsies():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_name", default="manu/project_gutenberg")
    ap.add_argument("--tokenizer_name", default="facebook/bart-base")
    ap.add_argument("--model_tokenwindow_size", default=1024)
    ap.add_argument("--token_count_cap", default=10000)

    return ap.parse_args()


# %% Main Functions
if __name__ == "__main__":
    args = argsies()

    # Load the Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    dataset = DatasetFactory(
        dataset_name=args.dataset_name,
        ptrnd_tknzr=tokenizer,
        window_size=args.model_tokenwindow_size,
        amnt_tkns_for_training=args.token_count_cap,
    )

    dataset.load_split()

    # Once the dataset is build we can load the model and train it on the stream
