import json
import os
from logging import DEBUG
from typing import Any, List, Tuple

import pandas as pd
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from kgraphs.utils.logging import create_logger

from .constants import (
    LEGALESE_END_MARKERS,
    LEGALESE_START_MARKERS,
    TEXT_END_MARKERS,
    TEXT_START_MARKERS,
)

"""
For now we will load project gutenberg and shard it into managable windows:
"""

_SPLIT_TO_POS = ["TRAIN", "VAL", "TEST"]


class BasicDataset(Dataset):
    def __init__(self, samples: List[Any]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def strip_headers(text):
    """
    Courtesy of `Gutenberg`. Cant find the github right now. If you can PR it please
    This on itself is a port from
    Note: this function is a port of the C++ utility by Johannes Krugel. The
    original version of the code can be found at:
    http://www14.in.tum.de/spp1307/src/strip_headers.cpp

    Args:
        text (unicode): The body of the text to clean up.

    Returns:
        unicode: The text with any non-text content removed.

    """
    lines = text.splitlines()
    sep = str(os.linesep)

    out = []
    i = 0
    footer_found = False
    ignore_section = False

    for line in lines:
        reset = False

        if i <= 600:
            # Check if the header ends here
            if any(line.startswith(token) for token in TEXT_START_MARKERS):
                reset = True

            # If it's the end of the header, delete the output produced so far.
            # May be done several times, if multiple lines occur indicating the
            # end of the header
            if reset:
                out = []
                continue

        if i >= 100:
            # Check if the footer begins here
            if any(line.startswith(token) for token in TEXT_END_MARKERS):
                footer_found = True

            # If it's the beginning of the footer, stop output
            if footer_found:
                break

        if any(line.startswith(token) for token in LEGALESE_START_MARKERS):
            ignore_section = True
            continue
        elif any(line.startswith(token) for token in LEGALESE_END_MARKERS):
            ignore_section = False
            continue

        if not ignore_section:
            out.append(line.rstrip(sep))
            i += 1

    return sep.join(out)


class DatasetFactory:
    CACHE_RELPATH = "./.cache/datasets/"
    SUPP_CTX_LEN = 128  # Supplementary Context Length
    CACHE_FORMAT = "{split}_{split_percent}_tknCap{token_cap}.parquet"

    def __init__(
        self,
        dataset_name: str,
        ptrnd_tknzr: PreTrainedTokenizer,
        window_size: int,
        amnt_tkns_for_training: int = -1,
        split: Tuple = (0.85, 0.15, 0),
    ):
        """
        dataset_name: usually just the same one
        ptrnd_tknzr: PreTrainedTokenizer,
        ds_size:
        window_size: int: How big of a window we will have when training language model
        tokenizer_cap: int = How many tokenizers for dataset. Mostly for keeping test trains fast
        """

        self.logger = create_logger(__class__.__name__, DEBUG)
        self.dataset_name = dataset_name
        self.window_size = window_size
        self.tknzr = ptrnd_tknzr
        self.split = split

        # Token Cap is semantically for training so we pull the rest as:
        self.amnt_tkns_for_trning = amnt_tkns_for_training
        self.tot_tkns = amnt_tkns_for_training / split[0]

        # TOREM: Trashy state use
        self.cur_amount_of_tokens = 0

        # Check for cache data

        # os.makedirs(CACHE_NAME, exists_ok=True)
        self.cached_files = self._check_cache()  # If empty then we have no cache

    def _check_cache(self):
        # Ensure Cache works
        file_path = os.path.abspath(__file__)
        file_dir = os.path.dirname(file_path)
        self.cache_path = os.path.join(file_dir, self.CACHE_RELPATH)

        files = []
        for i, s in enumerate(self.split):
            if s == 0.0:
                continue
            filename = self.CACHE_FORMAT.format(
                split=s,
                split_percent=_SPLIT_TO_POS[i],
                token_cap=self.amnt_tkns_for_trning,
            )
            filepath = os.path.join(self.cache_path, filename)
            if os.path.exists(filepath):
                files.append(filepath)
            else:
                files.clear()
                break

        return files

    def _load_cached_files(self, files_to_load: List[str]) -> List[BasicDataset]:
        dss: List[BasicDataset] = []

        for f in files_to_load:
            parquet: pd.DataFrame = pd.read_parquet(f)
            dss.append(BasicDataset(parquet.values.tolist()))

        return dss

    def load_split(self):
        if len(self.cached_files) > 0:
            self._load_cached_files(self.cached_files)
        else:
            self._compute_ds()

    def _compute_ds(self):
        """
        💫
        Assume we always load in the same order.
        """
        dataset_iter = load_dataset(self.dataset_name, split="en", streaming=True)
        train: List[List[int]] = []
        cap = 2
        f = open("./dumpy.log", "w")
        b = tqdm(total=cap + 1)
        self.cur_amount_of_tokens = 0
        for i, doc in enumerate(dataset_iter):
            if self.cur_amount_of_tokens > self.amnt_tkns_for_trning:
                break

            new_list = self._doc(doc["text"], b)  # type:ignore
            new_list_size = len(new_list)
            self.logger.debug(f"Have added {new_list_size} windows to our dataset")
            train += new_list  # type:ignore
            b.set_description(f"Added {len(train)} docs")
            b.update(1)
        # List to pretty, indent json
        pretty_list = json.dumps(train, indent=4)
        f.write(pretty_list)
        # Dump all the train list
        f.close()
        # Once that is all good we cache it in some parquet file

    def _cache_tokenized(self, samples_list):
        df = pd.DataFrame(samples_list)
        # TODO: finish
        # df.to_parquet(

    def _doc(self, doc: str, bar: tqdm):
        # Just dump the boook so I can see what it lopoks like for now
        with open("book.log", "w") as f:
            f.write(doc)

        doc = strip_headers(doc)
        with open("stripped_book.log", "w") as f:
            f.write(doc)

        self.logger.debug(f"Finding ourselves in _doc")
        segments = doc.split("\n")
        tkd_segs = [
            self.tknzr.encode(s.strip(), add_special_tokens=False) for s in segments
        ]

        self.logger.debug(f"With {len(segments)} segments split by \\n")
        bar.set_description(f"Working document with {len(segments)}")

        token_windows = []
        cur_segidx = 0
        cur_subsegidx = 0

        while (
            cur_segidx < len(tkd_segs)
            and self.cur_amount_of_tokens <= self.amnt_tkns_for_trning
        ):
            # Some Segments are emtpy
            if len(tkd_segs[cur_segidx]) == 0:
                cur_segidx += 1
                continue

            self.logger.debug(
                f"In loop with cur_segidx {cur_segidx} and cur_subsegidx {cur_subsegidx}"
                f" and {self.cur_amount_of_tokens} cur_amount_of_tokens"
            )

            review_bias = 0 if cur_subsegidx <= self.SUPP_CTX_LEN else self.SUPP_CTX_LEN
            beg = cur_subsegidx - review_bias

            tope = min(
                len(tkd_segs[cur_segidx]) - 1,
                cur_subsegidx - review_bias + self.window_size,
            )

            cur_window = tkd_segs[cur_segidx][beg:tope]
            cur_window_debug = self.tknzr.convert_ids_to_tokens(cur_window)
            pair = [f"{a} - {b}" for a, b in zip(cur_window, cur_window_debug)]
            token_windows.append(pair)
            # TODO: Change this to count actual tokens
            self.cur_amount_of_tokens += len(cur_window)

            cur_subsegidx += self.window_size
            if tope == len(tkd_segs[cur_segidx]) - 1:
                cur_segidx += 1
                cur_subsegidx = 0

        self.logger.debug(
            f"Wrapping up with the amount of tokens {self.cur_amount_of_tokens}"
        )

        return token_windows

    def getDatasets(self) -> BasicDataset:
        pass

    def save_tknized_in_local_cache(self):
        pass
