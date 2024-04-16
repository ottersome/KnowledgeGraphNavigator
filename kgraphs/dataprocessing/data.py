import copy
import json
import os
import posix
import re
from logging import DEBUG
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Tuple

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


# Lets create an iterable class that will take in :
# either a file diretory list or a return from `load_dataset`
# and *yield* text within them
class TextStream(Iterable):
    def __init__(self, dataset_iter: Iterable[Any]):
        self.dataset_iter = dataset_iter
        # HACK: but meh:
        first_item = next(iter(self.dataset_iter))
        self.local = False
        try:
            "text" in first_item  # type:ignore
        except:
            self.local = True

    def __iter__(self):
        # Check if it is an iterable of str or the one provided by load_dataset
        print("THE INSTANCE IS ", type(self.dataset_iter))
        if self.local:
            for filepath in self.dataset_iter:
                with open(filepath, "r") as f:
                    yield f.read()
        else:
            for doc in self.dataset_iter:
                yield doc["text"]


# DatasetFactory
class DatasetFactory:
    CACHE_RELPATH = "./.cache/datasets/"
    SUPP_CTX_LEN = 128  # Supplementary Context Length
    CACHE_FORMAT = "{split}_{split_percent}_tknCap{token_cap}.parquet"

    SPACE_PERC_THRESH = 0.1
    MAX_INTRASPACE_THRESHOLD = 5
    MIN_NUMBER_WORDS = 15
    WINDOW_OVERLAP = 128  # CHECK: implement if necessary
    REMOVE_REGEXES = [
        r"(?i)^\s*(Chapter|Section|Part)\s+\w+",  # Headings
        r"^\s*\d+(\.\d+)*\s+.*$",  # Numerical Patterns
        r"^\s*(M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}))\s*$",  # Roman Numerals
        r"\[\d+\]",  # References
    ]

    def __init__(
        self,
        dataset_name: str,
        ptrnd_tknzr: PreTrainedTokenizer,
        window_size: int,
        amnt_tkns_for_training: int = -1,
        split: Tuple = (0.85, 0.15, 0),
        ds_location: str = "",
        stream: bool = False,
    ):
        """
        dataset_name: usually just the same one
        ptrnd_tknzr: PreTrainedTokenizer,
        ds_size:
        window_size: int: How big of a window we will have when training language model
        tokenizer_cap: int = How many tokenizers for dataset. Mostly for keeping test trains fast
        """

        self.logger = create_logger(__class__.__name__, DEBUG)
        self.ds_location = ds_location
        self.dataset_name = dataset_name
        self.window_size = window_size
        self.tknzr = ptrnd_tknzr
        self.split = split
        self.stream = stream

        if ds_location != "" and self.stream == True:
            self.logger.warn(
                f"You have provided ds_location {ds_location}, yet are using `self.stream"
                f"Local dataset location {ds_location} will be ignored"
            )

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

    def _initialize_regex(self) -> List[re.Pattern]:
        return [re.compile(r) for r in self.REMOVE_REGEXES]

    def get_dataset_iter(self) -> Iterable[Any]:
        if self.stream:
            return TextStream(
                load_dataset(self.dataset_name, split="en", streaming=True)
            )
        else:
            return TextStream(Path(self.ds_location).iterdir())

    def _compute_ds(self):
        """
        💫
        Assume we always load in the same order.
        """
        dataset_iter = self.get_dataset_iter()
        train: List[List[int]] = []
        cap = 2
        f = open("./dumpy.log", "w")
        b = tqdm(total=cap + 1)
        self.cur_amount_of_tokens = 0

        clean_regexes = self._initialize_regex()

        for i, doc in enumerate(dataset_iter):
            if self.cur_amount_of_tokens > self.amnt_tkns_for_trning:
                break

            new_list = self._doc(doc, b, clean_regexes)  # type:ignore
            new_list_size = len(new_list)
            self.logger.debug(f"Have added {new_list_size} windows to our dataset")
            train += new_list  # type:ignore
            b.set_description(f"Added {len(train)} docs")
            b.update(1)
            break

        # List to pretty, indent json
        self.logger.info(f"Final Train boi is \n{train}")
        pretty_list = json.dumps(train, indent=4)
        f.write(pretty_list)
        # Dump all the train list
        f.close()
        # Once that is all good we cache it in some parquet file

    def _cache_tokenized(self, samples_list):
        df = pd.DataFrame(samples_list)
        # TODO: finish
        # df.to_parquet(

    def _clean_segment(self, segment: str, removal_rgxs: List[re.Pattern]) -> str:
        ### Miscelannea Round
        # TODO: remove if we deem it truly useless.
        # Count amount of words
        #
        # split_segment = segment.replace("\n", "").split(" ")
        # wc = len(split_segment)
        # if wc < self.MIN_NUMBER_WORDS:
        # Count amount of spaces in this string
        # count_spaces = segment.count(" ")
        # if (count_spaces / tot_len) > self.SPACE_PERC_THRESH:
        #     return ""

        # Check if spaces are too often close to each other
        # (Like in table of contents)

        tot_len = len(segment)
        if tot_len == 0:
            return ""

        max_space = 0
        for word in segment.split(" "):
            if word == "":
                max_space += 1
            if max_space > self.MAX_INTRASPACE_THRESHOLD:
                return ""

        ###  Regex round
        for r in removal_rgxs:
            segment = r.sub("", segment)

        ########################################
        # Looks clean, final touches
        ########################################

        final_segment = segment.strip()

        return final_segment

    def _doc_to_window_iterator(
        self, doc: str, removal_rgxs: List[re.Pattern]
    ) -> Iterator[List[int]]:
        # Make inner hard copy of doc
        copy_doc = copy.deepcopy(doc)
        tokensAvailable = lambda x: len(x) > 1

        while tokensAvailable(copy_doc):
            # Look ahead for the next new line
            return_tokens = []
            debug_list = []
            needMoreTokens = lambda x: len(x) < self.window_size

            while needMoreTokens(return_tokens):
                if not tokensAvailable(copy_doc):
                    break  # TODO: prettify
                next_newline = copy_doc.find("\n")

                clean_seg = self._clean_segment(copy_doc[:next_newline], removal_rgxs)

                if len(clean_seg) == 0:
                    copy_doc = copy_doc[next_newline + 1 :]
                    continue

                # Get to work on the clean segment
                word_split = clean_seg.split(" ")

                for i, word in enumerate(word_split):
                    # enc_word = self.tknzr.tokenize(
                    #     word, add_special_tokens=False
                    # )  # DEBUG: remove
                    enc_word = self.tknzr.encode(word, add_special_tokens=False)
                    return_tokens += enc_word

                    debug_list.append((word, return_tokens))

                    if len(return_tokens) > self.window_size:
                        copy_doc = " ".join(clean_seg[i:]) + copy_doc  # Leftovers
                        break

                copy_doc = copy_doc[next_newline + 1 :]

            yield return_tokens
        # Generator is done

    def _doc(self, doc: str, bar: tqdm, clean_regexs: List[re.Pattern]):
        # Just dump the boook so I can see what it lopoks like for now
        with open("book.log", "w") as f:
            f.write(doc)

        doc = strip_headers(doc)
        with open("stripped_book.log", "w") as f:
            f.write(doc)

        self.logger.debug(f"Finding ourselves in _doc :[")

        doc_tokenizer = self._doc_to_window_iterator(doc, clean_regexs)

        token_windows = []

        # While we can iterate over doc_tokenizer
        tkn_win_iterator = doc_tokenizer
        self.logger.debug("Amout to start parsing these bois")
        for tkn_win in tkn_win_iterator:
            ## Cleaning
            self.logger.debug(f"Added token window of size {len(tkn_win)}")
            token_windows.append(tkn_win)
        self.logger.debug("Done parsing these bois")
        return token_windows

    def save_tknized_in_local_cache(self):
        pass
