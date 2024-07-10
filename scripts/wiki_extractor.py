import bz2
import csv
import mmap
import os
import pdb
import pickle
import random
import signal
import sys
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from time import sleep, time
from typing import IO, Dict, List, Sequence, Tuple, Union

import indexed_bzip2 as ibz2
from lxml import etree
from tqdm import tqdm


def full_path(path) -> str:
    # Expand the user directory (~) and convert to an absolute path
    return os.path.abspath(os.path.expanduser(path))


def argsies():
    ap = ArgumentParser(
        description="Extracting Wikipedia articles from a dump file. And also sampling from it."
    )
    ap.add_argument(
        "-i",
        "--bz2_loc",
        default="~/Datasets/enwiki-20240620-pages-articles-multistream.xml.bz2",
        help="Path to your dump file.",
        type=full_path,
    )
    ap.add_argument(
        "-c",
        "--lineoffset_cache",
        default="./.cache/lineoffsets.dat",
        help="Where to store line offset cache",
    )
    ap.add_argument(
        "-j",
        "--index_file",
        default="~/Datasets/enwiki-20240620-pages-articles-multistream-index.txt",
        type=full_path,
    )
    ap.add_argument(
        "-l",
        "--output_size_limit",
        type=int,
        default=1024,
        help="Maximum size limit for the output (in megabytes)",
    )
    ap.add_argument(
        "-o",
        "--sample_output_path",
        help="Where the extraction will be dumped.",
        default="./output.csv",
        type=str,
    )
    # Two mutually exclusive arguments, samplikng_fraction and sampling_amount
    # group = ap.add_mutually_exclusive_group(required=True)
    group = ap.add_mutually_exclusive_group()
    group.add_argument(
        "--sampling_fraction",
        help="Percentage of large datatset to fraction.",
        type=float,
        required=False,
    )
    group.add_argument(
        "--sampling_amount",
        help="Amount of articles to sample",
        type=int,
        default=1000,
        required=False,
    )

    ap.add_argument(
        "--no-autoindent",
        action="store_true",
        help="Indent the output.",
    )

    return ap.parse_args()


def find_article(
    file_path: str,
    stream_offset: int,
    next_stream_offset: int,
    article_offset: int,
) -> ET.Element:
    # Find the next offset
    article = ""
    with open(file_path, "rb") as f:
        # Ensure we can find the next offset

        # Check EOF
        # Seek to the start of the compressed stream
        f.seek(stream_offset)
        article_binary = f.read(next_stream_offset - stream_offset)

        # Decompress the stream from the current file position
        decompressed_data = bz2.decompress(article_binary)

        # Convert bytes to string for processing
        decompressed_data = decompressed_data.decode("utf-8")

        article = find_article_within_stream(decompressed_data, article_offset)
    assert article is not None, "Article not found"
    return article


def count_lines(file: IO) -> int:
    with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        return mm.read().count(b"\n")


def get_line_offsets(file_path: str) -> List[Tuple[int, int, str, int]]:
    # Use the systems cat | wc -l to count lines
    line_count = count_lines(open(file_path))
    bar = tqdm(total=line_count)
    final_list = []
    with open(file_path) as f:
        need_offset_updated = []
        cur_offset = -1

        while line := f.readline():
            split = line.split(":")
            new_offset = int(split[0])
            if new_offset != cur_offset and cur_offset != -1:
                # Add to them the cur_offset and update final_list
                for i in range(len(need_offset_updated)):
                    need_offset_updated[i][3] = new_offset
                final_list += need_offset_updated
                need_offset_updated.clear()

            need_offset_updated.append([int(split[0]), int(split[1]), split[2], -1])
            cur_offset = new_offset
            bar.update(1)
        # The final addition
        for i in range(len(need_offset_updated)):
            need_offset_updated[i][3] = cur_offset
        final_list += need_offset_updated
    assert (
        len(final_list) == line_count
    ), f"Length of final list is not equal to line count but rather {len(final_list)}"
    return final_list


def find_article_within_stream(stream: str, id: int) -> ET.Element:
    # Parse all to xtml
    root = etree.fromstring("<root>" + stream + "</root>")
    xpath_expr = f".//page[id='{id}']"
    return root.xpath(xpath_expr)[0]
    # return etree.tostring(root.xpath(xpath_expr)[0]).decode("utf-8")


def pretty_print_element(element):
    return etree.tostring(element, pretty_print=True, encoding="unicode")


def passes_criteria(article_body: Union[ET.Element, None]):
    # Criteria on length
    if article_body is None:
        return False
    assert article_body.text is not None, "Text is None"
    if len(article_body.text) < 100:
        return False
    return True


if __name__ == "__main__":
    """
    Only really built for small fractions of the data
    """
    args = argsies()

    dump_file = open(
        "/home/ottersome/Datasets/enwiki-20240620-pages-articles-multistream.xml.bz2",
        "rb",
    )

    offset_info = []

    num_lines = count_lines(open(args.index_file))
    print(f"Found {num_lines} lines in the index file")
    if os.path.exists(args.lineoffset_cache):
        print(f"Found line offset cache at {args.lineoffset_cache}")
        with open(args.lineoffset_cache, "rb") as f:
            offset_info = pickle.load(f)
    else:
        print(
            f"No line offset cache found at {args.lineoffset_cache}. This might take a while..."
        )
        offset_info = get_line_offsets(args.index_file)
        print(f"Saving line offset cache at {args.lineoffset_cache}")
        pickle.dump(offset_info, open(args.lineoffset_cache, "wb"))

    total_samples = (
        args.sampling_amount
        if args.sampling_amount is not None
        else args.sampling_fraction * num_lines
    )

    # Ensure that the csv file does not exist, if it does then we will exit
    if os.path.exists(args.sample_output_path):
        print(f"Output file {args.sample_output_path} already exists. Exiting...")
        sys.exit(0)

    # Prep for dumping
    file = open(args.sample_output_path, "a")
    columns = ["id", "title", "text"]
    writer = csv.writer(file)
    writer.writerow(columns)

    # We then read the articles
    print(f"We will read {total_samples} articles...")
    # We will export this all as a readable csv format
    bar = tqdm(total=total_samples, desc="Exporting articles")
    sampled_sofar = 0
    byte_limit = args.output_size_limit * 1024 * 1024
    skips_counter = 0
    while sampled_sofar < total_samples:
        # TODO: Make it sample without replacement
        s = random.sample(range(num_lines), 1)[0]

        oi = offset_info[s]  # type: ignore
        data: ET.Element = find_article(args.bz2_loc, oi[0], oi[3], oi[1])

        # If no text available then skip
        text_el = data.find("revision/text")
        title_el = data.find("title")
        id_el = data.find("id")
        if (
            text_el in ["", None]
            or not passes_criteria(text_el)
            or title_el in ["", None]
            or id_el in ["", None]
        ):
            skips_counter += 1
            continue

        id = id_el.text if id_el is not None else None
        title = title_el.text if title_el is not None else None
        text_el = text_el.text if text_el is not None else None

        writer.writerow([id, title, text_el])

        output_size_so_far = os.path.getsize(args.sample_output_path)
        bar.set_description(
            f"Exporting articles. Output size so far {output_size_so_far/1e6:.2f} MB\n"
            f"Exported {id} with title {title}\n"
            f"Have skipped {skips_counter} articles\n"
        )
        bar.update(1)
        if output_size_so_far > byte_limit:
            print("Output size limit exceeded. Exiting...")
            print(f"Output size so far is {output_size_so_far}")
            print("Output is saved at {args.sample_output_path}")
            break
        sampled_sofar += 1

    # Close the file
    print(f"Done exporting articles to {args.sample_output_path}")
    file.close()
