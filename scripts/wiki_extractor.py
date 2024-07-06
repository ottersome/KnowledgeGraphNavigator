import bz2
import os
import pdb
import pickle
import random
import signal
import sys
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from time import sleep
from typing import IO, Dict, List, Tuple


def argsies():
    ap = ArgumentParser(
        description="Extracting Wikipedia articles from a dump file. And also sampling from it."
    )
    ap.add_argument(
        "-i",
        "--bz2_loc",
        default="/home/ottersome/Datasets/enwiki-20240620-pages-articles-multistream.xml.bz2",
        help="Path to your dump file.",
        type=str,
    )
    ap.add_argument(
        "-o",
        "--extract_output_dir",
        help="Where the extraction will be dumped.",
        type=str,
        required="--mode" in sys.argv
        and sys.argv[sys.argv.index("--mode") + 1] == "extract",
    )
    ap.add_argument(
        "-f",
        "--sampling_fraction",
        help="Percentage of large datatset to fraction.",
        type=float,
        required="--mode" in sys.argv
        and sys.argv[sys.argv.index("--mode") + 1] == "sample",
    )
    # for handling ipython auto-indentation
    ap.add_argument(
        "--no-autoindent",
        action="store_true",
        help="Indent the output.",
    )
    # Ask for mode, either extraction or sampling (multiple choice)
    ap.add_argument("--mode", choices=["extract", "sample"], default="extract")
    # Make it so we ask for extract_output_dir only if we are in extract mode
    ap.add_argument("--cachedir", help="Where to store the cache", default="./.cache")

    # Ensure existance of cachedir
    args = ap.parse_args()
    if args.mode == "extract" and not os.path.exists(args.cachedir):
        os.makedirs(args.cachedir)

    return args


def read_chunk(file_ptr: IO, read_chunk_size: int) -> Tuple[bytes, bool]:
    """
    Read a chunk of bytes from a file
    """
    # Assert that file pointe is 'rb'
    assert file_ptr.mode == "rb", "File pointer is not in read binary mode"
    assert peek(file_ptr) == b"BZ", "File pointer is not in read binary mode"

    header_loc = lambda bin_point: bin_point.find(b"BZh")
    cur_chunk_offset = lambda chunk: len(chunk)

    chunk_so_far = file_ptr.read(read_chunk_size)
    while header_loc(chunk_so_far[:-read_chunk_size]) == -1:
        next_chunk = file_ptr.read(read_chunk_size)
        if not next_chunk:
            break
        chunk_so_far += next_chunk

    final_hdr_loc = header_loc(chunk_so_far[:-read_chunk_size])
    if final_hdr_loc == -1:  # No more headers to be found
        return chunk_so_far, False
    else:
        return chunk_so_far[:-final_hdr_loc], True


def peek(file):
    current_position = file.tell()
    data = file.read(num_bytes)
    file.seek(current_position)

    return data


def create_offsets_index(file_path: str) -> Tuple[List[int], List[int]]:
    global page_offsets, page_ids
    current_file_offset = 0
    page_ids = []
    page_offsets = []

    with bz2.open(file_path, "rt") as wiki_dump_file:
        cur_page_id = -1
        cur_page_title = ""
        on_page = False
        page_so_far = ""

        for i, line in enumerate(wiki_dump_file):
            assert isinstance(line, str), "Line is not a string"
            stripped_line = line.strip()
            if stripped_line == "<page>":
                on_page = True
                # print(f"Found page at offests_{current_file_offset}")
                page_offsets.append(current_file_offset)
            elif stripped_line == "</page>":
                # print(f"Found end of page at offests_{current_file_offset}")
                assert on_page, "We are not on a page"
                # print(f"Pagerdump {page_so_far}")
                if cur_page_id == -1:
                    page_offsets.pop()
                cur_page_id = -1
                print(".", end="", flush=True)
                on_page = False
            elif len(stripped_line) >= 8 and stripped_line[:7] == "<title>":
                # Perhaps use later
                cur_page_title = stripped_line[7:-8]
            elif (
                len(stripped_line) >= 4
                and stripped_line[:4] == "<id>"
                and cur_page_id == -1
            ):
                cur_page_id = int(stripped_line[4:-5])
                # print(f"Found page id at offests_{current_file_offset}")
                page_ids.append(cur_page_id)
            page_so_far += line
            current_file_offset += len(line)
    return page_offsets, page_ids


def keyboard_interrupt_handler(signal, frame):
    global page_offsets, page_ids
    print("Keyboard interrupt detected. Exiting gracefully.")
    if args.mode == "extract":
        print("Saving index...")
        with open(args.cachedir + "/index.pkl", "wb") as f:
            pickle.dump(dict(zip(page_ids, page_offsets)), f)
    elif args.mode == "sample":
        print("Saving index...")
        with open(args.cachedir + "/index.pkl", "wb") as f:
            pickle.dump(index, f)
    sys.exit(0)


# def seek_and_read(bz2_file_path, index, target_offset, read_size=1024):
#     # Find the nearest index entry
#     nearest_index = max([i for i in index if i[0] <= target_offset], key=lambda x: x[0])
#     uncompressed_offset, compressed_offset = nearest_index
#     with open(bz2_file_path, "rb") as f:
#         f.seek(compressed_offset)
#         decompressor = bz2.BZ2Decompressor()
#         decompressed_data = b""
#         while len(decompressed_data) < (
#             target_offset - uncompressed_offset + read_size
#         ):
#             chunk = f.read(1024)
#             if not chunk:
#                 break
#             decompressed_data += decompressor.decompress(chunk)
#     start = target_offset - uncompressed_offset
#     return decompressed_data[start : start + read_size]


def seek_and_read(
    bz2_file_path, target_id, id_offset_dict: Dict[int, int], block_size=1024
):
    """
    Given an id that can be found in page_ids we use the corresonding page_offsets to find the
    correct offset in the bz2 file.
    """

    # Find the nearest index entry
    offset = id_offset_dict[target_id]
    with open(bz2_file_path, "rb") as f:
        f.seek(offset)
        decompressor = bz2.BZ2Decompressor()
        decompressed_data = b""
        while len(decompressed_data) < read_size:
            chunk = f.read(1024)
            if not chunk:
                break
            pdb.set_trace()
            decompressed_data += decompressor.decompress(chunk)
    return decompressed_data


if __name__ == "__main__":
    args = argsies()

    # # Lets first load and test the picle
    # with open(args.cachedir + "/index.pkl", "rb") as f:
    #     index = pickle.load(f)
    # print(f"Sampled index {index}")

    # Create a keyboard interrupt handler
    signal.signal(signal.SIGINT, keyboard_interrupt_handler)
    if args.mode == "extract":
        bz2_file_path = args.bz2_loc
        global index
        print(f"Using file {bz2_file_path}. This *will* take a while")
        index = create_offsets_index(bz2_file_path)

    elif args.mode == "sample":
        print("Sampling from file")
        with open(args.cachedir + "/index.pkl", "rb") as f:
            id_offset_dict = pickle.load(f)
            sample_id = random.choice(list(id_offset_dict.keys()))
            data = seek_and_read(args.bz2_loc, sample_id, id_offset_dict)
            print(data)
