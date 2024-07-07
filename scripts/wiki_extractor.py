import bz2
import os
import pickle
import random
import signal
import sys
import xml.etree.ElementTree as ET
from argparse import ArgumentParser
from time import sleep
from typing import IO, Dict, List, Sequence, Tuple

import indexed_bzip2 as ibz2
from tqdm import tqdm


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
        "--block_offset_map_cache",
        default="./.cache/bom_cache.dat",
        help="Where to store block offset map",
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
    ap.add_argument("--read_chunk_size", default=1024, type=int)
    # Ask for mode, either extraction or sampling (multiple choice)
    ap.add_argument("-m", "--mode", choices=["extract", "sample"], default="sample")
    # Make it so we ask for extract_output_dir only if we are in extract mode
    ap.add_argument("--cachedir", help="Where to store the cache", default="./.cache")

    # Ensure existance of cachedir
    args = ap.parse_args()
    if args.mode == "extract" and not os.path.exists(args.cachedir):
        os.makedirs(args.cachedir)

    return args


def create_page_index(
    file_path: str, block_offset_map: Dict[int, int] = {}
) -> Tuple[List[int], List[int]]:
    global page_offsets, page_ids
    current_file_offset = 0
    page_ids = []
    page_offsets = []
    # Read amount of bytes without loading file into RAM
    tot_numb_bytes = os.path.getsize(file_path)
    print(f"Total number of bytes is {tot_numb_bytes/1e9}")

    with ibz2.open(file_path, parallelization=os.cpu_count()) as wiki_dump_file:
        wiki_dump_file.set_block_offsets(block_offset_map)
        wiki_dump_file.seek(int(tot_numb_bytes * 0.98))
        cur_page_id = -1
        cur_page_title = ""
        on_page = False

        # For each line
        cur_line = wiki_dump_file.readline()
        items_added_so_far = 0
        bar = tqdm(total=tot_numb_bytes)

        while cur_line is not None:
            cur_line = cur_line.decode("utf-8")
            assert isinstance(
                cur_line, str
            ), f"Line is not a string, rather {type(cur_line)}"
            stripped_line = cur_line.strip()
            if stripped_line == "<page>":
                on_page = True
                page_offsets.append(current_file_offset)
            elif stripped_line == "</page>":
                # assert on_page, "We are not on a page"
                if cur_page_id == -1:
                    pass
                    # page_offsets.pop()
                else:
                    items_added_so_far += 1
                cur_page_id = -1
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
                page_ids.append(cur_page_id)
            # current_file_offset += len(cur_line)
            if items_added_so_far % 1000 == 0:
                bar.set_description(f"Added {items_added_so_far} items")
            bar.update(wiki_dump_file.tell() - current_file_offset)
            current_file_offset = wiki_dump_file.tell()
            cur_line = wiki_dump_file.readline()
            if wiki_dump_file.tell() > tot_numb_bytes:
                print(f"Cur line is :{cur_line}")
    return page_offsets, page_ids


def read_chunk_orderly(file_ptr: IO, read_chunk_size: int) -> Tuple[bytes, int]:
    """
    Unlike below, it assumes file_ptr points at the start of a chunk
    """
    # Assert that file pointe is 'rb'
    assert file_ptr.mode == "rb", "File pointer is not in read binary mode"
    assert has_data(file_ptr), "Nothing to read"
    assert file_ptr.read(3) == b"BZh", "File pointer is not in read binary mode"

    initial_pos = file_ptr.tell()
    # Read first chunk length
    # CHECK: We actually need to read 8 bytes
    unit = int(1e5)
    # Print read the single byte
    chunk_length = int.from_bytes(file_ptr.read(1), byteorder="big") * unit
    # Read thue chunk
    print(f"Currently at pos {file_ptr.tell()} with chunk length {chunk_length}")
    magic_number = file_ptr.read(3)
    crc_checksum = file_ptr.read(3)
    file_ptr.seek(0)
    chunk = decompress_bzip2_block(file_ptr.read(chunk_length))
    print(f"A bit of the chunk is {chunk[:100]}")

    exit()
    header_loc = file_ptr.tell()

    file_ptr.seek(initial_pos)
    return chunk, header_loc


def decompress_bzip2_block(encoded_block):
    # Decompress the bzip2 encoded block using the bz2 module
    try:
        decompressed_data = bz2.decompress(encoded_block)
        return decompressed_data.decode(
            "utf-8"
        )  # Assuming the original data was UTF-8 encoded
    except Exception as e:
        print(f"Error during bzip2 decompression: {str(e)}")
        return None


def read_chunk(file_ptr: IO, read_chunk_size: int) -> Tuple[bytes, int]:
    """
    Read a chunk of bytes from a file
    """
    # Assert that file pointe is 'rb'
    assert file_ptr.mode == "rb", "File pointer is not in read binary mode"
    assert has_data(file_ptr), "Nothing to read"

    # Get final pos
    og_pos = file_ptr.tell()
    file_ptr.seek(0, 2)
    final_pos = file_ptr.tell()
    file_ptr.seek(0)
    can_continue = lambda: final_pos > file_ptr.tell()

    chunk_so_far = file_ptr.read(read_chunk_size)
    print(f"Read chunk of size {len(chunk_so_far)}")
    header_loc = chunk_so_far.find(b"BZh")
    print(f"HEader found at {header_loc}")
    while header_loc == -1 and can_continue():
        next_chunk = file_ptr.read(read_chunk_size)
        if not next_chunk:
            break
        chunk_so_far += next_chunk
        header_loc = chunk_so_far[-read_chunk_size:].find(b"BZh")

    # Back to og_position
    file_ptr.seek(og_pos)
    if header_loc == -1:  # No more headers to be found
        return chunk_so_far, header_loc
    else:
        return chunk_so_far[:-header_loc], header_loc


def has_data(file: IO) -> bool:
    current_position = file.tell()
    file.seek(0, 2)
    final_position = file.tell()
    file.seek(current_position)
    return final_position > current_position


def peek(file, num_bytes=1):
    current_position = file.tell()
    data = file.read(num_bytes)
    file.seek(current_position)

    return data


def create_offsets_index(
    file_path: str, block_offset_map: Dict[int, int] = {}
) -> Tuple[List[int], List[int]]:
    global page_offsets, page_ids
    current_file_offset = 0
    page_ids = []
    page_offsets = []

    with ibz2.open(file_path, parallelization=os.cpu_count()) as wiki_dump_file:
        wiki_dump_file.set_block_offsets(block_offset_map)
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


def create_block_offsets_map(bz2_path: str, cache_path: str) -> Dict[int, int]:
    # Check if has already been created
    if os.path.exists(cache_path):
        print(f"Found cached offset map at {cache_path}. Loading... ")
        with open(cache_path, "rb") as offsets_file:
            return pickle.load(offsets_file)

    file = ibz2.open(bz2_path, parallelization=os.cpu_count())
    print(f"Creating block offset map for {bz2_path}")
    block_offsets = file.block_offsets()  # can take a while
    print(f"Block offset map created for {bz2_path}")
    with open(cache_path, "wb") as offsets_file:
        pickle.dump(block_offsets, offsets_file)
    file.close()

    return block_offsets


def seek_and_read(
    bz2_file_path,
    target_id,
    id_offset_dict: Dict[int, int],
    block_offset_map: Dict[int, int],
):
    """
    Given an id that can be found in page_ids we use the corresonding page_offsets to find the
    correct offset in the bz2 file.
    """

    # Find the nearest index entry
    offset = id_offset_dict[target_id]
    print(f"Id {target_id} has offset {offset}")
    # with open(bz2_file_path, "rb") as f:
    with ibz2.open(bz2_file_path, parallelization=os.cpu_count()) as f:
        f.set_block_offsets(block_offset_map)
        f.seek(offset)
        page_txt = retrieve_page(f)

    return page_text


def retrieve_page(
    file_io: IO,
) -> str:
    # Ensure first line is <page>
    assert file_io.readline().decode("utf-8") == "<page>"
    # Look for content within <text> tags
    while file_io.readline().decode("utf-8") != "</text>":
        pass
    content_ofinterst = ""
    while file_io.readline().decode("utf-8") != "</page>":
        content_ofinterst += file_io.readline().decode("utf-8")
    return content_ofinterst


def find_article(
    file_path: str,
    stream_offset: int,
    article_offset: int,
    article_title: str,
):
    next_offset = 700582
    with open(file_path, "rb") as f:
        # Seek to the start of the compressed stream
        f.seek(stream_offset)
        print(f"Reading {next_offset - stream_offset} bytes")
        article_binary = f.read(next_offset - stream_offset)

        # Decompress the stream from the current file position
        print(f"Decompressing {len(article_binary)} bytes")
        decompressed_data = bz2.decompress(article_binary)

        # Convert bytes to string for processing
        decompressed_data = decompressed_data.decode("utf-8")

        print(f"Decompressed data is {decompressed_data}")

        # Find the article within the decompressed data
        start_index = decompressed_data.find(f"<title>{article_title}</title>")

        if start_index == -1:
            print(f"Article '{article_title}' not found.")
            return None

        # Extract the article content
        end_index = decompressed_data.find("</page>", start_index) + len("</page>")
        article_content = decompressed_data[start_index:end_index]

        return article_content


if __name__ == "__main__":
    args = argsies()

    find_article(args.bz2_loc, 553, 247, "AbeceDarians")

    exit()

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
        # index = create_offsets_index(bz2_file_path)
        # First Create the block offset map for future use
        bom = create_block_offsets_map(bz2_file_path, args.block_offset_map_cache)
        print("About to calculate index")
        index = create_page_index(bz2_file_path, bom)

    elif args.mode == "sample":
        with open(args.cachedir + "/index.pkl", "rb") as f:
            id_offset_dict = pickle.load(f)
            # Also look for block offset map
            assert os.path.exists(
                args.block_offset_map_cache
            ), f"Block offset map not found at {args.block_offset_map_cache}"
            bom = pickle.load(open(args.block_offset_map_cache, "rb"))
            sample_id = random.choice(list(id_offset_dict.keys()))
            print(f"Sampling index {sample_id} from file")
            data = seek_and_read(args.bz2_loc, sample_id, id_offset_dict, bom)
            print(data)
