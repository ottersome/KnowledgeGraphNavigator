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


# Maybe TOREM:
# def keyboard_interrupt_handler(signal, frame):
#     global page_offsets, page_ids
#     print("Keyboard interrupt detected. Exiting gracefully.")
#     print("Saving index...")
#     # with open(args.cachedir + "/index.pkl", "wb") as f:
#     # pickle.dump(index, f)
#     sys.exit(0)


def seek_and_read(
    bz2_file_path,
    target_id,
    id_offset_dict: Dict[int, int],
    block_offset_map: Dict[int, int],
) -> str:
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
        page_text = retrieve_page(f)

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
        print(
            f"Given current offset is {stream_offset} and next offset is {next_stream_offset}"
        )
        print(f"Reading {next_stream_offset - stream_offset} bytes")
        article_binary = f.read(next_stream_offset - stream_offset)

        # Decompress the stream from the current file position
        print(f"Decompressing {len(article_binary)} bytes")
        decompressed_data = bz2.decompress(article_binary)

        # Convert bytes to string for processing
        decompressed_data = decompressed_data.decode("utf-8")

        article = find_article_within_stream(decompressed_data, article_offset)
    assert article is not None, "Article not found"
    return article


def count_lines(file: IO) -> int:
    with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        return mm.read().count(b"\n")


def find_text_offset(file: IO, text: str) -> int:
    with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        offset = mm.find(text.encode())
        return offset


def find_nth_line(file: IO, n: int) -> str:
    with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        line_start = 0
        line_count = 0
        while line_count < n:
            line_end = mm.find(b"\n", line_start)
            if line_end == -1:
                # If we reach the end of the file before finding the nth line
                raise ValueError(f"File has fewer than {n} lines.")
            line_start = line_end + 1
            line_count += 1
        # Find the end of the nth line
        line_end = mm.find(b"\n", line_start)
        if line_end == -1:
            # If the nth line is the last line and doesn't end with a newline
            line_end = len(mm)
        return mm[line_start:line_end].decode()


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
            else:
                need_offset_updated.append([int(split[0]), int(split[1]), split[2], -1])

            cur_offset = new_offset
            bar.update(1)
        # The final addition
        for i in range(len(need_offset_updated)):
            need_offset_updated[i][3] = cur_offset
        final_list += need_offset_updated
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
    output_size_so_far = 0
    bar = tqdm(total=total_samples, desc="Exporting articles")
    sampled_sofar = 0
    byte_limit = args.output_size_limit * 1024 * 1024
    while sampled_sofar < total_samples:
        # TODO: Make it sample without replacement
        s = random.sample(range(num_lines), 1)[0]

        oi = offset_info[s]  # type: ignore
        print(f"Reading with title {oi[2]} at ofset {oi[0]}")
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
            print(
                f"Skipping {id_el if id_el is None else id_el.text} with title {title_el if title_el is None else title_el.text}"
            )
            continue

        id = id_el.text if id_el is not None else None
        title = title_el.text if title_el is not None else None
        text_el = text_el.text if text_el is not None else None

        bar.update(1)

        print(f"Saving {id} with title {title} and text")

        writer.writerow([id, title, text_el])

        output_size_so_far += os.path.getsize(args.sample_output_path)
        bar.set_description(
            f"Exporting articles. Output size so far {output_size_so_far/1e6:.2f} MB"
        )
        if output_size_so_far > byte_limit:
            print("Output size limit exceeded. Exiting...")
            print(f"Output size so far is {output_size_so_far}")
            print("Output is saved at {args.sample_output_path}")
            break
        sampled_sofar += 1
        os.system("clear")

    # Close the file
    print(f"Done exporting articles to {args.sample_output_path}")
    file.close()
