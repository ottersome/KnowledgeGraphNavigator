# imports for connecting to freedb database
import json
import os
from argparse import ArgumentParser

import pandas as pd
import psycopg2
from SPARQLWrapper import JSON, SPARQLWrapper

from kgraphs.dataprocessing.wikipedia_data import articlestr_to_wellformatted
from kgraphs.net import get_wikipedia_json_content, get_wikipedia_raw_content
from kgraphs.queries.queries import WIKIPEDIA_URI, sample_fuseki_db
from kgraphs.utils.logging import MAIN_LOGGER_NAME, create_logger


def argsies():
    ap = ArgumentParser()
    ap.add_argument("--db_host", default="http://localhost:3030/fusekiservice/query")

    parsed_args = ap.parse_args()
    return parsed_args


if __name__ == "__main__":
    # Query
    args = argsies()
    logger = create_logger(MAIN_LOGGER_NAME)

    # Create Fuseki object
    sparql = SPARQLWrapper(args.db_host)
    logger.info("Obtaining the triplets")
    urls = sample_fuseki_db(
        sparql, sampling_amount=1, random_sample=False, db_uri=WIKIPEDIA_URI
    )

    logger.info("Obtained triplets")
    logger.info(f"Sampled triplets look like:\n{urls}")

    # Do some clean up
    for u in urls:
        content = get_wikipedia_json_content(u)
        pretty_json = json.dumps(content, indent=4)
        logger.debug(f"Raw content looks like\n{pretty_json}")
        clean_content = articlestr_to_wellformatted(content)
        logger.debug(f"Well formatted content looks like\n{content}")


# Load your own interpreter model
