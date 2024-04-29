# imports for connecting to freedb database
import os
from argparse import ArgumentParser

import pandas as pd
import psycopg2
from SPARQLWrapper import JSON, SPARQLWrapper

from kgraphs.net import get_wikipedia_content
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
    triplets = sample_fuseki_db(sparql, 100, random_sample=False, db_uri=WIKIPEDIA_URI)

    logger.info("Obtained triplets")
    logger.info(f"Sampled triplets look like:\n{triplets}")

    # Load the pre-trained embedding model

    # Load your own interpreter model
