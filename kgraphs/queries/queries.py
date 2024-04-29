WIKIPEDIA_URI = "<http://huginns.io/graph/wikipedia-links_lang=en>"

import random
from typing import List

from SPARQLWrapper import JSON, SPARQLWrapper

from ..utils.logging import MAIN_LOGGER_NAME, create_logger


def sample_fuseki_db(
    sparql: SPARQLWrapper,
    sampling_amount: int,
    random_sample: bool = False,
    db_uri: str = WIKIPEDIA_URI,
) -> List:
    """
    Given a db name will return a sparql query that samples the db
    """
    limit_expression = f"LIMIT {sampling_amount}" if random_sample == False else ""
    query = f"""
    SELECT DISTINCT ?subject ?predicate ?object
    WHERE {{
        GRAPH {db_uri} {{
            ?subject ?predicate ?object .
        }}
    }}
    {limit_expression}
    """
    logger = create_logger(MAIN_LOGGER_NAME)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    triplets = [
        result["object"]["value"]  # type: ignore
        for result in results["results"]["bindings"]  # type: ignore
    ]
    if random_sample:
        if len(triplets) < sampling_amount:
            logger.error(
                f"Amount of triplets found is less than the sampling amount {len(triplets)}"
            )
            raise ValueError(
                f"Amount of triplets found ({len(triplets)}) is less than the sampling amount"
            )
        random_samples = random.sample(triplets, sampling_amount)
    else:
        random_samples = triplets

    return random_samples


def dburi_to_wikiuri(dburi: str) -> str:
    """
    Given a particular dbpedia uri will try to obtain a wikipedia uri
    """
    # TODO: sinc we are sampling from the wikidb likely wont need this.
    pass


def get_avaiable_graphs():
    """
    This one is pretty slow
    """
    return "SELECT DISTINCT ?g WHERE { GRAPH ?g { ?s ?p ?o } }"
