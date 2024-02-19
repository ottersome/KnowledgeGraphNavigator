"""
Scripting for loading the data
"""
import os

from rdflib import Graph

# Load DBPedia and Wikipedia Together


def explore_ttl_data(meep):
    """
    Use Parql
    """
    pass


if __name__ == "__main__":
    # Take args
    import sys

    args = sys.argv
    if len(args) < 2:
        raise ValueError("Please provide the path to the data")
    file_path = args[1]
    cwd = os.getcwd()
    full_path = os.path.join(cwd, file_path)
    if not os.path.exists(full_path) or not file_path.endswith("ttl"):
        raise ValueError("Provided does not exist or is incorrect")

    graph = Graph()
    graph.parse(full_path, format="ttl")

    qres = graph.query(
        """ SELECT DISTINCT
           ?subject ?predicate ?object
           WHERE {
           ?subject ?predicate ?object .
           } LIMIT 10
    """
    )
    for row in qres:
        print(f"S: {row.subject} P: {row.predicate} O: {row.object}")  # type:ignore

    print("Loading file: ")
