"""
Scripting for loading the data
"""
import os

from SPARQLWrapper import JSON, SPARQLWrapper

# Load DBPedia and Wikipedia Together

# Define the Fuseki server endpoint


def explore_ttl_data(endpoint: str):
    """
    Use Parql
    """
    sparql = SPARQLWrapper(fuseki_endpoint)
    # Define your SPARQL query
    query = """
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    SELECT ?subject ?predicate ?object
    WHERE {
      ?subject ?predicate ?object
    }
    LIMIT 10
    """
    # Set the query to the SPARQLWrapper
    sparql.setQuery(query)
    # Select the return format (e.g., JSON or XML)
    sparql.setReturnFormat(JSON)
    # Execute the query and convert the response to a Python dictionary
    try:
        response = sparql.query().convert()
        for result in response["results"]["bindings"]:  # type:ignore
            print(result)
    except Exception as e:
        print(f"An error occurred: {e}")


def explore_local(
    file_path: str,
):
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


if __name__ == "__main__":
    # Take args
    # Take the
    fuseki_endpoint = "http://localhost:3030/wiki_tdb/query"

    explore_ttl_data(fuseki_endpoint)
