import requests
from bs4 import BeautifulSoup

from ..utils.logging import MAIN_LOGGER_NAME, create_logger


def get_wikipedia_raw_content(url: str) -> str:
    logger = create_logger(MAIN_LOGGER_NAME)
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        # Raise an exception if the response status is not 200
        response.raise_for_status()

        # Parse the HTML content of the page using BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")

        # Find the main content div in the Wikipedia page
        content = soup.find("div", {"id": "mw-content-text"})

        # Extract and return the text from the content div
        if content:
            return content.get_text()
        else:
            return ""
    except requests.RequestException as e:
        logger.warn(
            f"An error occurred when trying to pull wikipedia article {url} information: {e}"
        )
        return ""


def get_wikipedia_json_content(url: str) -> dict:
    """
    Query the wikipedia json endpoing and return as dictionary
    """
    logger = create_logger(MAIN_LOGGER_NAME)
    try:
        page_id = url.split("/")[-1]
        url = f"https://en.wikipedia.org/w/api.php?action=parse&page={page_id}&format=json&prop=sections|wikitext"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.warn(
            f"An error occurred when trying to pull wikipedia article {url} information: {e}"
        )
        return {}
