import requests
from bs4 import BeautifulSoup

from ..utils.logging import MAIN_LOGGER_NAME, create_logger


def get_wikipedia_content(url: str) -> str:
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
