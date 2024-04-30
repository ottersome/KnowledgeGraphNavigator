"""
So far two main embedding models will be consider. 
One which will be used for the testing using chatgpt
There other will be the one that we might want to train ourselves
"""
import json
import os

import requests
from ABC import ABC, abstractmethod
from torch.nn import Module as TModule

from ..utils.logging import MAIN_LOGGER_NAME, create_logger


class DocEmbedder(ABC):
    def get_embedding(self: str) -> TModule:  # type:ignore
        pass


class OpenAIEmbedder(DocEmbedder):
    def __init__(self, openai_api_key):
        self.logger = create_logger(MAIN_LOGGER_NAME)
        self.openai_api_key = openai_api_key


    @abstractmethod
    def get_embedding(self: str) -> TModule:
        """
        Something like curl_inspiration_cmd above
        """
        requestor = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"
                Bearer {self.openai_api_key}",
            },
            data=json.dumps(
                {
                    "input": self,
                    "model": "text-embedding-3-small",
                }
            ),
        )

        self.logger.debug("The entire request response is: ")
        return requestor.json()["data"]["embedding"]
