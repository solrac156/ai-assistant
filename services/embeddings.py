import os
from typing import List

# from openai import AzureOpenAI
from openai import OpenAI


class Embedder:
    def __init__(self):
        # todo check if I will need these env vars or if I need to change
        #  them since I'm not using Azure OpenAI
        self._api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        self._endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        self._model = os.environ.get("OPENAI_MODEL")
        # self._client = AzureOpenAI(
        #     api_key=self._api_key,
        #     api_version="2023-05-15",
        #     azure_endpoint=self._endpoint,
        # )
        # todo finish adding parameters to client
        self._client = OpenAI()

    def embed(self, text: str) -> List[float]:
        text = text.replace("\n", " ")
        return (
            self._client.embeddings.create(
                input=[text],
                model=self._model,
            )
            .data[0]
            .embedding
        )
