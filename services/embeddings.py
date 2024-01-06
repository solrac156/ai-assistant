import os
from typing import List

from langchain_community.embeddings.openai import OpenAIEmbeddings


class Embedder:
    def __init__(self):
        self._api_key = os.environ.get("OPENAI_API_KEY")
        self._model = os.environ.get("OPENAI_MODEL")
        self._client = OpenAIEmbeddings(
            openai_api_key=self._api_key, deployment=self._model, model=self._model
        )

    def embed(self, text: str) -> List[float]:
        text = text.replace("\n", " ")
        return self._client.embed_query(text)
