import os
from typing import List

from langchain_community.vectorstores.azuresearch import AzureSearch
from azure.search.documents.indexes.models import (
    SemanticSettings,
    SemanticConfiguration,
    SemanticField,
    PrioritizedFields,
)
from langchain_core.documents import Document

from services.embeddings import Embedder


class VectorStore:
    def __init__(self):
        self._embedder = Embedder()
        self._endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT")
        self._api_key = os.environ.get("AZURE_SEARCH_API_KEY")
        self._index = os.environ.get("AZURE_SEARCH_INDEX")
        self._client = AzureSearch(
            azure_search_endpoint=self._endpoint,
            azure_search_key=self._api_key,
            index_name=self._index,
            embedding_function=self._embedder.embed,
            semantic_configuration_name="config",
            semantic_settings=SemanticSettings(
                default_configuration="config",
                configurations=[
                    SemanticConfiguration(
                        name="config",
                        prioritized_fields=PrioritizedFields(
                            title_field=SemanticField(field_name="name"),
                            prioritized_content_fields=[
                                SemanticField(field_name="page_content")
                            ],
                            prioritized_keywords_fields=[
                                SemanticField(field_name="keywords")
                            ],
                        ),
                    )
                ],
            ),
        )

    def similarity_search(self, text: str, n: int = 3) -> List[Document]:
        docs = self._client.similarity_search(query=text, k=n, search_type="similarity")
        return docs

    def hybrid_search(self, text: str, n: int = 3) -> List[Document]:
        docs = self._client.similarity_search(query=text, k=n, search_type="hybrid")
        return docs

    def add(self, document: Document) -> str:
        return self._client.add_documents([document])[0]
