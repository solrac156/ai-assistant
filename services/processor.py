import os
from typing import List

from langchain.docstore.document import Document

from services.chunker import Chunker
from services.keywords import KeywordsExtractor
from services.vector_store import VectorStore


class DocumentProcessor:
    def __init__(self):
        self._chunker = Chunker()
        self._keywords_extractor = KeywordsExtractor()
        self._vector_store = VectorStore()
        self._input_path = os.environ.get("DATA_INPUT_PATH")

    def process(self, filename: str) -> List[str]:
        with open(os.path.join(self._input_path, filename)) as f:
            text = f.readlines()
        text = "".join(text)
        documents = self._chunker.chunk(text)
        ids = []
        for doc in documents:
            keywords = self._keywords_extractor.extract(doc.page_content)
            document = self._create_document(doc, keywords)
            id_ = self._vector_store.add(document)
            ids.append(id_)
        return ids

    def _create_document(self, chunk, keywords) -> Document:
        wine_name = chunk.metadata.get("Wine Name")
        metadata = {
            "name": wine_name,
            "category": chunk.metadata.get("Category", ""),
            "subcategory": chunk.metadata.get("Sub-Category", ""),
            "keywords": keywords,
        }
        return Document(page_content=chunk.page_content, metadata=metadata)
