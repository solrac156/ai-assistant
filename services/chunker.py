from typing import List

from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_core.documents import Document


class Chunker:
    def __init__(self):
        self._headers_to_split_on = [
            ("#", "Wine Name"),
            ("##", "Category"),
            ("###", "Sub-Category"),
        ]
        self._chunker = MarkdownHeaderTextSplitter(
            headers_to_split_on=self._headers_to_split_on
        )

    def chunk(self, text: str) -> List[Document]:
        return self._chunker.split_text(text)
