"""Adapt the BM25 index to the common retriever interface."""

from astrarag.index import BM25Index
from astrarag.schemas import RetrievalResult


class BM25Retriever:
    def __init__(self, index: BM25Index) -> None:
        self.index = index

    def retriever(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        return self.index.search(
            query=query,
            limit=top_k,
        )
