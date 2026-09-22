import re
from collections.abc import Sequence

from rank_bm25 import BM25Okapi
from astrarag.schemas import Chunk, RetrievalResult

class BM25Index:
    def __init__(self) -> None:
        
        self._chunk: list[Chunk] = []
        self._index: BM25Okapi | None = None
    
    def build(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            raise ValueError("chunks cannot be empty")
        
        self._chunk = list(chunks)
        
        tokenized_corpus = [
            self._tokenize(chunk.text)
            for chunk in self._chunk
        ]
        
        self._index = BM25Okapi(tokenized_corpus)
    
    def search(self, query: str, limit: int= 5) -> list[RetrievalResult]:
        
        if self._index is None:
            raise ValueError("BM25 index has not been built")
        
        if not query.strip():
            raise ValueError("query cannot be empty")
        
        if limit <= 0:
            raise ValueError("limit must be greater than 0")
        
        scores = self._index.get_scores(self._tokenize(query))
        
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:limit]
        
        return [
            RetrievalResult(
                chunk_id=self._chunk[index].id,
                document_id=self._chunk[index].document_id,
                text=self._chunk[index].text,
                score=float(scores[index]),
                page_numbers=self._chunk[index].page_numbers,
                metadata=self._chunk[index].metadata,
            ) for index in ranked_indices]
    
    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())