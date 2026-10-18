"""Production sorgularında kullanılan dense ve hibrit arama stratejilerini içerir.

Sözlük:
- `Retriever`: Arama sınıflarının uyması gereken küçük protokol.
- `DenseRetriever`: Sorgu embedding'iyle Qdrant üzerinde anlamsal arama yapar.
- `HybridRetriever`: Dense ve BM25 sonuçlarını reciprocal-rank fusion ile birleştirir.
"""

from typing import Protocol

from astrarag.embedding import EmbeddingEncoder
from astrarag.indexes import DenseVectorIndex
from astrarag.models import RetrievalResult


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]: ...


class DenseRetriever:
    def __init__(self, encoder: EmbeddingEncoder, index: DenseVectorIndex) -> None:
        self.encoder = encoder
        self.index = index

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        return self.index.search(self.encoder.encode_query(query), limit=top_k)


class HybridRetriever:
    def __init__(
        self,
        dense: Retriever,
        sparse: Retriever,
        rrf_k: int = 60,
        candidate_k: int = 20,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than 0")
        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than 0")
        self.dense = dense
        self.sparse = sparse
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        candidate_k = max(top_k, self.candidate_k)
        rankings = (
            self.dense.retrieve(query, candidate_k),
            self.sparse.retrieve(query, candidate_k),
        )
        scores: dict[str, float] = {}
        results: dict[str, RetrievalResult] = {}

        for ranking in rankings:
            for rank, result in enumerate(ranking, start=1):
                results[result.chunk_id] = result
                scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (
                    self.rrf_k + rank
                )

        ranked_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
        return [
            results[chunk_id].model_copy(update={"score": scores[chunk_id]})
            for chunk_id in ranked_ids
        ]
