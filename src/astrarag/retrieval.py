"""Production retrieval strategies."""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from sentence_transformers import CrossEncoder

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


class CrossEncoderReranker:
    """Own a local cross-encoder and score query-passage pairs."""

    def __init__(
        self,
        model_name: str | Path = "models/ms-marco-MiniLM-L6-v2",
    ) -> None:
        self.model_name = model_name
        self._model = CrossEncoder(str(model_name), local_files_only=True)

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if not candidates:
            return []

        scores = self._model.predict(
            [(query, candidate.text) for candidate in candidates],
            show_progress_bar=False,
        )
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [
            candidate.model_copy(update={"score": float(score)})
            for candidate, score in ranked[:top_k]
        ]


class RerankedRetriever:
    def __init__(
        self,
        base: Retriever,
        reranker: CrossEncoderReranker,
        candidate_k: int = 20,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than 0")
        self.base = base
        self.reranker = reranker
        self.candidate_k = candidate_k

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        candidates = self.base.retrieve(query, max(self.candidate_k, top_k))
        return self.reranker.rerank(query, candidates, top_k)
