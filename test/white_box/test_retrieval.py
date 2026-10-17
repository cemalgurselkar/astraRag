"""Verify production retrieval composition and score propagation."""

import numpy as np
import pytest

from astrarag.indexes import BM25Index
from astrarag.models import Chunk, RetrievalResult
from astrarag.retrieval import CrossEncoderReranker, HybridRetriever


class FakeRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        return self.results[:top_k]


class FakeCrossEncoder:
    def predict(self, pairs, show_progress_bar: bool = False) -> np.ndarray:
        return np.asarray([0.1, 0.9], dtype=np.float32)


def result(chunk_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id="document",
        text=f"text-{chunk_id}",
        score=score,
    )


def test_bm25_index_is_directly_retrievable() -> None:
    index = BM25Index(
        [
            Chunk(id="a", document_id="doc", text="dense retrieval", index=0),
            Chunk(id="b", document_id="doc", text="language model", index=1),
        ]
    )

    assert index.retrieve("dense", top_k=1)[0].chunk_id == "a"


def test_hybrid_fuses_duplicate_results() -> None:
    hybrid = HybridRetriever(
        dense=FakeRetriever([result("shared", 0.9), result("dense", 0.8)]),
        sparse=FakeRetriever([result("shared", 4.0), result("sparse", 3.0)]),
        candidate_k=2,
    )

    results = hybrid.retrieve("query", top_k=2)

    assert results[0].chunk_id == "shared"
    assert results[0].score > results[1].score


def test_cross_encoder_replaces_the_public_score() -> None:
    reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
    reranker._model = FakeCrossEncoder()

    results = reranker.rerank(
        "query",
        [result("first", 10.0), result("second", 1.0)],
        top_k=2,
    )

    assert [item.chunk_id for item in results] == ["second", "first"]
    assert results[0].score == pytest.approx(0.9)
    assert "scores" not in results[0].model_dump()
