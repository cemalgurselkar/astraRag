"""Verify production retrieval composition and score propagation."""

from astrarag.indexes import BM25Index
from astrarag.models import Chunk, RetrievalResult
from astrarag.retrieval import HybridRetriever


class FakeRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        return self.results[:top_k]


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
