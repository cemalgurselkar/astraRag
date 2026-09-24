import numpy as np

from astrarag.context import ClusteredContextEngine
from astrarag.schemas import RetrievalResult


class FakeEncoder:
    def __init__(self, embeddings: dict[str, list[float]]) -> None:
        self.embeddings = embeddings

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            [self.embeddings[text] for text in texts],
            dtype=np.float32,
        )


def make_result(
    chunk_id: str,
    text: str,
    score: float,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=text,
        score=score,
        page_numbers=[1],
        metadata={"filename": "paper.pdf"},
    )


def test_clustered_engine_removes_semantically_similar_chunks() -> None:
    results = [
        make_result("chunk-1", "attention-a", 0.95),
        make_result("chunk-2", "attention-b", 0.90),
        make_result("chunk-3", "optimization", 0.80),
    ]

    encoder = FakeEncoder(
        {
            "attention-a": [1.0, 0.0],
            "attention-b": [0.99, 0.10],
            "optimization": [0.0, 1.0],
        }
    )

    engine = ClusteredContextEngine(
        encoder=encoder,  # type: ignore[arg-type]
        max_char=1000,
        max_items=10,
        similarity_threshold=0.90,
    )

    context = engine.build(results)

    assert [item.chunk_id for item in context.items] == [
        "chunk-1",
        "chunk-3",
    ]


def test_clustered_engine_keeps_highest_scoring_representative() -> None:
    results = [
        make_result("low", "similar-low", 0.50),
        make_result("high", "similar-high", 0.95),
    ]

    encoder = FakeEncoder(
        {
            "similar-low": [1.0, 0.0],
            "similar-high": [1.0, 0.0],
        }
    )

    engine = ClusteredContextEngine(
        encoder=encoder,  # type: ignore[arg-type]
        max_char=1000,
        max_items=10,
        similarity_threshold=0.90,
    )

    context = engine.build(results)

    assert len(context.items) == 1
    assert context.items[0].chunk_id == "high"


def test_clustered_engine_keeps_diverse_chunks() -> None:
    results = [
        make_result("chunk-1", "dense retrieval", 0.95),
        make_result("chunk-2", "cross encoder", 0.90),
    ]

    encoder = FakeEncoder(
        {
            "dense retrieval": [1.0, 0.0],
            "cross encoder": [0.0, 1.0],
        }
    )

    engine = ClusteredContextEngine(
        encoder=encoder,  # type: ignore[arg-type]
        max_char=1000,
        max_items=10,
        similarity_threshold=0.90,
    )

    context = engine.build(results)

    assert len(context.items) == 2


def test_clustered_engine_respects_item_limit() -> None:
    results = [
        make_result("chunk-1", "one", 0.9),
        make_result("chunk-2", "two", 0.8),
        make_result("chunk-3", "three", 0.7),
    ]

    encoder = FakeEncoder(
        {
            "one": [1.0, 0.0, 0.0],
            "two": [0.0, 1.0, 0.0],
            "three": [0.0, 0.0, 1.0],
        }
    )

    engine = ClusteredContextEngine(
        encoder=encoder,  # type: ignore[arg-type]
        max_char=1000,
        max_items=2,
        similarity_threshold=0.90,
    )

    context = engine.build(results)

    assert len(context.items) == 2


def test_clustered_engine_handles_empty_results() -> None:
    encoder = FakeEncoder({})

    engine = ClusteredContextEngine(
        encoder=encoder,  # type: ignore[arg-type]
        max_char=1000,
        max_items=10,
    )

    context = engine.build([])

    assert context.items == []
    assert context.text == ""
    assert context.total_characters == 0