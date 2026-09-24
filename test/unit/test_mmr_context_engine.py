import numpy as np

from astrarag.context import MMRContextEngine
from astrarag.schemas.retrieval import RetrievalResult


class FakeEncoder:
    def __init__(
        self,
        vectors: dict[str, list[float]],
    ) -> None:
        self.vectors = vectors

    def encode(
        self,
        texts: list[str],
    ) -> np.ndarray:
        return np.asarray(
            [self.vectors[text] for text in texts],
            dtype=np.float32,
        )


def result(
    chunk_id: str,
    text: str,
    score: float,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id="doc",
        text=text,
        score=score,
    )


def test_mmr_prefers_diverse_relevant_chunks() -> None:
    encoder = FakeEncoder(
        {
            "A": [1.0, 0.0],
            "A similar": [0.99, 0.01],
            "B": [0.0, 1.0],
        }
    )

    engine = MMRContextEngine(
        encoder=encoder,  # type: ignore[arg-type]
        max_char=10_000,
        max_items=2,
        lambda_mult=0.5,
    )

    context = engine.build(
        [
            result("a", "A", 1.0),
            result("a2", "A similar", 0.95),
            result("b", "B", 0.80),
        ]
    )

    assert [
        item.chunk_id
        for item in context.items
    ] == ["a", "b"]


def test_mmr_respects_item_limit() -> None:
    encoder = FakeEncoder(
        {
            "A": [1.0, 0.0],
            "B": [0.0, 1.0],
            "C": [-1.0, 0.0],
        }
    )

    engine = MMRContextEngine(
        encoder=encoder,  # type: ignore[arg-type]
        max_char=10_000,
        max_items=2,
    )

    context = engine.build(
        [
            result("a", "A", 1.0),
            result("b", "B", 0.9),
            result("c", "C", 0.8),
        ]
    )

    assert len(context.items) == 2


def test_mmr_respects_character_budget() -> None:
    encoder = FakeEncoder(
        {
            "AAAA": [1.0, 0.0],
            "BBBB": [0.0, 1.0],
        }
    )

    engine = MMRContextEngine(
        encoder=encoder,  # type: ignore[arg-type]
        max_char=4,
        max_items=2,
    )

    context = engine.build(
        [
            result("a", "AAAA", 1.0),
            result("b", "BBBB", 0.9),
        ]
    )

    assert len(context.items) == 1
    assert context.total_characters == 4


def test_mmr_handles_empty_results() -> None:
    encoder = FakeEncoder({})

    engine = MMRContextEngine(
        encoder=encoder,  # type: ignore[arg-type]
        max_char=10_000,
        max_items=10,
    )

    context = engine.build([])

    assert context.items == []
    assert context.text == ""
    assert context.total_characters == 0