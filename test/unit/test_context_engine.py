"""Verify basic context assembly, deduplication, and size constraints."""

from astrarag.context import ContextEngine
from astrarag.schemas import RetrievalResult


def make_result(
    chunk_id: str,
    text: str,
    score: float = 1.0,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=text,
        score=score,
        page_numbers=[1],
        metadata={
            "filename": "paper.pdf",
        },
    )


def test_context_engine_builds_context() -> None:
    engine = ContextEngine(
        max_char=1000,
        max_items=3,
    )

    results = [
        make_result("chunk-1", "Attention mechanism."),
        make_result("chunk-2", "Transformer architecture."),
    ]

    context = engine.build(results)

    assert len(context.items) == 2
    assert "Attention mechanism." in context.text
    assert "Transformer architecture." in context.text
    assert "[SOURCE 1]" in context.text
    assert "[SOURCE 2]" in context.text


def test_context_engine_removes_exact_duplicates() -> None:
    engine = ContextEngine()

    results = [
        make_result("chunk-1", "Attention mechanism."),
        make_result("chunk-2", "  attention   mechanism.  "),
    ]

    context = engine.build(results)

    assert len(context.items) == 1


def test_context_engine_respects_item_limit() -> None:
    engine = ContextEngine(
        max_items=2,
    )

    results = [
        make_result("chunk-1", "First"),
        make_result("chunk-2", "Second"),
        make_result("chunk-3", "Third"),
    ]

    context = engine.build(results)

    assert len(context.items) == 2


def test_context_engine_respects_character_budget() -> None:
    engine = ContextEngine(
        max_char=10,
        max_items=5,
    )

    results = [
        make_result(
            "chunk-1",
            "abcdefghijklmnopqrstuvwxyz",
        )
    ]

    context = engine.build(results)

    assert len(context.items) == 1
    assert context.items[0].text == "abcdefghij"
    assert context.total_characters == 10
