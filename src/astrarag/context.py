"""Build bounded generation context from retrieval results."""

from collections.abc import Sequence

from astrarag.models import ContextItem, ContextPackage, RetrievalResult


def build_context(
    results: Sequence[RetrievalResult],
    max_characters: int = 12_000,
    max_items: int = 8,
) -> ContextPackage:
    if max_characters <= 0:
        raise ValueError("max_characters must be greater than 0")
    if max_items <= 0:
        raise ValueError("max_items must be greater than 0")

    selected: list[ContextItem] = []
    seen_texts: set[str] = set()
    used_characters = 0

    for result in results:
        if len(selected) >= max_items:
            break

        normalized = " ".join(result.text.casefold().split())
        if not normalized or normalized in seen_texts:
            continue

        remaining = max_characters - used_characters
        if remaining <= 0:
            break

        text = result.text.strip()[:remaining].rstrip()
        if not text:
            continue

        selected.append(
            ContextItem(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                text=text,
                score=result.score,
                page_numbers=result.page_numbers,
                metadata=result.metadata,
            )
        )
        seen_texts.add(normalized)
        used_characters += len(text)

    return ContextPackage(
        items=selected,
        text=_format_context(selected),
        total_characters=used_characters,
    )


def _format_context(items: Sequence[ContextItem]) -> str:
    sections: list[str] = []
    for index, item in enumerate(items, start=1):
        filename = item.metadata.get("filename", item.document_id)
        pages = ", ".join(str(page) for page in item.page_numbers) or "unknown"
        sections.append(
            f"[SOURCE {index}]\n"
            f"Document: {filename}\n"
            f"Pages: {pages}\n"
            f"Chunk ID: {item.chunk_id}\n"
            f"Content:\n{item.text}"
        )
    return "\n\n".join(sections)
