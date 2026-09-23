from collections.abc import Sequence

from astrarag.schemas import ContextItem, ContextPackage, RetrievalResult


class ContextEngine:
    def __init__(self, max_char: int = 12_000, max_items: int = 8) -> None:

        if max_char <= 0:
            raise ValueError("max_char must be greater than 0")

        if max_items <= 0:
            raise ValueError("max_items must be greater than 0")

        self.max_char = max_char
        self.max_items = max_items

    def build(self, results: Sequence[RetrievalResult]) -> ContextPackage:

        selected: list[ContextItem] = []
        seen_texts: set[str] = set()
        used_chars = 0

        for result in results:
            if len(selected) >= self.max_items:
                break

            normalized_text = self._normalize_for_dedup(result.text)

            if not normalized_text:
                continue
            if normalized_text in seen_texts:
                continue

            remaining = self.max_char - used_chars

            if remaining <= 0:
                break

            text = result.text.strip()

            if len(text) > remaining:
                text = text[:remaining].rstrip()

            if not text:
                continue

            item = ContextItem(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                text=text,
                score=result.score,
                page_numbers=result.page_numbers,
                metadata=result.metadata,
            )

            selected.append(item)
            seen_texts.add(normalized_text)
            used_chars += len(text)

        context_text = self._format_context(selected)

        return ContextPackage(
            items=selected,
            text=context_text,
            total_characters=used_chars,
        )

    @staticmethod
    def _normalize_for_dedup(text: str) -> str:
        return " ".join(text.casefold().split())

    @staticmethod
    def _format_context(items: Sequence[ContextItem]) -> str:
        sections: list[str] = []

        for index, item in enumerate(items, start=1):
            filename = item.metadata.get("filename", item.document_id)

            pages = (
                ", ".join(str(page) for page in item.page_numbers)
                if item.page_numbers
                else "unkown"
            )

            sections.append(
                f"[SOURCE {index}]\n"
                f"Document: {filename}\n"
                f"Pages: {pages}\n"
                f"Chunk ID: {item.chunk_id}\n"
                f"Content:\n{item.text}"
            )

        return "\n\n".join(sections)
