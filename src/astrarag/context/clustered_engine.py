"""Build compact contexts by clustering semantically similar retrieval results."""

from collections.abc import Sequence

import numpy as np

from astrarag.embedding import EmbeddingEncoder
from astrarag.schemas import ContextItem, ContextPackage, RetrievalResult


class ClusteredContextEngine:
    def __init__(
        self,
        encoder: EmbeddingEncoder,
        max_char: int,
        max_items: int,
        similarity_threshold: float = 0.90,
    ) -> None:
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0 and 1.")

        self.encoder = encoder
        self.max_char = max_char
        self.max_items = max_items
        self.similarity_threshold = similarity_threshold

    def build(
        self,
        results: Sequence[RetrievalResult],
    ) -> ContextPackage:
        if not results:
            return ContextPackage(
                items=[],
                text="",
                total_characters=0,
            )

        representatives = self._select_representatives(results)

        selected: list[ContextItem] = []
        used_chars = 0

        for result in representatives:
            if len(selected) >= self.max_items:
                break

            remaining = self.max_char - used_chars
            if remaining <= 0:
                break

            text = result.text.strip()
            if len(text) > remaining:
                text = text[:remaining].rstrip()

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
            used_chars += len(text)

        return ContextPackage(
            items=selected,
            text=self._format_context(selected),
            total_characters=used_chars,
        )

    def _select_representatives(
        self,
        results: Sequence[RetrievalResult],
    ) -> list[RetrievalResult]:
        ordered = sorted(
            results,
            key=lambda result: result.score,
            reverse=True,
        )

        embeddings = np.asarray(
            self.encoder.encode([result.text for result in ordered]),
            dtype=np.float32,
        )

        representatives: list[RetrievalResult] = []
        representative_embeddings: list[np.ndarray] = []

        for result, embedding in zip(ordered, embeddings, strict=True):
            if representative_embeddings:
                similarities = np.dot(
                    np.stack(representative_embeddings),
                    embedding,
                )

                if np.max(similarities) >= self.similarity_threshold:
                    continue

            representatives.append(result)
            representative_embeddings.append(embedding)

        return representatives

    @staticmethod
    def _format_context(items: Sequence[ContextItem]) -> str:
        blocks: list[str] = []

        for index, item in enumerate(items, start=1):
            filename = item.metadata.get("filename", item.document_id)
            pages = ", ".join(str(page) for page in item.page_numbers)

            source = f"[Source {index}] {filename}"
            if pages:
                source += f" | pages: {pages}"

            blocks.append(f"{source}\n{item.text}")

        return "\n\n".join(blocks)
