"""Select relevant, diverse context items with maximal marginal relevance."""

import numpy as np

from astrarag.embedding import EmbeddingEncoder
from astrarag.schemas.context import ContextItem, ContextPackage
from astrarag.schemas.retrieval import RetrievalResult


class MMRContextEngine:
    """
    Select context using Maximal Marginal Relevance.

    Balances retrieval relevance against semantic redundancy.
    """

    def __init__(
        self,
        encoder: EmbeddingEncoder,
        max_char: int,
        max_items: int,
        lambda_mult: float = 0.7,
    ) -> None:
        if max_char <= 0:
            raise ValueError("max_char must be greater than 0.")

        if max_items <= 0:
            raise ValueError("max_items must be greater than 0.")

        if not 0.0 <= lambda_mult <= 1.0:
            raise ValueError(
                "lambda_mult must be between 0.0 and 1.0."
            )

        self.encoder = encoder
        self.max_char = max_char
        self.max_items = max_items
        self.lambda_mult = lambda_mult

    def build(
        self,
        results: list[RetrievalResult],
    ) -> ContextPackage:
        if not results:
            return ContextPackage(
                items=[],
                text="",
                total_characters=0,
            )

        selected = self._select(results)

        items: list[ContextItem] = []
        used_chars = 0

        for result in selected:
            if len(items) >= self.max_items:
                break

            if used_chars + len(result.text) > self.max_char:
                continue

            items.append(
                ContextItem(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    text=result.text,
                    score=result.score,
                    page_numbers=result.page_numbers,
                    metadata=result.metadata,
                )
            )

            used_chars += len(result.text)

        text = self._format(items)

        return ContextPackage(
            items=items,
            text=text,
            total_characters=used_chars,
        )

    def _select(
        self,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        candidates = sorted(
            results,
            key=lambda result: result.score,
            reverse=True,
        )

        embeddings = np.asarray(
            self.encoder.encode(
                [result.text for result in candidates]
            ),
            dtype=np.float32,
        )

        # Retrieval scores from BM25/dense/hybrid can have
        # different scales. Normalize them before combining
        # relevance with cosine similarity.
        relevance = self._normalize_scores(
            np.asarray(
                [result.score for result in candidates],
                dtype=np.float32,
            )
        )

        selected_indices: list[int] = []
        remaining = set(range(len(candidates)))

        while remaining and len(selected_indices) < self.max_items:
            if not selected_indices:
                best_index = max(
                    remaining,
                    key=lambda index: float(relevance[index]),
                )
            else:
                best_index = max(
                    remaining,
                    key=lambda index: self._mmr_score(
                        index=index,
                        relevance=relevance,
                        embeddings=embeddings,
                        selected_indices=selected_indices,
                    ),
                )

            selected_indices.append(best_index)
            remaining.remove(best_index)

        return [
            candidates[index]
            for index in selected_indices
        ]

    def _mmr_score(
        self,
        index: int,
        relevance: np.ndarray,
        embeddings: np.ndarray,
        selected_indices: list[int],
    ) -> float:
        max_similarity = max(
            float(
                embeddings[index]
                @ embeddings[selected_index]
            )
            for selected_index in selected_indices
        )

        return (
            self.lambda_mult * float(relevance[index])
            - (1.0 - self.lambda_mult) * max_similarity
        )

    @staticmethod
    def _normalize_scores(
        scores: np.ndarray,
    ) -> np.ndarray:
        minimum = float(scores.min())
        maximum = float(scores.max())

        if maximum == minimum:
            return np.ones_like(scores)

        return (
            (scores - minimum)
            / (maximum - minimum)
        )

    @staticmethod
    def _format(
        items: list[ContextItem],
    ) -> str:
        sections = []

        for index, item in enumerate(items, start=1):
            pages = (
                ", ".join(
                    str(page)
                    for page in item.page_numbers
                )
                if item.page_numbers
                else "unknown"
            )

            sections.append(
                f"[Source {index}]\n"
                f"Document: {item.document_id}\n"
                f"Pages: {pages}\n"
                f"{item.text}"
            )

        return "\n\n".join(sections)
