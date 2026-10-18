"""Benchmarklerde karşılaştırılan deneysel bağlam seçme yöntemlerini içerir.

Sözlük:
- `build_clustered_context`: Benzer sonuçları eleyerek temsilci parçalar seçer.
- `build_mmr_context`: İlgi ve çeşitliliği MMR puanıyla dengeler.
- `_package`: Seçilen sonuçları sınırlandırılmış `ContextPackage` haline getirir.
- `_normalize_scores`: Puanları karşılaştırılabilir 0–1 aralığına taşır.
"""

from collections.abc import Sequence

import numpy as np

from astrarag.embedding import EmbeddingEncoder
from astrarag.models import ContextItem, ContextPackage, RetrievalResult


def build_clustered_context(
    results: Sequence[RetrievalResult],
    encoder: EmbeddingEncoder,
    max_characters: int,
    max_items: int,
    similarity_threshold: float = 0.90,
) -> ContextPackage:
    if max_characters <= 0 or max_items <= 0:
        raise ValueError("context limits must be greater than 0")
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0 and 1")
    if not results:
        return ContextPackage(items=[], text="", total_characters=0)

    ordered = sorted(results, key=lambda result: result.score, reverse=True)
    embeddings = np.asarray(
        encoder.encode([result.text for result in ordered]), dtype=np.float32
    )
    representatives: list[RetrievalResult] = []
    representative_embeddings: list[np.ndarray] = []

    for result, embedding in zip(ordered, embeddings, strict=True):
        if representative_embeddings:
            similarities = np.dot(np.stack(representative_embeddings), embedding)
            if np.max(similarities) >= similarity_threshold:
                continue
        representatives.append(result)
        representative_embeddings.append(embedding)

    return _package(representatives, max_characters, max_items)


def build_mmr_context(
    results: Sequence[RetrievalResult],
    encoder: EmbeddingEncoder,
    max_characters: int,
    max_items: int,
    lambda_mult: float = 0.7,
) -> ContextPackage:
    if max_characters <= 0 or max_items <= 0:
        raise ValueError("context limits must be greater than 0")
    if not 0.0 <= lambda_mult <= 1.0:
        raise ValueError("lambda_mult must be between 0 and 1")
    if not results:
        return ContextPackage(items=[], text="", total_characters=0)

    candidates = sorted(results, key=lambda result: result.score, reverse=True)
    embeddings = np.asarray(
        encoder.encode([result.text for result in candidates]), dtype=np.float32
    )
    relevance = _normalize_scores(
        np.asarray([result.score for result in candidates], dtype=np.float32)
    )
    selected: list[int] = []
    remaining = set(range(len(candidates)))

    while remaining and len(selected) < max_items:
        if not selected:
            best = max(remaining, key=lambda index: float(relevance[index]))
        else:
            best = max(
                remaining,
                key=lambda index: (
                    lambda_mult * float(relevance[index])
                    - (1.0 - lambda_mult)
                    * max(
                        float(embeddings[index] @ embeddings[chosen])
                        for chosen in selected
                    )
                ),
            )
        selected.append(best)
        remaining.remove(best)

    return _package(
        [candidates[index] for index in selected], max_characters, max_items
    )


def _package(
    results: Sequence[RetrievalResult],
    max_characters: int,
    max_items: int,
) -> ContextPackage:
    items: list[ContextItem] = []
    used = 0
    for result in results:
        if len(items) >= max_items:
            break
        remaining = max_characters - used
        if remaining <= 0:
            break
        text = result.text.strip()[:remaining].rstrip()
        if not text:
            continue
        items.append(
            ContextItem(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                text=text,
                score=result.score,
                page_numbers=result.page_numbers,
                metadata=result.metadata,
            )
        )
        used += len(text)

    blocks = []
    for index, item in enumerate(items, start=1):
        filename = item.metadata.get("filename", item.document_id)
        pages = ", ".join(str(page) for page in item.page_numbers)
        source = f"[SOURCE {index}] {filename}"
        if pages:
            source += f" | pages: {pages}"
        blocks.append(f"{source}\n{item.text}")

    return ContextPackage(
        items=items,
        text="\n\n".join(blocks),
        total_characters=used,
    )


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    minimum = float(scores.min())
    maximum = float(scores.max())
    if maximum == minimum:
        return np.ones_like(scores)
    return (scores - minimum) / (maximum - minimum)
