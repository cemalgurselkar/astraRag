"""Retrieval veri setini, ground-truth eşleşmesini ve kalite metriklerini yönetir.

Sözlük:
- `load_evaluation_dataset`: JSON değerlendirme sorgularını Pydantic ile doğrular.
- `recall_at_k`, `reciprocal_rank`, `ndcg_at_k`: Sıralama kalitesi metrikleri.
- `resolve_relevant_chunk_ids`: Kanıt metinlerini gerçek chunk kimlikleriyle eşleştirir.
- `resolve_dataset_ground_truth`: Tüm veri setinin eşleşmelerini tek seferde çözer.
- `is_relevant_result`: Bir retrieval sonucunun beklenen kanıtla uyuşmasını denetler.
- `_normalize_text`: Metin eşleştirmesi için yazıyı standartlaştırır.
"""

import json
import math
import re
import unicodedata
from collections.abc import Sequence
from pathlib import Path

from pydantic import TypeAdapter

from astrarag.models import Chunk, EvaluationQuery, RetrievalResult


def load_evaluation_dataset(path: Path) -> list[EvaluationQuery]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return TypeAdapter(list[EvaluationQuery]).validate_python(data)


def recall_at_k(
    retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than 0")
    if not relevant_ids:
        raise ValueError("relevant_ids cannot be empty")
    return len(set(retrieved_ids[:k]) & set(relevant_ids)) / len(set(relevant_ids))


def reciprocal_rank(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("relevant_ids cannot be empty")
    return next(
        (1.0 / rank for rank, item in enumerate(retrieved_ids, 1) if item in relevant),
        0.0,
    )


def ndcg_at_k(
    retrieved_ids: Sequence[str], relevant_ids: Sequence[str], k: int
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than 0")
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("relevant_ids cannot be empty")

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(retrieved_ids[:k], 1)
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg


def resolve_relevant_chunk_ids(
    query: EvaluationQuery,
    chunks: Sequence[Chunk],
) -> list[str]:
    relevant_ids: set[str] = set()
    for evidence in query.evidence:
        document_chunks = [
            chunk
            for chunk in chunks
            if chunk.metadata.get("filename") == evidence.document
        ]
        if not document_chunks:
            raise ValueError(f"Document not found: {evidence.document}")

        normalized_evidence = _normalize_text(evidence.text)
        matches = [
            chunk
            for chunk in document_chunks
            if normalized_evidence in _normalize_text(chunk.text)
        ]
        if not matches:
            raise ValueError(f"{query.id}: {evidence.document} -> {evidence.text!r}")
        relevant_ids.update(chunk.id for chunk in matches)
    return sorted(relevant_ids)


def resolve_dataset_ground_truth(
    queries: Sequence[EvaluationQuery],
    chunks: Sequence[Chunk],
) -> dict[str, list[str]]:
    ground_truth: dict[str, list[str]] = {}
    errors: list[str] = []
    for query in queries:
        try:
            ground_truth[query.id] = resolve_relevant_chunk_ids(query, chunks)
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        details = "\n".join(f"  - {error}" for error in errors)
        raise ValueError(
            f"Evaluation dataset contains {len(errors)} unresolved queries:\n{details}"
        )
    return ground_truth


def is_relevant_result(result: RetrievalResult, query: EvaluationQuery) -> bool:
    result_text = _normalize_text(result.text)
    filename = result.metadata.get("filename")
    return any(
        filename == evidence.document and _normalize_text(evidence.text) in result_text
        for evidence in query.evidence
    )


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("\ufffe", "").replace("\ufffd", "").replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-(?=\w)", "", text)
    return " ".join(re.sub(r"[^\w]+", " ", text).split())
