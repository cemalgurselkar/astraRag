"""Resolve evidence annotations to chunks and judge retrieved-result relevance."""

import re
import unicodedata
from collections.abc import Sequence

from astrarag.schemas import Chunk, EvaluationQuery, RetrievalResult


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()

    text = text.replace("\ufffe", "")
    text = text.replace("\ufffd", "")
    text = text.replace("\u00ad", "")

    # Treat punctuation/hyphenation consistently.
    text = re.sub(r"(?<=\w)-(?=\w)", "", text)
    text = re.sub(r"[^\w]+", " ", text)

    return " ".join(text.split())


def resolve_relevant_chunk_ids(
    query: EvaluationQuery,
    chunks: Sequence[Chunk],
) -> list[str]:
    relevant_ids: set[str] = set()

    for evidence in query.evidence:
        normalized_evidence = _normalize_text(evidence.text)

        document_chunks = [
            chunk
            for chunk in chunks
            if chunk.metadata.get("filename") == evidence.document
        ]

        if not document_chunks:
            raise ValueError(f"Document not found: {evidence.document}")

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
            ground_truth[query.id] = resolve_relevant_chunk_ids(
                query=query,
                chunks=chunks,
            )
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

    for evidence in query.evidence:
        if filename != evidence.document:
            continue

        evidence_text = _normalize_text(evidence.text)

        if evidence_text in result_text:
            return True

    return False
