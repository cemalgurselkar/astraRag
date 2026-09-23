import math
from collections.abc import Sequence


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if not relevant_ids:
        raise ValueError("relevant_ids cannot be empty")

    retrieved_at_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)

    return len(retrieved_at_k & relevant) / len(relevant)


def reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
) -> float:
    relevant = set(relevant_ids)

    if not relevant:
        raise ValueError("relevant_ids cannot be empty")

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if not relevant_ids:
        raise ValueError("relevant_ids cannot be empty")

    relevant = set(relevant_ids)

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved_ids[:k], start=1)
        if chunk_id in relevant
    )

    ideal_hits = min(len(relevant), k)

    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg
