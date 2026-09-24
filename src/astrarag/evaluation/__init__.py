"""Export dataset, ground-truth, and retrieval evaluation utilities."""

from astrarag.evaluation.dataset import load_evaluation_dataset
from astrarag.evaluation.ground_truth import resolve_relevant_chunk_ids
from astrarag.evaluation.retrieval import (
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "load_evaluation_dataset",
    "ndcg_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "resolve_relevant_chunk_ids",
]
