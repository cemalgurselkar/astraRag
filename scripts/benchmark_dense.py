"""Evaluate the dense retrieval baseline with ranking-quality and latency metrics."""

from pathlib import Path
from statistics import mean
from time import perf_counter

from astrarag.embedding import EmbeddingEncoder
from astrarag.evaluation import (
    load_evaluation_dataset,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from astrarag.index import DenseVectorIndex
from astrarag.retrieval import DenseRetriever

TOP_K = 10


def main() -> None:
    encoder = EmbeddingEncoder()

    index = DenseVectorIndex(
        path=Path("data/qdrant"),
        collection_name="attention",
        dimension=encoder.dimension,
    )

    retriever = DenseRetriever(
        encoder=encoder,
        index=index,
    )

    evaluation_queries = load_evaluation_dataset(
        Path("data/eval/attention.json")
    )

    recalls_at_5: list[float] = []
    recalls_at_10: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs_at_10: list[float] = []
    latencies_ms: list[float] = []

    for evaluation_query in evaluation_queries:
        start = perf_counter()

        results = retriever.retriever(
            query=evaluation_query.query,
            top_k=TOP_K,
        )

        latency_ms = (perf_counter() - start) * 1000

        retrieved_ids = [
            result.chunk_id
            for result in results
        ]

        recall_5 = recall_at_k(
            retrieved_ids,
            evaluation_query.relevant_chunk_ids,
            k=5,
        )

        recall_10 = recall_at_k(
            retrieved_ids,
            evaluation_query.relevant_chunk_ids,
            k=10,
        )

        rr = reciprocal_rank(
            retrieved_ids,
            evaluation_query.relevant_chunk_ids,
        )

        ndcg_10 = ndcg_at_k(
            retrieved_ids,
            evaluation_query.relevant_chunk_ids,
            k=10,
        )

        recalls_at_5.append(recall_5)
        recalls_at_10.append(recall_10)
        reciprocal_ranks.append(rr)
        ndcgs_at_10.append(ndcg_10)
        latencies_ms.append(latency_ms)

        print(
            f"{evaluation_query.id:<15} "
            f"R@5={recall_5:.3f} "
            f"R@10={recall_10:.3f} "
            f"RR={rr:.3f} "
            f"nDCG@10={ndcg_10:.3f} "
            f"latency={latency_ms:.2f}ms"
        )

    print("\nDense Retrieval Baseline")
    print("-" * 50)
    print(f"Queries:    {len(evaluation_queries)}")
    print(f"Recall@5:   {mean(recalls_at_5):.4f}")
    print(f"Recall@10:  {mean(recalls_at_10):.4f}")
    print(f"MRR:        {mean(reciprocal_ranks):.4f}")
    print(f"nDCG@10:    {mean(ndcgs_at_10):.4f}")
    print(f"Mean latency: {mean(latencies_ms):.2f} ms")


if __name__ == "__main__":
    main()
