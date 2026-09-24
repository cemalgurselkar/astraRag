"""Compare BM25, dense, hybrid, hierarchical, and reranked retrieval strategies."""

import math
from pathlib import Path
from statistics import mean
from time import perf_counter

from qdrant_client import QdrantClient

from astrarag.embedding import EmbeddingEncoder
from astrarag.evaluation import load_evaluation_dataset
from astrarag.evaluation.ground_truth import (
    is_relevant_result,
    resolve_dataset_ground_truth,
)
from astrarag.index import BM25Index, DenseVectorIndex
from astrarag.reranking import CrossEncoderReRanker
from astrarag.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    ParentChildRetriever,
    RerankedRetriever,
)

TOP_K = 10


def hit_at_k(
    relevance: list[bool],
    k: int,
) -> float:
    """Return 1 if at least one relevant result exists in the top-k."""

    return float(any(relevance[:k]))


def reciprocal_rank_from_relevance(
    relevance: list[bool],
) -> float:
    """Return reciprocal rank of the first relevant result."""

    for rank, relevant in enumerate(relevance, start=1):
        if relevant:
            return 1.0 / rank

    return 0.0


def ndcg_from_relevance(
    relevance: list[bool],
    k: int,
) -> float:
    """Compute nDCG@k from binary relevance labels."""

    gains = relevance[:k]

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, relevant in enumerate(gains, start=1)
        if relevant
    )

    relevant_count = sum(relevance)

    if relevant_count == 0:
        return 0.0

    ideal_count = min(relevant_count, k)

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )

    return dcg / idcg


def evaluate(
    name,
    retriever,
    queries,
):
    hit_5_scores = []
    hit_10_scores = []
    rr_scores = []
    ndcg_scores = []
    latencies = []

    for query in queries:
        start = perf_counter()

        results = retriever.retriever(
            query=query.query,
            top_k=TOP_K,
        )

        latency_ms = (perf_counter() - start) * 1000

        relevance = [
            is_relevant_result(result, query)
            for result in results
        ]

        hit_5_scores.append(
            hit_at_k(
                relevance,
                k=5,
            )
        )

        hit_10_scores.append(
            hit_at_k(
                relevance,
                k=10,
            )
        )

        rr_scores.append(
            reciprocal_rank_from_relevance(relevance)
        )

        ndcg_scores.append(
            ndcg_from_relevance(
                relevance,
                k=10,
            )
        )

        latencies.append(latency_ms)

    return {
        "name": name,
        "hit_5": mean(hit_5_scores),
        "hit_10": mean(hit_10_scores),
        "mrr": mean(rr_scores),
        "ndcg_10": mean(ndcg_scores),
        "latency_ms": mean(latencies),
    }


def main() -> None:
    encoder = EmbeddingEncoder()

    qdrant_client = QdrantClient(
            path="data/qdrant"
    )

    dense_index = DenseVectorIndex(
        path=Path("data/qdrant"),
        collection_name="astrarag-corpus",
        dimension=encoder.dimension,
        client=qdrant_client
    )

    chunks = dense_index.get_chunks()

    print(f"Benchmark corpus: {len(chunks)} chunks")

    dense = DenseRetriever(
        encoder=encoder,
        index=dense_index,
    )

    bm25_index = BM25Index()
    bm25_index.build(chunks)

    bm25 = BM25Retriever(
        index=bm25_index,
    )

    hybrid = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=bm25,
    )

    reranker = CrossEncoderReRanker()

    hybrid_reranked = RerankedRetriever(
        base_retriever=hybrid,
        reranker=reranker,
        candidate_k=20,
    )

    parent_child_index = DenseVectorIndex(
        path=Path("data/qdrant"),
        collection_name="astrarag-parent-child",
        dimension=encoder.dimension,
        client=qdrant_client
    )

    parent_child_dense = DenseRetriever(
        encoder=encoder,
        index=parent_child_index,
    )

    parent_child = ParentChildRetriever(
        child_retriever=parent_child_dense,
        candidate_k=20,
    )

    queries = load_evaluation_dataset(
        Path("data/eval/retrieval_v1.json")
    )

    print(f"Evaluation queries: {len(queries)}")
    print("Validating evaluation ground truth...")

    ground_truth = resolve_dataset_ground_truth(
        queries=queries,
        chunks=chunks,
    )

    print(
        f"Ground truth validated: "
        f"{len(ground_truth)}/{len(queries)} queries"
    )

    experiments = [
        ("Dense", dense),
        ("BM25", bm25),
        ("Hybrid RRF", hybrid),
        ("Hybrid + CE", hybrid_reranked),
        ("Parent-Child", parent_child),
    ]

    results = [
        evaluate(
            name=name,
            retriever=retriever,
            queries=queries,
        )
        for name, retriever in experiments
    ]

    print("\nRetrieval Benchmark")
    print("=" * 90)

    print(
        f"{'Strategy':<18}"
        f"{'Hit@5':>10}"
        f"{'Hit@10':>10}"
        f"{'MRR':>10}"
        f"{'nDCG@10':>12}"
        f"{'Latency':>14}"
    )

    print("-" * 90)

    for result in results:
        print(
            f"{result['name']:<18}"
            f"{result['hit_5']:>10.3f}"
            f"{result['hit_10']:>10.3f}"
            f"{result['mrr']:>10.3f}"
            f"{result['ndcg_10']:>12.3f}"
            f"{result['latency_ms']:>11.2f} ms"
        )


if __name__ == "__main__":
    main()
    
    
"""
Aynen öyle, router genişletme işi proje sonunda yapmak daha doğru olacak. burada bi daha çok retrievel eklemeye kalkalrsak işin içinden çıkamayız. 
"""
