"""Compare fixed retrieval strategies with adaptive rule-based query routing."""

from collections import Counter
from pathlib import Path
from statistics import mean
from time import perf_counter

from qdrant_client import QdrantClient

from astrarag.embedding import EmbeddingEncoder
from astrarag.evaluation.dataset import load_evaluation_dataset
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
    RerankedRetriever,
)
from astrarag.routing import QueryProfiler, RuleBasedRouter
from astrarag.schemas import RetrievalRoute

TOP_K = 10


def hit_at_k(
    relevance: list[bool],
    k: int,
) -> float:
    return float(any(relevance[:k]))


def reciprocal_rank(
    relevance: list[bool],
) -> float:
    for rank, relevant in enumerate(relevance, start=1):
        if relevant:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    relevance: list[bool],
    k: int,
) -> float:
    import math

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


def evaluate_strategy(
    name: str,
    retriever,
    queries,
) -> dict[str, object]:
    hit_5_scores: list[float] = []
    hit_10_scores: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcg_scores: list[float] = []
    latencies: list[float] = []

    for query in queries:
        start = perf_counter()

        results = retriever.retriever(
            query=query.query,
            top_k=TOP_K,
        )

        latency_ms = (
            perf_counter() - start
        ) * 1000

        relevance = [
            is_relevant_result(result, query)
            for result in results
        ]

        hit_5_scores.append(
            hit_at_k(relevance, 5)
        )
        hit_10_scores.append(
            hit_at_k(relevance, 10)
        )
        reciprocal_ranks.append(
            reciprocal_rank(relevance)
        )
        ndcg_scores.append(
            ndcg_at_k(relevance, 10)
        )
        latencies.append(latency_ms)

    return {
        "name": name,
        "hit_5": mean(hit_5_scores),
        "hit_10": mean(hit_10_scores),
        "mrr": mean(reciprocal_ranks),
        "ndcg_10": mean(ndcg_scores),
        "latency_ms": mean(latencies),
    }


def evaluate_adaptive(
    queries,
    profiler: QueryProfiler,
    router: RuleBasedRouter,
    retrievers: dict[RetrievalRoute, object],
) -> tuple[dict[str, object], Counter]:
    hit_5_scores: list[float] = []
    hit_10_scores: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcg_scores: list[float] = []
    latencies: list[float] = []

    route_counts: Counter = Counter()

    for query in queries:
        start = perf_counter()

        profile = profiler.profile(query.query)
        route = router.route(profile)

        route_counts[route] += 1

        retriever = retrievers[route]

        results = retriever.retriever(
            query=query.query,
            top_k=TOP_K,
        )

        latency_ms = (
            perf_counter() - start
        ) * 1000

        relevance = [
            is_relevant_result(result, query)
            for result in results
        ]

        hit_5_scores.append(
            hit_at_k(relevance, 5)
        )
        hit_10_scores.append(
            hit_at_k(relevance, 10)
        )
        reciprocal_ranks.append(
            reciprocal_rank(relevance)
        )
        ndcg_scores.append(
            ndcg_at_k(relevance, 10)
        )
        latencies.append(latency_ms)

    metrics = {
        "name": "Adaptive Router",
        "hit_5": mean(hit_5_scores),
        "hit_10": mean(hit_10_scores),
        "mrr": mean(reciprocal_ranks),
        "ndcg_10": mean(ndcg_scores),
        "latency_ms": mean(latencies),
    }

    return metrics, route_counts


def print_results(
    results: list[dict[str, object]],
) -> None:
    print()
    print(
        f"{'Strategy':<22}"
        f"{'Hit@5':>10}"
        f"{'Hit@10':>10}"
        f"{'MRR':>10}"
        f"{'nDCG@10':>12}"
        f"{'Latency':>14}"
    )

    for result in results:
        print(
            f"{result['name']:<22}"
            f"{result['hit_5']:>10.3f}"
            f"{result['hit_10']:>10.3f}"
            f"{result['mrr']:>10.3f}"
            f"{result['ndcg_10']:>12.3f}"
            f"{result['latency_ms']:>11.2f} ms"
        )


def main() -> None:
    print("Loading models and indexes...")

    encoder = EmbeddingEncoder()

    qdrant_client = QdrantClient(
        path="data/qdrant"
    )

    dense_index = DenseVectorIndex(
        path=Path("data/qdrant"),
        collection_name="astrarag-corpus",
        dimension=encoder.dimension,
        client=qdrant_client,
    )

    chunks = dense_index.get_chunks()

    print(f"Benchmark corpus: {len(chunks)} chunks")

    dense = DenseRetriever(
        encoder=encoder,
        index=dense_index,
    )

    bm25_index = BM25Index()
    bm25_index.build(chunks=chunks)

    bm25 = BM25Retriever(
        index=bm25_index,
    )

    hybrid = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=bm25,
        rrf_k=20,
    )

    reranker = CrossEncoderReRanker()

    hybrid_ce = RerankedRetriever(
        base_retriever=hybrid,
        reranker=reranker,
        candidate_k=20,
    )

    queries = load_evaluation_dataset(
        Path("data/eval/retrieval_v1.json")
    )

    print(f"Evaluation queries: {len(queries)}")

    ground_truth = resolve_dataset_ground_truth(
        queries,
        chunks,
    )

    print(
        "Ground truth validated: "
        f"{len(ground_truth)}/{len(queries)}"
    )

    profiler = QueryProfiler()
    router = RuleBasedRouter()

    adaptive_retrievers = {
        RetrievalRoute.CHEAP: bm25,
        RetrievalRoute.MEDIUM: hybrid,
        RetrievalRoute.EXPENSIVE: hybrid_ce,
    }

    results = [
        evaluate_strategy(
            "Always BM25",
            bm25,
            queries,
        ),
        evaluate_strategy(
            "Always Hybrid",
            hybrid,
            queries,
        ),
        evaluate_strategy(
            "Always Hybrid + CE",
            hybrid_ce,
            queries,
        ),
    ]

    adaptive_result, route_counts = evaluate_adaptive(
        queries=queries,
        profiler=profiler,
        router=router,
        retrievers=adaptive_retrievers,
    )

    results.append(adaptive_result)

    print_results(results)

    print("\nAdaptive route distribution:")

    total_queries = len(queries)

    for route in RetrievalRoute:
        count = route_counts[route]
        percentage = (
            count / total_queries * 100
            if total_queries
            else 0.0
        )

        print(
            f"{route.value:<10}"
            f"{count:>4} "
            f"({percentage:>5.1f}%)"
        )

    qdrant_client.close()


if __name__ == "__main__":
    main()
