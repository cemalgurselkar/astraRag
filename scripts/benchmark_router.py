"""Compare fixed retrieval strategies with adaptive rule-based query routing."""

from collections import Counter
from statistics import mean
from time import perf_counter

from astrarag.config import Settings
from astrarag.embedding import EmbeddingEncoder
from astrarag.evaluation import (
    is_relevant_result,
    load_evaluation_dataset,
    resolve_dataset_ground_truth,
)
from astrarag.indexes import BM25Index, DenseVectorIndex, create_qdrant_client
from astrarag.models import RetrievalRoute
from astrarag.retrieval import (
    DenseRetriever,
    HybridRetriever,
)
from astrarag.routing import choose_route, profile_query

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

    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))

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

        results = retriever.retrieve(
            query=query.query,
            top_k=TOP_K,
        )

        latency_ms = (perf_counter() - start) * 1000

        relevance = [is_relevant_result(result, query) for result in results]

        hit_5_scores.append(hit_at_k(relevance, 5))
        hit_10_scores.append(hit_at_k(relevance, 10))
        reciprocal_ranks.append(reciprocal_rank(relevance))
        ndcg_scores.append(ndcg_at_k(relevance, 10))
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

        route = choose_route(profile_query(query.query))

        route_counts[route] += 1

        retriever = retrievers[route]

        results = retriever.retrieve(
            query=query.query,
            top_k=TOP_K,
        )

        latency_ms = (perf_counter() - start) * 1000

        relevance = [is_relevant_result(result, query) for result in results]

        hit_5_scores.append(hit_at_k(relevance, 5))
        hit_10_scores.append(hit_at_k(relevance, 10))
        reciprocal_ranks.append(reciprocal_rank(relevance))
        ndcg_scores.append(ndcg_at_k(relevance, 10))
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

    settings = Settings()
    encoder = EmbeddingEncoder(
        base_url=settings.embedding_url,
        model=settings.embedding_model,
        dimension=settings.embedding_dimension,
        timeout_seconds=settings.embedding_timeout_seconds,
    )
    qdrant_client = create_qdrant_client(
        url=settings.qdrant_url,
        path=settings.qdrant_path,
        api_key=settings.qdrant_api_key,
        timeout=settings.qdrant_timeout_seconds,
    )

    try:
        encoder.check_health()
        dense_index = DenseVectorIndex(
            path=settings.qdrant_path,
            collection_name=settings.qdrant_collection,
            dimension=encoder.dimension,
            hnsw_ef_search=settings.qdrant_hnsw_ef_search,
            expected_metadata=settings.index_metadata,
            validate_hnsw=settings.qdrant_url is not None,
            client=qdrant_client,
        )

        chunks = dense_index.get_chunks()
        print(f"Benchmark corpus: {len(chunks)} chunks")

        dense = DenseRetriever(encoder=encoder, index=dense_index)
        bm25 = BM25Index(chunks)
        hybrid = HybridRetriever(
            dense=dense,
            sparse=bm25,
            rrf_k=settings.hybrid_rrf_k,
        )

        queries = load_evaluation_dataset(
            settings.evaluation_path / "retrieval_v1.json"
        )
        print(f"Evaluation queries: {len(queries)}")

        ground_truth = resolve_dataset_ground_truth(queries, chunks)
        print(f"Ground truth validated: {len(ground_truth)}/{len(queries)}")
        print(
            f"Embedding: {settings.embedding_model} via {settings.embedding_url}; "
            f"chunking: {settings.chunk_size_tokens}/{settings.chunk_overlap_tokens} tokens"
        )

        # Warm llama.cpp and Qdrant before comparing routing strategies.
        dense.retrieve(queries[0].query, top_k=TOP_K)

        adaptive_retrievers = {
            RetrievalRoute.CHEAP: bm25,
            RetrievalRoute.MEDIUM: hybrid,
        }
        results = [
            evaluate_strategy("Always BM25", bm25, queries),
            evaluate_strategy("Always Hybrid", hybrid, queries),
        ]

        adaptive_result, route_counts = evaluate_adaptive(
            queries=queries,
            retrievers=adaptive_retrievers,
        )
        results.append(adaptive_result)

        print_results(results)

        print("\nAdaptive route distribution:")

        total_queries = len(queries)

        for route in RetrievalRoute:
            count = route_counts[route]
            percentage = count / total_queries * 100 if total_queries else 0.0

            print(f"{route.value:<10}{count:>4} ({percentage:>5.1f}%)")
    finally:
        encoder.close()
        qdrant_client.close()


if __name__ == "__main__":
    main()
