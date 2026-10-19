"""Benchmark production retrieval quality and llama.cpp/Qdrant latency."""

import math
from collections.abc import Sequence
from statistics import mean, median
from time import perf_counter

import numpy as np

from astrarag.config import Settings
from astrarag.embedding import EmbeddingEncoder
from astrarag.evaluation import (
    is_relevant_result,
    load_evaluation_dataset,
    resolve_dataset_ground_truth,
)
from astrarag.indexes import BM25Index, DenseVectorIndex, create_qdrant_client
from astrarag.models import EvaluationQuery
from astrarag.retrieval import DenseRetriever, HybridRetriever, Retriever

TOP_K = 10


def hit_at_k(relevance: list[bool], k: int) -> float:
    return float(any(relevance[:k]))


def reciprocal_rank(relevance: list[bool]) -> float:
    for rank, relevant in enumerate(relevance, start=1):
        if relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevance: list[bool], k: int) -> float:
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


def latency_stats(values: Sequence[float]) -> tuple[float, float, float]:
    return mean(values), median(values), float(np.percentile(values, 95))


def evaluate(
    name: str,
    retriever: Retriever,
    queries: Sequence[EvaluationQuery],
) -> dict[str, float | str]:
    hit_5_scores: list[float] = []
    hit_10_scores: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcg_scores: list[float] = []
    latencies: list[float] = []

    for query in queries:
        start = perf_counter()
        results = retriever.retrieve(query.query, top_k=TOP_K)
        latencies.append((perf_counter() - start) * 1000)

        relevance = [is_relevant_result(result, query) for result in results]
        hit_5_scores.append(hit_at_k(relevance, 5))
        hit_10_scores.append(hit_at_k(relevance, 10))
        reciprocal_ranks.append(reciprocal_rank(relevance))
        ndcg_scores.append(ndcg_at_k(relevance, 10))

    latency_mean, latency_p50, latency_p95 = latency_stats(latencies)
    return {
        "name": name,
        "hit_5": mean(hit_5_scores),
        "hit_10": mean(hit_10_scores),
        "mrr": mean(reciprocal_ranks),
        "ndcg_10": mean(ndcg_scores),
        "latency_mean": latency_mean,
        "latency_p50": latency_p50,
        "latency_p95": latency_p95,
    }


def benchmark_components(
    queries: Sequence[EvaluationQuery],
    encoder: EmbeddingEncoder,
    dense_index: DenseVectorIndex,
    bm25: BM25Index,
) -> dict[str, tuple[float, float, float]]:
    embedding_latencies: list[float] = []
    qdrant_latencies: list[float] = []
    bm25_latencies: list[float] = []

    for query in queries:
        start = perf_counter()
        vector = encoder.encode_query(query.query)
        embedding_latencies.append((perf_counter() - start) * 1000)

        start = perf_counter()
        dense_index.search(vector, limit=TOP_K)
        qdrant_latencies.append((perf_counter() - start) * 1000)

        start = perf_counter()
        bm25.retrieve(query.query, top_k=TOP_K)
        bm25_latencies.append((perf_counter() - start) * 1000)

    return {
        "llama.cpp embedding": latency_stats(embedding_latencies),
        "Qdrant ANN search": latency_stats(qdrant_latencies),
        "BM25 search": latency_stats(bm25_latencies),
    }


def main() -> None:
    settings = Settings()
    encoder = EmbeddingEncoder(
        base_url=settings.embedding_url,
        model=settings.embedding_model,
        dimension=settings.embedding_dimension,
        timeout_seconds=settings.embedding_timeout_seconds,
    )
    client = create_qdrant_client(
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
            client=client,
        )

        startup = perf_counter()
        chunks = dense_index.get_chunks()
        chunk_load_ms = (perf_counter() - startup) * 1000

        startup = perf_counter()
        bm25 = BM25Index(chunks)
        bm25_build_ms = (perf_counter() - startup) * 1000

        queries = load_evaluation_dataset(
            settings.evaluation_path / "retrieval_v1.json"
        )
        ground_truth = resolve_dataset_ground_truth(queries=queries, chunks=chunks)

        dense = DenseRetriever(encoder=encoder, index=dense_index)
        hybrid = HybridRetriever(
            dense=dense,
            sparse=bm25,
            rrf_k=settings.hybrid_rrf_k,
        )

        # Exclude one-time model and connection warmup from latency measurements.
        warmup_vector = encoder.encode_query(queries[0].query)
        dense_index.search(warmup_vector, limit=TOP_K)
        bm25.retrieve(queries[0].query, top_k=TOP_K)

        print("Production Retrieval Benchmark")
        print("=" * 108)
        print(
            f"Embedding model: {settings.embedding_model} via {settings.embedding_url}"
        )
        print(
            f"Chunking: {settings.chunk_size_tokens} tokens / "
            f"{settings.chunk_overlap_tokens} overlap"
        )
        print(
            f"Qdrant collection: {settings.qdrant_collection} "
            f"(HNSW ef_search={settings.qdrant_hnsw_ef_search})"
        )
        print(f"Corpus: {len(chunks)} chunks; queries: {len(queries)}")
        print(f"Ground truth: {len(ground_truth)}/{len(queries)} queries validated")
        print(
            f"Startup: chunk load {chunk_load_ms:.2f} ms; "
            f"BM25 build {bm25_build_ms:.2f} ms"
        )

        component_results = benchmark_components(queries, encoder, dense_index, bm25)
        print("\nComponent latency (warm, milliseconds)")
        print(f"{'Component':<28}{'Mean':>12}{'p50':>12}{'p95':>12}")
        for name, (average, p50, p95) in component_results.items():
            print(f"{name:<28}{average:>12.2f}{p50:>12.2f}{p95:>12.2f}")

        results = [
            evaluate("Dense", dense, queries),
            evaluate("BM25", bm25, queries),
            evaluate("Hybrid RRF", hybrid, queries),
        ]

        print("\nQuality and end-to-end retrieval latency")
        print(
            f"{'Strategy':<18}{'Hit@5':>9}{'Hit@10':>9}{'MRR':>9}"
            f"{'nDCG@10':>11}{'Mean':>11}{'p50':>11}{'p95':>11}"
        )
        for result in results:
            print(
                f"{result['name']:<18}{result['hit_5']:>9.3f}"
                f"{result['hit_10']:>9.3f}{result['mrr']:>9.3f}"
                f"{result['ndcg_10']:>11.3f}{result['latency_mean']:>8.2f} ms"
                f"{result['latency_p50']:>8.2f} ms{result['latency_p95']:>8.2f} ms"
            )
    finally:
        encoder.close()
        client.close()


if __name__ == "__main__":
    main()
