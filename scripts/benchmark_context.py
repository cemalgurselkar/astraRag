"""Compare context builders on evidence coverage, redundancy, size, and latency."""

import re
from functools import partial
from statistics import mean
from time import perf_counter

import numpy as np

from astrarag.config import Settings
from astrarag.context import build_context
from astrarag.embedding import EmbeddingEncoder
from astrarag.evaluation import load_evaluation_dataset
from astrarag.experiments import build_clustered_context, build_mmr_context
from astrarag.indexes import BM25Index, DenseVectorIndex, create_qdrant_client

TOP_N = 20
MAX_CHAR = 12_000
MAX_ITEMS = 10

CLUSTER_THRESHOLDS = (0.80, 0.85, 0.90, 0.95)
MMR_LAMBDAS = (0.50, 0.70, 0.90)
# Used only for measuring how many highly similar pairs remain
# in the final context.
REDUNDANCY_THRESHOLD = 0.90


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def evidence_metrics(context, query) -> tuple[float, float]:
    """
    Returns:
        coverage: 1.0 if at least one expected evidence is present.
        evidence_ratio: fraction of expected evidence retained.
    """

    context_text = normalize(context.text)

    matched = sum(
        normalize(evidence.text) in context_text for evidence in query.evidence
    )

    total = len(query.evidence)

    if total == 0:
        return 0.0, 0.0

    coverage = float(matched > 0)
    evidence_ratio = matched / total

    return coverage, evidence_ratio


def semantic_redundancy_metrics(
    context,
    encoder,
) -> tuple[float, float]:
    """
    Returns:
        mean_similarity:
            Mean pairwise cosine similarity.

        redundant_pair_rate:
            Fraction of context-item pairs whose cosine similarity
            is >= REDUNDANCY_THRESHOLD.
    """

    if len(context.items) < 2:
        return 0.0, 0.0

    embeddings = np.asarray(
        encoder.encode([item.text for item in context.items]),
        dtype=np.float32,
    )

    similarity_matrix = embeddings @ embeddings.T

    pairwise = similarity_matrix[
        np.triu_indices(
            len(context.items),
            k=1,
        )
    ]

    if len(pairwise) == 0:
        return 0.0, 0.0

    mean_similarity = float(np.mean(pairwise))

    redundant_pair_rate = float(np.mean(pairwise >= REDUNDANCY_THRESHOLD))

    return (
        mean_similarity,
        redundant_pair_rate,
    )


def percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0

    return float(
        np.percentile(
            values,
            95,
        )
    )


def evaluate_context(
    name,
    builder,
    retrieved_queries,
    encoder,
):
    item_counts = []
    char_counts = []

    mean_similarities = []
    redundant_pair_rates = []

    evidence_coverages = []
    evidence_ratios = []

    compression_ratios = []

    latencies = []

    for query, retrieval_results in retrieved_queries:
        start = perf_counter()

        context = builder(retrieval_results)

        latency_ms = (perf_counter() - start) * 1000

        item_count = len(context.items)

        if retrieval_results:
            compression_ratio = 1.0 - item_count / len(retrieval_results)
        else:
            compression_ratio = 0.0

        (
            mean_similarity,
            redundant_pair_rate,
        ) = semantic_redundancy_metrics(
            context,
            encoder,
        )

        (
            evidence_coverage,
            evidence_ratio,
        ) = evidence_metrics(
            context,
            query,
        )

        item_counts.append(item_count)

        char_counts.append(context.total_characters)

        mean_similarities.append(mean_similarity)

        redundant_pair_rates.append(redundant_pair_rate)

        evidence_coverages.append(evidence_coverage)

        evidence_ratios.append(evidence_ratio)

        compression_ratios.append(compression_ratio)

        latencies.append(latency_ms)

    return {
        "name": name,
        "items": mean(item_counts),
        "chars": mean(char_counts),
        "mean_similarity": mean(mean_similarities),
        "redundant_pair_rate": mean(redundant_pair_rates),
        "evidence_coverage": mean(evidence_coverages),
        "evidence_ratio": mean(evidence_ratios),
        "compression_ratio": mean(compression_ratios),
        "avg_latency_ms": mean(latencies),
        "p95_latency_ms": percentile_95(latencies),
    }


def main() -> None:
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

        bm25 = BM25Index(chunks)

        queries = load_evaluation_dataset(
            settings.evaluation_path / "retrieval_v1.json"
        )

        print(f"Evaluation queries: {len(queries)}")

        # --------------------------------------------------
        # Fixed retrieval
        #
        # Retrieval is executed only once.
        # Every Context Engine receives exactly the same
        # candidate chunks.
        # --------------------------------------------------

        print(f"Retrieving fixed Top-{TOP_N} candidate sets...")

        retrieved_queries = []

        for query in queries:
            retrieval_results = bm25.retrieve(
                query=query.query,
                top_k=TOP_N,
            )

            retrieved_queries.append(
                (
                    query,
                    retrieval_results,
                )
            )

        experiments = [
            (
                "Baseline",
                partial(
                    build_context,
                    max_characters=MAX_CHAR,
                    max_items=MAX_ITEMS,
                ),
            )
        ]

        for threshold in CLUSTER_THRESHOLDS:
            experiments.append(
                (
                    f"Cluster {threshold:.2f}",
                    partial(
                        build_clustered_context,
                        encoder=encoder,
                        max_characters=MAX_CHAR,
                        max_items=MAX_ITEMS,
                        similarity_threshold=threshold,
                    ),
                )
            )

        for lambda_mult in MMR_LAMBDAS:
            experiments.append(
                (
                    f"MMR {lambda_mult:.2f}",
                    partial(
                        build_mmr_context,
                        encoder=encoder,
                        max_characters=MAX_CHAR,
                        max_items=MAX_ITEMS,
                        lambda_mult=lambda_mult,
                    ),
                )
            )

        results = [
            evaluate_context(
                name=name,
                builder=builder,
                retrieved_queries=retrieved_queries,
                encoder=encoder,
            )
            for name, builder in experiments
        ]

        print()
        print("Context Benchmark")
        print("=" * 140)

        print(
            f"{'Strategy':<16}"
            f"{'Items':>8}"
            f"{'Chars':>10}"
            f"{'MeanSim':>10}"
            f"{'RedPair%':>11}"
            f"{'Ev.Cov':>10}"
            f"{'Ev.Ratio':>10}"
            f"{'Compress':>11}"
            f"{'Avg Lat':>12}"
            f"{'P95 Lat':>12}"
        )

        print("-" * 140)

        for result in results:
            print(
                f"{result['name']:<16}"
                f"{result['items']:>8.2f}"
                f"{result['chars']:>10.0f}"
                f"{result['mean_similarity']:>10.3f}"
                f"{result['redundant_pair_rate']:>10.1%}"
                f"{result['evidence_coverage']:>10.1%}"
                f"{result['evidence_ratio']:>10.1%}"
                f"{result['compression_ratio']:>10.1%}"
                f"{result['avg_latency_ms']:>9.2f} ms"
                f"{result['p95_latency_ms']:>9.2f} ms"
            )

    finally:
        encoder.close()
        qdrant_client.close()


if __name__ == "__main__":
    main()
