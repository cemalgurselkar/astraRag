import json
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
from astrarag.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
)
from astrarag.schemas import RetrievalResult


TOP_K = 10
CANDIDATE_K = 20
RRF_K = 60

TRANSFORMATIONS_PATH = Path(
    "data/eval/query_transformations.json"
)


def hit_at_k(
    relevance: list[bool],
    k: int,
) -> float:
    return float(any(relevance[:k]))


def reciprocal_rank(
    relevance: list[bool],
) -> float:
    for rank, relevant in enumerate(
        relevance,
        start=1,
    ):
        if relevant:
            return 1.0 / rank

    return 0.0


def ndcg(
    relevance: list[bool],
    k: int,
) -> float:
    gains = relevance[:k]

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, relevant in enumerate(
            gains,
            start=1,
        )
        if relevant
    )

    relevant_count = sum(relevance)

    if relevant_count == 0:
        return 0.0

    ideal_count = min(
        relevant_count,
        k,
    )

    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(
            1,
            ideal_count + 1,
        )
    )

    return dcg / idcg


def fuse_rrf(
    result_sets: list[list[RetrievalResult]],
    top_k: int,
) -> list[RetrievalResult]:
    scores: dict[str, float] = {}
    results_by_id: dict[str, RetrievalResult] = {}

    for results in result_sets:
        for rank, result in enumerate(
            results,
            start=1,
        ):
            results_by_id[result.chunk_id] = result

            scores[result.chunk_id] = (
                scores.get(result.chunk_id, 0.0)
                + 1.0 / (RRF_K + rank)
            )

    ranked_ids = sorted(
        scores,
        key=scores.__getitem__,
        reverse=True,
    )[:top_k]

    return [
        results_by_id[chunk_id].model_copy(
            update={
                "score": scores[chunk_id],
            }
        )
        for chunk_id in ranked_ids
    ]


def evaluate(
    name,
    queries,
    retrieve,
):
    hit5 = []
    hit10 = []
    rr = []
    ndcg10 = []
    latencies = []

    for query in queries:
        start = perf_counter()

        results = retrieve(query)

        latency_ms = (
            perf_counter() - start
        ) * 1000

        relevance = [
            is_relevant_result(
                result,
                query,
            )
            for result in results
        ]

        hit5.append(
            hit_at_k(relevance, 5)
        )

        hit10.append(
            hit_at_k(relevance, 10)
        )

        rr.append(
            reciprocal_rank(relevance)
        )

        ndcg10.append(
            ndcg(relevance, 10)
        )

        latencies.append(
            latency_ms
        )

    return {
        "name": name,
        "hit5": mean(hit5),
        "hit10": mean(hit10),
        "mrr": mean(rr),
        "ndcg10": mean(ndcg10),
        "latency": mean(latencies),
    }


def main() -> None:
    encoder = EmbeddingEncoder()

    queries = load_evaluation_dataset(
        Path("data/eval/retrieval_v1.json")
    )

    transformations_data = json.loads(
        TRANSFORMATIONS_PATH.read_text(
            encoding="utf-8"
        )
    )

    transformations = {
        item["query_id"]: item
        for item in transformations_data
    }

    if len(transformations) != len(queries):
        raise RuntimeError(
            "Transformation cache does not match "
            "evaluation dataset."
        )

    client = QdrantClient(
        path="data/qdrant"
    )

    try:
        dense_index = DenseVectorIndex(
            path=Path("data/qdrant"),
            collection_name="astrarag-corpus",
            dimension=encoder.dimension,
            client=client,
        )

        chunks = dense_index.get_chunks()

        print(
            f"Benchmark corpus: {len(chunks)} chunks"
        )
        print(
            f"Evaluation queries: {len(queries)}"
        )

        resolve_dataset_ground_truth(
            queries=queries,
            chunks=chunks,
        )

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

        def dense_baseline(query):
            return dense.retriever(
                query=query.query,
                top_k=TOP_K,
            )

        def hybrid_baseline(query):
            return hybrid.retriever(
                query=query.query,
                top_k=TOP_K,
            )

        def multi_query_dense(query):
            cached = transformations[
                query.id
            ]

            search_queries = [
                query.query,
                *cached["alternative_queries"],
            ]

            result_sets = [
                dense.retriever(
                    query=search_query,
                    top_k=CANDIDATE_K,
                )
                for search_query in search_queries
            ]

            return fuse_rrf(
                result_sets,
                TOP_K,
            )

        def multi_query_hybrid(query):
            cached = transformations[
                query.id
            ]

            search_queries = [
                query.query,
                *cached["alternative_queries"],
            ]

            result_sets = [
                hybrid.retriever(
                    query=search_query,
                    top_k=CANDIDATE_K,
                    candidate_k=CANDIDATE_K,
                )
                for search_query in search_queries
            ]

            return fuse_rrf(
                result_sets,
                TOP_K,
            )

        def hyde_dense(query):
            cached = transformations[
                query.id
            ]

            hypothetical_document = (
                cached["hyde_document"]
            )

            vector = encoder.encode_query(
                hypothetical_document
            )

            return dense_index.search(
                query_vector=vector,
                limit=TOP_K,
            )

        experiments = [
            (
                "Dense",
                dense_baseline,
            ),
            (
                "Hybrid",
                hybrid_baseline,
            ),
            (
                "MultiQuery Dense",
                multi_query_dense,
            ),
            (
                "MultiQuery Hybrid",
                multi_query_hybrid,
            ),
            (
                "HyDE Dense",
                hyde_dense,
            ),
        ]

        results = [
            evaluate(
                name=name,
                queries=queries,
                retrieve=retrieve,
            )
            for name, retrieve in experiments
        ]

        transformation_latency = mean(
            item["transformation_latency_ms"]
            for item in transformations_data
        )

        print()
        print("Advanced Retrieval Benchmark")
        print("=" * 100)

        print(
            f"{'Strategy':<20}"
            f"{'Hit@5':>10}"
            f"{'Hit@10':>10}"
            f"{'MRR':>10}"
            f"{'nDCG@10':>12}"
            f"{'Retrieval':>15}"
        )

        print("-" * 100)

        for result in results:
            print(
                f"{result['name']:<20}"
                f"{result['hit5']:>10.3f}"
                f"{result['hit10']:>10.3f}"
                f"{result['mrr']:>10.3f}"
                f"{result['ndcg10']:>12.3f}"
                f"{result['latency']:>12.2f} ms"
            )

        print()
        print(
            "Average cached Gemini transformation latency: "
            f"{transformation_latency:.2f} ms/query"
        )

        print(
            "(Transformation latency is NOT included in "
            "retrieval latency above.)"
        )

    finally:
        client.close()


if __name__ == "__main__":
    main()