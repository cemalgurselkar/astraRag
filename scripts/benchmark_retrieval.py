from pathlib import Path
from statistics import mean
from time import perf_counter

from astrarag.evaluation.ground_truth import resolve_dataset_ground_truth

from astrarag.embedding import EmbeddingEncoder
from astrarag.evaluation import (
    load_evaluation_dataset,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

from astrarag.reranking import CrossEncoderReRanker
from astrarag.index import BM25Index, DenseVectorIndex
from astrarag.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RerankedRetriever
)


TOP_K = 10


def evaluate(name, retriever, queries, ground_truth):
    recall_5_scores = []
    recall_10_scores = []
    rr_scores = []
    ndcg_scores = []
    latencies = []

    for query in queries:
        
        relevant_ids = ground_truth[query.id]
        
        start = perf_counter()

        results = retriever.retriever(
            query=query.query,
            top_k=TOP_K,
        )

        latency_ms = (perf_counter() - start) * 1000

        retrieved_ids = [
            result.chunk_id
            for result in results
        ]

        recall_5_scores.append(
            recall_at_k(
                retrieved_ids,
                relevant_ids,
                k=5,
            )
        )

        recall_10_scores.append(
            recall_at_k(
                retrieved_ids,
                relevant_ids,
                k=10,
            )
        )

        rr_scores.append(
            reciprocal_rank(
                retrieved_ids,
                relevant_ids,
            )
        )

        ndcg_scores.append(
            ndcg_at_k(
                retrieved_ids,
                relevant_ids,
                k=10,
            )
        )

        latencies.append(latency_ms)

    return {
        "name": name,
        "recall_5": mean(recall_5_scores),
        "recall_10": mean(recall_10_scores),
        "mrr": mean(rr_scores),
        "ndcg_10": mean(ndcg_scores),
        "latency_ms": mean(latencies),
    }


def main() -> None:

    encoder = EmbeddingEncoder()

    dense_index = DenseVectorIndex(
        path=Path("data/qdrant"),
        collection_name="astrarag-corpus",
        dimension=encoder.dimension,
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
        candidate_k=20
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
        ("Hybrid + CE", hybrid_reranked)
    ]

    results = [
        evaluate(
            name,
            retriever,
            queries,
            ground_truth,
        )
        for name, retriever in experiments
    ]

    print("\nRetrieval Benchmark")
    print("=" * 90)

    print(
        f"{'Strategy':<15}"
        f"{'R@5':>10}"
        f"{'R@10':>10}"
        f"{'MRR':>10}"
        f"{'nDCG@10':>12}"
        f"{'Latency':>14}"
    )

    print("-" * 90)

    for result in results:
        print(
            f"{result['name']:<15}"
            f"{result['recall_5']:>10.3f}"
            f"{result['recall_10']:>10.3f}"
            f"{result['mrr']:>10.3f}"
            f"{result['ndcg_10']:>12.3f}"
            f"{result['latency_ms']:>11.2f} ms"
        )


if __name__ == "__main__":
    main()