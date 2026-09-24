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
from astrarag.ingestion.chunker import FixedSizeChunker
from astrarag.ingestion.parser import PDFParser
from astrarag.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
)


TOP_K = 10

CONFIGS = [
    (1000, 200),
    (1200, 200),
    (1400, 200),
    (1600, 200),
    (1800, 200),
    (1000, 300),
    (1200, 300),
    (1400, 300),
    (1600, 300),
    (1800, 300),
    (1000, 400),
    (1200, 400),
    (1400, 400),
    (1600, 400),
    (1800, 400),
]


def hit_at_k(relevance: list[bool], k: int) -> float:
    return float(any(relevance[:k]))


def reciprocal_rank(relevance: list[bool]) -> float:
    for rank, relevant in enumerate(relevance, start=1):
        if relevant:
            return 1.0 / rank

    return 0.0


def ndcg(relevance: list[bool], k: int) -> float:
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


def evaluate(retriever, queries):
    hit5 = []
    hit10 = []
    mrr = []
    ndcg10 = []
    latencies = []

    for query in queries:
        start = perf_counter()

        results = retriever.retriever(
            query=query.query,
            top_k=TOP_K,
        )

        latencies.append(
            (perf_counter() - start) * 1000
        )

        relevance = [
            is_relevant_result(result, query)
            for result in results
        ]

        hit5.append(hit_at_k(relevance, 5))
        hit10.append(hit_at_k(relevance, 10))
        mrr.append(reciprocal_rank(relevance))
        ndcg10.append(ndcg(relevance, 10))

    return {
        "hit5": mean(hit5),
        "hit10": mean(hit10),
        "mrr": mean(mrr),
        "ndcg10": mean(ndcg10),
        "latency": mean(latencies),
    }


def main() -> None:
    encoder = EmbeddingEncoder()
    parser = PDFParser()

    queries = load_evaluation_dataset(
        Path("data/eval/retrieval_v1.json")
    )

    pdf_paths = sorted(
        Path("data/documents").rglob("*.pdf")
    )

    documents = [
        parser.parse(path)
        for path in pdf_paths
    ]

    print(f"Documents: {len(documents)}")
    print(f"Queries: {len(queries)}")

    rows = []

    for chunk_size, overlap in CONFIGS:
        print()
        print(
            f"Testing chunk_size={chunk_size}, "
            f"overlap={overlap}"
        )

        chunker = FixedSizeChunker(
            chunk_size=chunk_size,
            overlap=overlap,
        )

        chunks = [
            chunk
            for document in documents
            for chunk in chunker.chunk(document)
        ]

        # Important: validate the same evaluation evidence
        # against this chunk configuration.
        resolve_dataset_ground_truth(
            queries=queries,
            chunks=chunks,
        )

        embeddings = encoder.encode(
            [chunk.text for chunk in chunks]
        )

        client = QdrantClient(":memory:")

        try:
            dense_index = DenseVectorIndex(
                path=Path("."),
                collection_name="chunk-ablation",
                dimension=encoder.dimension,
                client=client,
            )

            dense_index.add(
                chunks=chunks,
                embeddings=embeddings,
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

            for strategy_name, retriever in [
                ("Dense", dense),
                ("BM25", bm25),
                ("Hybrid", hybrid),
            ]:
                metrics = evaluate(
                    retriever,
                    queries,
                )

                rows.append(
                    {
                        "config": f"{chunk_size}/{overlap}",
                        "chunks": len(chunks),
                        "strategy": strategy_name,
                        **metrics,
                    }
                )

        finally:
            client.close()

    print()
    print("Chunking Ablation")
    print("=" * 105)

    print(
        f"{'Config':<12}"
        f"{'Chunks':>8}"
        f"{'Strategy':>12}"
        f"{'Hit@5':>10}"
        f"{'Hit@10':>10}"
        f"{'MRR':>10}"
        f"{'nDCG@10':>12}"
        f"{'Latency':>14}"
    )

    print("-" * 105)

    for row in rows:
        print(
            f"{row['config']:<12}"
            f"{row['chunks']:>8}"
            f"{row['strategy']:>12}"
            f"{row['hit5']:>10.3f}"
            f"{row['hit10']:>10.3f}"
            f"{row['mrr']:>10.3f}"
            f"{row['ndcg10']:>12.3f}"
            f"{row['latency']:>11.2f} ms"
        )


if __name__ == "__main__":
    main()