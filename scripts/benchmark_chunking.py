"""Compare token chunk configurations with llama.cpp embeddings."""

import math
from statistics import mean
from time import perf_counter

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from tokenizers import Tokenizer

from astrarag.config import Settings
from astrarag.documents import chunk_document_by_tokens, parse_pdf
from astrarag.embedding import EmbeddingEncoder
from astrarag.evaluation import (
    is_relevant_result,
    load_evaluation_dataset,
    resolve_dataset_ground_truth,
)
from astrarag.indexes import BM25Index, DenseVectorIndex
from astrarag.retrieval import DenseRetriever, HybridRetriever

TOP_K = 10
CONFIGS = ((128, 16), (128, 32), (192, 24), (192, 48), (254, 32), (254, 64))


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
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / idcg


def evaluate(retriever, queries) -> dict[str, float]:
    hit5: list[float] = []
    hit10: list[float] = []
    mrr: list[float] = []
    ndcg10: list[float] = []
    latencies: list[float] = []

    for query in queries:
        start = perf_counter()
        results = retriever.retrieve(query.query, top_k=TOP_K)
        latencies.append((perf_counter() - start) * 1000)
        relevance = [is_relevant_result(result, query) for result in results]
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


def encode_in_batches(
    encoder: EmbeddingEncoder,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    batches = [
        encoder.encode(texts[start : start + batch_size])
        for start in range(0, len(texts), batch_size)
    ]
    return np.vstack(batches)


def main() -> None:
    settings = Settings()
    tokenizer = Tokenizer.from_file(str(settings.embedding_tokenizer_path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    encoder = EmbeddingEncoder(
        base_url=settings.embedding_url,
        model=settings.embedding_model,
        dimension=settings.embedding_dimension,
        timeout_seconds=settings.embedding_timeout_seconds,
    )

    try:
        encoder.check_health()
        queries = load_evaluation_dataset(
            settings.evaluation_path / "retrieval_v1.json"
        )
        pdf_paths = sorted(settings.documents_path.rglob("*.pdf"))
        documents = [parse_pdf(path) for path in pdf_paths]

        print(
            f"Embedding model: {settings.embedding_model} via {settings.embedding_url}"
        )
        print(f"Documents: {len(documents)}; queries: {len(queries)}")
        print(
            "Chunk ablation uses exact in-memory Qdrant; "
            "use the production benchmark for ANN latency."
        )
        rows: list[dict[str, float | int | str]] = []

        for chunk_size, overlap in CONFIGS:
            print(f"\nTesting token chunking {chunk_size}/{overlap}")
            chunks = [
                chunk
                for document in documents
                for chunk in chunk_document_by_tokens(
                    document,
                    tokenizer,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            ]
            resolve_dataset_ground_truth(queries=queries, chunks=chunks)

            embedding_start = perf_counter()
            embeddings = encode_in_batches(
                encoder,
                [chunk.text for chunk in chunks],
                settings.embedding_batch_size,
            )
            embedding_seconds = perf_counter() - embedding_start

            client = QdrantClient(":memory:")
            try:
                client.create_collection(
                    collection_name="chunk-ablation",
                    vectors_config=VectorParams(
                        size=encoder.dimension,
                        distance=Distance.COSINE,
                    ),
                )
                dense_index = DenseVectorIndex(
                    path=settings.qdrant_path,
                    collection_name="chunk-ablation",
                    dimension=encoder.dimension,
                    hnsw_ef_search=settings.qdrant_hnsw_ef_search,
                    client=client,
                )
                dense_index.add(chunks=chunks, embeddings=embeddings)

                dense = DenseRetriever(encoder=encoder, index=dense_index)
                bm25 = BM25Index(chunks)
                hybrid = HybridRetriever(
                    dense=dense,
                    sparse=bm25,
                    rrf_k=settings.hybrid_rrf_k,
                )

                for strategy_name, retriever in (
                    ("Dense", dense),
                    ("BM25", bm25),
                    ("Hybrid", hybrid),
                ):
                    rows.append(
                        {
                            "config": f"{chunk_size}/{overlap}",
                            "chunks": len(chunks),
                            "embed_s": embedding_seconds,
                            "strategy": strategy_name,
                            **evaluate(retriever, queries),
                        }
                    )
            finally:
                client.close()

        print("\nToken Chunking Ablation")
        print("=" * 118)
        print(
            f"{'Config':<12}{'Chunks':>8}{'Embed':>11}{'Strategy':>12}"
            f"{'Hit@5':>9}{'Hit@10':>9}{'MRR':>9}{'nDCG@10':>11}{'Latency':>13}"
        )
        for row in rows:
            print(
                f"{row['config']:<12}{row['chunks']:>8}{row['embed_s']:>8.2f} s"
                f"{row['strategy']:>12}{row['hit5']:>9.3f}{row['hit10']:>9.3f}"
                f"{row['mrr']:>9.3f}{row['ndcg10']:>11.3f}"
                f"{row['latency']:>10.2f} ms"
            )
    finally:
        encoder.close()


if __name__ == "__main__":
    main()
