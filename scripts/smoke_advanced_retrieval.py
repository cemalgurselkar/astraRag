"""Smoke-test multi-query and HyDE retrieval against the configured vector index."""

from pathlib import Path

from qdrant_client import QdrantClient

from astrarag.config import Settings
from astrarag.embedding import EmbeddingEncoder
from astrarag.experiments import (
    GeminiQueryTransformer,
    HyDERetriever,
    MultiQueryRetriever,
)
from astrarag.indexes import DenseVectorIndex
from astrarag.retrieval import DenseRetriever

settings = Settings()
if not settings.gemini_api_key:
    raise RuntimeError("GEMINI_API_KEY is required for this experiment")
encoder = EmbeddingEncoder()

client = QdrantClient(path="data/qdrant")

index = DenseVectorIndex(
    path=Path("data/qdrant"),
    collection_name="astrarag-corpus",
    dimension=encoder.dimension,
    client=client,
)

dense = DenseRetriever(
    encoder=encoder,
    index=index,
)

transformer = GeminiQueryTransformer(
    api_key=settings.gemini_api_key,
    model=settings.gemini_model,
)

multi = MultiQueryRetriever(
    base=dense,
    transformer=transformer,
)

hyde = HyDERetriever(
    encoder=encoder,
    index=index,
    transformer=transformer,
)

query = "What is the main idea behind Sentence-BERT?"
try:
    print("MULTI QUERY")
    for result in multi.retrieve(query, top_k=5):
        print(result.score, result.document_id)

    print("\nHYDE")
    for result in hyde.retrieve(query, top_k=5):
        print(result.score, result.document_id)

finally:
    transformer.close()
    client.close()
