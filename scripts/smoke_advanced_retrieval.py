from pathlib import Path

from qdrant_client import QdrantClient

from astrarag.config import Settings
from astrarag.embedding import EmbeddingEncoder
from astrarag.index import DenseVectorIndex
from astrarag.retrieval import (
    GeminiQueryTransformer,
    HyDERetriever,
    MultiQueryRetriever,
    DenseRetriever,
)


settings = Settings()
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
    base_retriever=dense,
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
    for result in multi.retriever(query, top_k=5):
        print(result.score, result.document_id)

    print("\nHYDE")
    for result in hyde.retriever(query, top_k=5):
        print(result.score, result.document_id)

finally:
    transformer.close()
    client.close()