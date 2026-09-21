from collections.abc import Sequence
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from numpy.typing import NDArray
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from astrarag.schemas import Chunk, RetrievalResult


class DenseVectorIndex:
    
    def __init__(self, path:Path, collection_name: str, dimension: int) -> None:
        
        if dimension <= 0:
            raise ValueError("Dimension must be greater than 0")
        
        self.collection_name = collection_name
        self.dimension = dimension
        self._client = QdrantClient(path=str(path))
        
        if not self._client.collection_exists(collection_name=collection_name):
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance = Distance.COSINE,
                ),
            )
    
    def add(self, 
        chunks: Sequence[Chunk],
        embeddings: NDArray[np.float32]) -> None:
        
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must contain the same number of items")
        
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension {self.dimension},"
                f"got {embeddings.shape[1]}"
            )
        
        point = [
            PointStruct(
                id=str(uuid5(NAMESPACE_URL, chunk.id)),
                vector=embedding.tolist(),
                payload={
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "index": chunk.index,
                    "page_numbers": chunk.page_numbers,
                    "metadata": chunk.metadata,
                },
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        
        self._client.upsert(
            collection_name=self.collection_name,
            points=point,
        )
    
    def search(
        self,
        query_vector: NDArray[np.float32],
        limit: int,
    ) -> list[RetrievalResult]:
        
        if query_vector.ndim != 1:
            raise ValueError("query_vector must be a 1D array")
        
        if query_vector.shape[0] != self.dimension:
            raise ValueError(
                f"Expected query dimension {self.dimension},"
                f"got {query_vector.shape[0]}"
            )
        
        if limit <= 0:
            raise ValueError("limit must be greater than 0")
        
        points = self._client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=limit,
            with_payload=True
        ).points
        
        results: list[RetrievalResult] = []
        
        for point in points:
            payload = point.payload or {}
            
            results.append(
                RetrievalResult(
                    chunk_id=str(payload["chunk_id"]),
                    document_id=str(payload["document_id"]),
                    text=str(payload["text"]),
                    score=point.score,
                    page_numbers=list(payload.get("page_numbers", [])),
                    metadata=dict(payload.get("metadata", {})),
                )
            )
        return results