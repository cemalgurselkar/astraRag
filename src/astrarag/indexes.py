"""Dense Qdrant and in-memory BM25 indexes."""

import re
from collections.abc import Sequence
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from numpy.typing import NDArray
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from astrarag.models import Chunk, RetrievalResult


class DenseVectorIndex:
    """Persist chunk embeddings in Qdrant and perform vector search."""

    def __init__(
        self,
        path: Path,
        collection_name: str,
        dimension: int,
        client: QdrantClient | None = None,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than 0")

        self.collection_name = collection_name
        self.dimension = dimension
        self._client = client or QdrantClient(path=str(path))

        if not self._client.collection_exists(collection_name=collection_name):
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )

    def add(self, chunks: Sequence[Chunk], embeddings: NDArray[np.float32]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension {self.dimension}, "
                f"got {embeddings.shape[1]}"
            )

        points = [
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
        self._client.upsert(collection_name=self.collection_name, points=points)

    def search(
        self,
        query_vector: NDArray[np.float32],
        limit: int,
    ) -> list[RetrievalResult]:
        if query_vector.ndim != 1:
            raise ValueError("query_vector must be a 1D array")
        if query_vector.shape[0] != self.dimension:
            raise ValueError(
                f"Expected query dimension {self.dimension}, "
                f"got {query_vector.shape[0]}"
            )
        if limit <= 0:
            raise ValueError("limit must be greater than 0")

        points = self._client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=limit,
            with_payload=True,
        ).points

        return [
            RetrievalResult(
                chunk_id=str((point.payload or {})["chunk_id"]),
                document_id=str((point.payload or {})["document_id"]),
                text=str((point.payload or {})["text"]),
                score=point.score,
                page_numbers=list((point.payload or {}).get("page_numbers", [])),
                metadata=dict((point.payload or {}).get("metadata", {})),
            )
            for point in points
        ]

    def get_chunks(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        offset = None

        while True:
            points, offset = self._client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                chunks.append(
                    Chunk(
                        id=str(payload["chunk_id"]),
                        document_id=str(payload["document_id"]),
                        text=str(payload["text"]),
                        index=int(payload["index"]),
                        page_numbers=list(payload.get("page_numbers", [])),
                        metadata=dict(payload.get("metadata", {})),
                    )
                )
            if offset is None:
                return chunks


class BM25Index:
    """Own a tokenized corpus and retrieve lexical matches."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            raise ValueError("chunks cannot be empty")
        self._chunks = list(chunks)
        self._index = BM25Okapi([_tokenize(chunk.text) for chunk in self._chunks])

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        scores = self._index.get_scores(_tokenize(query))
        ranked = sorted(
            range(len(scores)), key=lambda index: scores[index], reverse=True
        )[:top_k]

        return [
            RetrievalResult(
                chunk_id=self._chunks[index].id,
                document_id=self._chunks[index].document_id,
                text=self._chunks[index].text,
                score=float(scores[index]),
                page_numbers=self._chunks[index].page_numbers,
                metadata=self._chunks[index].metadata,
            )
            for index in ranked
        ]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())
