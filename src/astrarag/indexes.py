"""Qdrant dense vektör indeksini ve bellek içi BM25 indeksini yönetir.

Sözlük:
- `create_qdrant_client`: Uzak Qdrant'a veya yerel geliştirme deposuna bağlanır.
- `DenseVectorIndex`: Chunk'ları yazar/okur, ayarları doğrular ve ANN araması yapar.
- `BM25Index`: Chunk metinleri üzerinde sözcük tabanlı sparse arama yapar.
- `_tokenize`: BM25 için metni basit terimlere ayırır.
"""

import re
from collections.abc import Sequence
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from numpy.typing import NDArray
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SearchParams
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from astrarag.models import Chunk, RetrievalResult


def create_qdrant_client(
    *,
    url: str | None,
    path: Path,
    api_key: str | None,
    timeout: int,
) -> QdrantClient:
    """Connect to Qdrant server in production or local storage in development."""
    if url:
        return QdrantClient(
            url=url,
            api_key=api_key,
            timeout=timeout,
        )
    return QdrantClient(path=str(path))


class DenseVectorIndex:
    """Persist chunk embeddings in Qdrant and perform vector search."""

    def __init__(
        self,
        path: Path,
        collection_name: str,
        dimension: int,
        hnsw_ef_search: int = 128,
        expected_metadata: dict[str, object] | None = None,
        validate_hnsw: bool = False,
        client: QdrantClient | None = None,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than 0")

        self.collection_name = collection_name
        self.dimension = dimension
        self.hnsw_ef_search = hnsw_ef_search
        self._client = client or QdrantClient(path=str(path))

        if not self._client.collection_exists(collection_name=collection_name):
            raise RuntimeError(
                f"Qdrant collection '{collection_name}' does not exist. "
                "Index documents before starting the application."
            )

        collection = self._client.get_collection(collection_name)
        vectors = collection.config.params.vectors
        actual_dimension = getattr(vectors, "size", None)
        if actual_dimension != dimension:
            raise RuntimeError(
                f"Qdrant collection '{collection_name}' has vector dimension "
                f"{actual_dimension}; expected {dimension}. Rebuild the index."
            )
        metadata = collection.config.metadata or {}
        if expected_metadata:
            mismatches = {
                key: (metadata.get(key), value)
                for key, value in expected_metadata.items()
                if metadata.get(key) != value
            }
            if mismatches:
                details = ", ".join(
                    f"{key}={actual!r} (expected {expected!r})"
                    for key, (actual, expected) in mismatches.items()
                )
                raise RuntimeError(
                    f"Qdrant collection '{collection_name}' is incompatible: "
                    f"{details}. Rebuild the index."
                )
        if validate_hnsw and expected_metadata:
            expected_m = expected_metadata["hnsw_m"]
            expected_ef = expected_metadata["hnsw_ef_construct"]
            expected_full_scan = expected_metadata["hnsw_full_scan_threshold_kb"]
            expected_threshold = expected_metadata["hnsw_indexing_threshold_kb"]
            hnsw = collection.config.hnsw_config
            optimizer = collection.config.optimizer_config
            if (
                hnsw.m != expected_m
                or hnsw.ef_construct != expected_ef
                or hnsw.full_scan_threshold != expected_full_scan
                or optimizer.indexing_threshold != expected_threshold
            ):
                raise RuntimeError(
                    f"Qdrant collection '{collection_name}' has incompatible "
                    "HNSW settings. Rebuild the index."
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
        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
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
                f"Expected query dimension {self.dimension}, "
                f"got {query_vector.shape[0]}"
            )
        if limit <= 0:
            raise ValueError("limit must be greater than 0")

        points = self._client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            search_params=SearchParams(
                hnsw_ef=self.hnsw_ef_search,
                exact=False,
            ),
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
