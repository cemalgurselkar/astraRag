"""AstraRAG çalışma bileşenlerini kurar ve tek bir uygulama arayüzü sunar.

Sözlük:
- `AstraRAG.start`: Ayarları doğrular; indeks, retriever, generator ve pipeline'ı kurar.
- `AstraRAG.query`: Hazır pipeline üzerinde bir kullanıcı sorgusu çalıştırır.
- `AstraRAG.close`: HTTP, Gemini ve Qdrant kaynaklarını güvenle kapatır.
"""

from typing import Self

from qdrant_client import QdrantClient

from astrarag.config import Settings
from astrarag.embedding import EmbeddingEncoder
from astrarag.generation import GeminiGenerator
from astrarag.indexes import BM25Index, DenseVectorIndex, create_qdrant_client
from astrarag.models import PipelineResult, RetrievalRoute
from astrarag.pipeline import RAGPipeline
from astrarag.retrieval import (
    DenseRetriever,
    HybridRetriever,
    Retriever,
)


class AstraRAG:
    """Public application interface for the AstraRAG engine."""

    def __init__(
        self,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or Settings()

        self._qdrant_client: QdrantClient | None = None
        self._encoder: EmbeddingEncoder | None = None
        self._generator: GeminiGenerator | None = None
        self._pipeline: RAGPipeline | None = None

    def start(self) -> None:
        if self._pipeline is not None:
            return
        try:
            self._start()
        except Exception:
            self.close()
            raise

    def _start(self) -> None:
        if not self.settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required to start AstraRAG")

        self._encoder = EmbeddingEncoder(
            base_url=self.settings.embedding_url,
            model=self.settings.embedding_model,
            dimension=self.settings.embedding_dimension,
            timeout_seconds=self.settings.embedding_timeout_seconds,
        )

        self._qdrant_client = create_qdrant_client(
            url=self.settings.qdrant_url,
            path=self.settings.qdrant_path,
            api_key=self.settings.qdrant_api_key,
            timeout=self.settings.qdrant_timeout_seconds,
        )

        dense_index = DenseVectorIndex(
            path=self.settings.qdrant_path,
            collection_name=self.settings.qdrant_collection,
            dimension=self._encoder.dimension,
            hnsw_ef_search=self.settings.qdrant_hnsw_ef_search,
            expected_metadata=self.settings.index_metadata,
            validate_hnsw=self.settings.qdrant_url is not None,
            client=self._qdrant_client,
        )

        chunks = dense_index.get_chunks()

        if not chunks:
            raise RuntimeError(
                "AstraRAG corpus is empty. "
                "Index documents before starting the application."
            )

        self._encoder.check_health()

        dense = DenseRetriever(
            encoder=self._encoder,
            index=dense_index,
        )

        bm25 = BM25Index(chunks)

        hybrid = HybridRetriever(
            dense=dense,
            sparse=bm25,
            rrf_k=self.settings.hybrid_rrf_k,
        )

        retrievers: dict[RetrievalRoute, Retriever] = {
            RetrievalRoute.CHEAP: bm25,
            RetrievalRoute.MEDIUM: hybrid,
        }

        self._generator = GeminiGenerator(
            api_key=self.settings.gemini_api_key,
            model=self.settings.gemini_model,
        )

        self._pipeline = RAGPipeline(
            retrievers=retrievers,
            generator=self._generator,
            top_k=self.settings.retrieval_top_k,
            context_max_items=self.settings.context_max_items,
            context_max_characters=self.settings.context_max_char,
        )

    def query(self, query: str) -> PipelineResult:
        if self._pipeline is None:
            raise RuntimeError("AstraRAG has not been started. Call start() first.")

        return self._pipeline.run(query)

    def close(self) -> None:
        if self._generator is not None:
            self._generator.close()
            self._generator = None

        if self._qdrant_client is not None:
            self._qdrant_client.close()
            self._qdrant_client = None

        if self._encoder is not None:
            self._encoder.close()
            self._encoder = None

        self._pipeline = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()
