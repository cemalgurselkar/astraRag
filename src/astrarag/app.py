"""Compose and expose the AstraRAG runtime."""

from typing import Self

from qdrant_client import QdrantClient

from astrarag.config import Settings
from astrarag.embedding import EmbeddingEncoder
from astrarag.generation import GeminiGenerator
from astrarag.indexes import BM25Index, DenseVectorIndex
from astrarag.models import PipelineResult, RetrievalRoute
from astrarag.pipeline import RAGPipeline
from astrarag.retrieval import (
    CrossEncoderReranker,
    DenseRetriever,
    HybridRetriever,
    RerankedRetriever,
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
        self._generator: GeminiGenerator | None = None
        self._pipeline: RAGPipeline | None = None

    def start(self) -> None:
        if self._pipeline is not None:
            return
        if not self.settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required to start AstraRAG")

        encoder = EmbeddingEncoder(model_name=self.settings.embedding_model_path)

        self._qdrant_client = QdrantClient(
            path=str(self.settings.qdrant_path),
        )

        dense_index = DenseVectorIndex(
            path=self.settings.qdrant_path,
            collection_name=self.settings.qdrant_collection,
            dimension=encoder.dimension,
            client=self._qdrant_client,
        )

        chunks = dense_index.get_chunks()

        if not chunks:
            raise RuntimeError(
                "AstraRAG corpus is empty. "
                "Index documents before starting the application."
            )

        dense = DenseRetriever(
            encoder=encoder,
            index=dense_index,
        )

        bm25 = BM25Index(chunks)

        hybrid = HybridRetriever(
            dense=dense,
            sparse=bm25,
            rrf_k=self.settings.hybrid_rrf_k,
        )

        reranker = CrossEncoderReranker(model_name=self.settings.reranker_model_path)

        hybrid_ce = RerankedRetriever(
            base=hybrid,
            reranker=reranker,
            candidate_k=self.settings.reranker_candidate_k,
        )

        retrievers: dict[RetrievalRoute, Retriever] = {
            RetrievalRoute.CHEAP: bm25,
            RetrievalRoute.MEDIUM: hybrid,
            RetrievalRoute.EXPENSIVE: hybrid_ce,
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
