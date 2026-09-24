"""Compose AstraRAG's indexing, retrieval, routing, context, and generation services."""

from typing import Self

from qdrant_client import QdrantClient

from astrarag.config import Settings
from astrarag.context import ContextEngine
from astrarag.embedding import EmbeddingEncoder
from astrarag.generation import GeminiGenerator
from astrarag.index import BM25Index, DenseVectorIndex
from astrarag.pipeline import AdaptiveRAGPipeline
from astrarag.reranking import CrossEncoderReRanker
from astrarag.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    RerankedRetriever,
)
from astrarag.routing import QueryProfiler, RuleBasedRouter
from astrarag.schemas import PipelineResult, RetrievalRoute


class AstraRAG:
    """Public application interface for the AstraRAG engine."""

    def __init__(
        self,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or Settings()

        self._qdrant_client: QdrantClient | None = None
        self._pipeline: AdaptiveRAGPipeline | None = None

    def start(self) -> None:
        if self._pipeline is not None:
            return

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

        bm25_index = BM25Index()
        bm25_index.build(chunks)

        bm25 = BM25Retriever(
            index=bm25_index,
        )

        hybrid = HybridRetriever(
            dense_retriever=dense,
            sparse_retriever=bm25,
            rrf_k=self.settings.hybrid_rrf_k,
        )

        reranker = CrossEncoderReRanker(model_name=self.settings.reranker_model_path)

        hybrid_ce = RerankedRetriever(
            base_retriever=hybrid,
            reranker=reranker,
            candidate_k=self.settings.reranker_candidate_k,
        )

        retrievers = {
            RetrievalRoute.CHEAP: bm25,
            RetrievalRoute.MEDIUM: hybrid,
            RetrievalRoute.EXPENSIVE: hybrid_ce,
        }

        self._pipeline = AdaptiveRAGPipeline(
            profiler=QueryProfiler(),
            router=RuleBasedRouter(),
            retrievers=retrievers,
            context_engine=ContextEngine(
                max_items=self.settings.context_max_items,
                max_char=self.settings.context_max_char,
            ),
            generator=GeminiGenerator(
                api_key=self.settings.gemini_api_key,
                model=self.settings.gemini_model,
            ),
            top_k=self.settings.retrieval_top_k,
        )

    def query(self, query: str) -> PipelineResult:
        if self._pipeline is None:
            raise RuntimeError("AstraRAG has not been started. Call start() first.")

        return self._pipeline.run(query)

    def close(self) -> None:
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
