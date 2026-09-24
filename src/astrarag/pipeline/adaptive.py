"""Route queries through retrieval tiers and assemble generated pipeline results."""

from time import perf_counter
from typing import Protocol

from astrarag.context import ContextEngine
from astrarag.generation import GeminiGenerator
from astrarag.routing import QueryProfiler, RuleBasedRouter
from astrarag.schemas import (
    PipelineResult,
    RetrievalResult,
    RetrievalRoute,
)


class Retriever(Protocol):
    def retriever(self, query: str, top_k: int = 10) -> list[RetrievalResult]: ...


class AdaptiveRAGPipeline:
    def __init__(
        self,
        profiler: QueryProfiler,
        router: RuleBasedRouter,
        retrievers: dict[RetrievalRoute, Retriever],
        context_engine: ContextEngine,
        generator: GeminiGenerator,
        top_k: int = 10,
    ) -> None:

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        missing_routes = set(RetrievalRoute) - set(retrievers)

        if missing_routes:
            missing = ", ".join(sorted(route.value for route in missing_routes))
            raise ValueError(f"Missing retrievers for routes: {missing}")

        self.profiler = profiler
        self.router = router
        self.retrievers = retrievers
        self.context_engine = context_engine
        self.generator = generator
        self.top_k = top_k

    def run(self, query: str) -> PipelineResult:

        query = query.strip()

        if not query:
            raise ValueError("query must not be empty")

        total_start = perf_counter()

        profile = self.profiler.profile(query)
        route = self.router.route(profile)

        retriever = self.retrievers[route]

        retrieval_start = perf_counter()

        retrieval_results = retriever.retriever(
            query=query,
            top_k=self.top_k,
        )

        retrieval_latency_ms = (perf_counter() - retrieval_start) * 1000

        context = self.context_engine.build(retrieval_results)

        generation = self.generator.generate(
            query=query,
            context=context,
        )

        total_latency_ms = (perf_counter() - total_start) * 1000

        return PipelineResult(
            query=query,
            route=route,
            query_profile=profile,
            retrieval_results=retrieval_results,
            context=context,
            generation=generation,
            retrieval_latency_ms=retrieval_latency_ms,
            total_latency_ms=total_latency_ms,
        )
