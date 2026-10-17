"""Run the query-to-answer pipeline."""

from time import perf_counter
from typing import Protocol

from astrarag.context import build_context
from astrarag.models import (
    ContextPackage,
    GeneratedAnswer,
    PipelineResult,
    RetrievalRoute,
)
from astrarag.retrieval import Retriever
from astrarag.routing import choose_route, profile_query


class Generator(Protocol):
    """Anything that can turn a query and context into a grounded answer."""

    def generate(
        self,
        query: str,
        context: ContextPackage,
    ) -> GeneratedAnswer: ...


class RAGPipeline:
    """Hold runtime strategies and execute the deterministic outer pipeline."""

    def __init__(
        self,
        retrievers: dict[RetrievalRoute, Retriever],
        generator: Generator,
        top_k: int = 10,
        context_max_items: int = 10,
        context_max_characters: int = 12_000,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        missing = set(RetrievalRoute) - set(retrievers)
        if missing:
            names = ", ".join(sorted(route.value for route in missing))
            raise ValueError(f"Missing retrievers for routes: {names}")

        self.retrievers = retrievers
        self.generator: Generator = generator
        self.top_k = top_k
        self.context_max_items = context_max_items
        self.context_max_characters = context_max_characters

    def run(self, query: str) -> PipelineResult:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")

        total_start = perf_counter()
        profile = profile_query(query)
        route = choose_route(profile)

        retrieval_start = perf_counter()
        retrieval_results = self.retrievers[route].retrieve(query, self.top_k)
        retrieval_latency_ms = (perf_counter() - retrieval_start) * 1000

        context = build_context(
            retrieval_results,
            max_characters=self.context_max_characters,
            max_items=self.context_max_items,
        )
        generation = self.generator.generate(query, context)

        return PipelineResult(
            query=query,
            route=route,
            query_profile=profile,
            retrieval_results=retrieval_results,
            context=context,
            generation=generation,
            retrieval_latency_ms=retrieval_latency_ms,
            total_latency_ms=(perf_counter() - total_start) * 1000,
        )
