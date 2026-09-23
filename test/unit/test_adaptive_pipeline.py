from astrarag.context import ContextEngine
from astrarag.pipeline import AdaptiveRAGPipeline
from astrarag.routing import QueryProfiler, RuleBasedRouter
from astrarag.schemas import (
    GeneratedAnswer,
    RetrievalResult,
    RetrievalRoute,
)


class FakeRetriever:
    def __init__(self, name: str) -> None:
        self.name = name

    def retriever(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                chunk_id=f"{self.name}-chunk",
                document_id="test-document",
                text=f"Evidence from {self.name}.",
                score=1.0,
                page_numbers=[1],
                metadata={
                    "filename": "test.pdf",
                },
            )
        ]


class FakeGenerator:
    def generate(
        self,
        query: str,
        context,
    ) -> GeneratedAnswer:
        return GeneratedAnswer(
            answer="Generated answer [SOURCE 1].",
            model="fake-model",
            source_ids=[item.chunk_id for item in context.items],
        )


def build_pipeline() -> AdaptiveRAGPipeline:
    return AdaptiveRAGPipeline(
        profiler=QueryProfiler(),
        router=RuleBasedRouter(),
        retrievers={
            RetrievalRoute.CHEAP: FakeRetriever("cheap"),
            RetrievalRoute.MEDIUM: FakeRetriever("medium"),
            RetrievalRoute.EXPENSIVE: FakeRetriever("expensive"),
        },
        context_engine=ContextEngine(),
        generator=FakeGenerator(),
    )


def test_pipeline_uses_cheap_route() -> None:
    pipeline = build_pipeline()

    result = pipeline.run("What is self attention?")

    assert result.route == RetrievalRoute.CHEAP
    assert result.retrieval_results[0].chunk_id == "cheap-chunk"
    assert result.generation.answer


def test_pipeline_uses_medium_route() -> None:
    pipeline = build_pipeline()

    result = pipeline.run("What happens with 512 tokens?")

    assert result.route == RetrievalRoute.MEDIUM
    assert result.retrieval_results[0].chunk_id == "medium-chunk"


def test_pipeline_uses_expensive_route() -> None:
    pipeline = build_pipeline()

    result = pipeline.run("Compare BERT and Sentence-BERT.")

    assert result.route == RetrievalRoute.EXPENSIVE
    assert result.retrieval_results[0].chunk_id == "expensive-chunk"


def test_pipeline_records_latency() -> None:
    pipeline = build_pipeline()

    result = pipeline.run("What is BERT?")

    assert result.retrieval_latency_ms >= 0
    assert result.total_latency_ms >= 0


def test_pipeline_context_contains_retrieval() -> None:
    pipeline = build_pipeline()

    result = pipeline.run("What is BERT?")

    assert result.context.items
    assert "Evidence from cheap." in result.context.text
