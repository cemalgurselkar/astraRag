"""Verify query profiling and rule-based selection of retrieval cost tiers."""

import pytest

from astrarag.routing import QueryProfiler, RuleBasedRouter
from astrarag.schemas import RetrievalRoute


@pytest.fixture
def profiler() -> QueryProfiler:
    return QueryProfiler()


@pytest.fixture
def router() -> RuleBasedRouter:
    return RuleBasedRouter()


def test_simple_query_routes_to_cheap(
    profiler: QueryProfiler,
    router: RuleBasedRouter,
) -> None:
    profile = profiler.profile("What is self attention?")

    assert router.route(profile) == RetrievalRoute.CHEAP


def test_comparison_routes_to_expensive(
    profiler: QueryProfiler,
    router: RuleBasedRouter,
) -> None:
    profile = profiler.profile("Compare BERT and Sentence-BERT.")

    assert router.route(profile) == RetrievalRoute.EXPENSIVE


def test_multi_hop_signal_routes_to_expensive(
    profiler: QueryProfiler,
    router: RuleBasedRouter,
) -> None:
    profile = profiler.profile(
        "What is the relationship between dense retrieval and reranking?"
    )

    assert router.route(profile) == RetrievalRoute.EXPENSIVE


def test_numeric_query_routes_to_medium(
    profiler: QueryProfiler,
    router: RuleBasedRouter,
) -> None:
    profile = profiler.profile("What happens with 512 tokens?")

    assert router.route(profile) == RetrievalRoute.MEDIUM


def test_exact_phrase_routes_to_medium(
    profiler: QueryProfiler,
    router: RuleBasedRouter,
) -> None:
    profile = profiler.profile('What does "late interaction" mean?')

    assert router.route(profile) == RetrievalRoute.MEDIUM


def test_empty_query_raises_error(
    profiler: QueryProfiler,
) -> None:
    with pytest.raises(
        ValueError,
        match="query must not be empty",
    ):
        profiler.profile("   ")
