"""Verify query profiling and retrieval route selection."""

import pytest

from astrarag.models import RetrievalRoute
from astrarag.routing import choose_route, profile_query


def test_simple_query_routes_to_cheap() -> None:
    assert (
        choose_route(profile_query("What is self attention?")) == RetrievalRoute.CHEAP
    )


def test_comparison_routes_to_medium() -> None:
    profile = profile_query("Compare BERT and Sentence-BERT.")
    assert choose_route(profile) == RetrievalRoute.MEDIUM


def test_multi_hop_signal_routes_to_medium() -> None:
    profile = profile_query(
        "What is the relationship between dense retrieval and reranking?"
    )
    assert choose_route(profile) == RetrievalRoute.MEDIUM


def test_numeric_query_routes_to_medium() -> None:
    assert (
        choose_route(profile_query("What happens with 512 tokens?"))
        == RetrievalRoute.MEDIUM
    )


def test_exact_phrase_routes_to_medium() -> None:
    profile = profile_query('What does "late interaction" mean?')
    assert choose_route(profile) == RetrievalRoute.MEDIUM


def test_empty_query_raises_error() -> None:
    with pytest.raises(ValueError, match="query must not be empty"):
        profile_query("   ")
