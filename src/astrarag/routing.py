"""Sorgunun yapısını analiz eder ve uygun retrieval yolunu seçer.

Sözlük:
- `profile_query`: Uzunluk, sayı, karşılaştırma ve multi-hop sinyallerini çıkarır.
- `choose_route`: Sorgu profiline göre BM25 veya hibrit retrieval seçer.
"""

import re

from astrarag.models import QueryProfile, RetrievalRoute

_COMPARISON_PATTERNS = (
    "compare",
    "comparison",
    "difference",
    "differences",
    "differ",
    "versus",
    " vs ",
    "similarities",
    "similar",
)

_MULTI_HOP_PATTERNS = (
    "relationship between",
    "relation between",
    "how does",
    "why does",
    "how do",
    "why do",
    "across",
    "both",
)


def profile_query(query: str) -> QueryProfile:
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")

    normalized = query.casefold()
    return QueryProfile(
        word_count=len(normalized.split()),
        has_numbers=bool(re.search(r"\d", query)),
        has_comparison=any(item in normalized for item in _COMPARISON_PATTERNS),
        has_multi_hop_signal=any(item in normalized for item in _MULTI_HOP_PATTERNS),
        has_exact_phrase=bool(re.search(r'"[^"]+"', query)),
    )


def choose_route(profile: QueryProfile) -> RetrievalRoute:
    if profile.has_comparison or profile.has_multi_hop_signal:
        return RetrievalRoute.MEDIUM
    if profile.has_numbers or profile.has_exact_phrase or profile.word_count >= 12:
        return RetrievalRoute.MEDIUM
    return RetrievalRoute.CHEAP
