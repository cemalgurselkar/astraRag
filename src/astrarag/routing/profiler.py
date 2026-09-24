"""Extract lexical signals that describe a query's retrieval complexity."""

import re

from astrarag.schemas import QueryProfile


class QueryProfiler:
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

    def profile(self, query: str) -> QueryProfile:
        query = query.strip()

        if not query:
            raise ValueError("query must not be empty")

        normalized = query.casefold()
        words = normalized.split()

        return QueryProfile(
            word_count=len(words),
            has_numbers=bool(re.search(r"\d", query)),
            has_comparison=self._contains_any(
                normalized,
                self._COMPARISON_PATTERNS,
            ),
            has_multi_hop_signal=self._contains_any(
                normalized,
                self._MULTI_HOP_PATTERNS,
            ),
            has_exact_phrase=self._has_exact_phrase(query),
        )

    @staticmethod
    def _contains_any(query: str, patterns: tuple[str, ...]) -> bool:
        return any(pattern in query for pattern in patterns)

    @staticmethod
    def _has_exact_phrase(query: str) -> bool:
        return bool(re.search(r'"[^"]+"', query))
