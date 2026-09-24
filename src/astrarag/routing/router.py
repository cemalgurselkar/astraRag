"""Map query profiles to cheap, medium, or expensive retrieval routes."""

from astrarag.schemas import QueryProfile, RetrievalRoute


class RuleBasedRouter:
    def route(self, profile: QueryProfile) -> RetrievalRoute:
        if profile.has_comparison or profile.has_multi_hop_signal:
            return RetrievalRoute.EXPENSIVE

        if profile.has_numbers or profile.has_exact_phrase or profile.word_count >= 12:
            return RetrievalRoute.MEDIUM

        return RetrievalRoute.CHEAP
