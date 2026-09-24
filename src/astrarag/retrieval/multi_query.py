"""Expand a question into multiple queries and fuse their retrieval rankings."""

from astrarag.retrieval.query_transformer import GeminiQueryTransformer
from astrarag.schemas import RetrievalResult


class MultiQueryRetriever:
    def __init__(
        self,
        base_retriever,
        transformer: GeminiQueryTransformer,
        query_count: int = 3,
        candidate_k: int = 20,
        rrf_k: int = 60,
    ) -> None:
        if query_count <= 0:
            raise ValueError("query_count must be greater than 0")

        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than 0")

        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than 0")

        self.base_retriever = base_retriever
        self.transformer = transformer
        self.query_count = query_count
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def retriever(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        transformed = self.transformer.generate_queries(
            query=query,
            count=self.query_count,
        )

        queries = [query]

        for candidate in transformed:
            if candidate not in queries:
                queries.append(candidate)

        fused_scores: dict[str, float] = {}
        results_by_id: dict[str, RetrievalResult] = {}

        for search_query in queries:
            results = self.base_retriever.retriever(
                query=search_query,
                top_k=self.candidate_k,
            )

            for rank, result in enumerate(results, start=1):
                results_by_id[result.chunk_id] = result

                fused_scores[result.chunk_id] = (
                    fused_scores.get(result.chunk_id, 0.0)
                    + 1.0 / (self.rrf_k + rank)
                )

        ranked_ids = sorted(
            fused_scores,
            key=fused_scores.__getitem__,
            reverse=True,
        )[:top_k]

        return [
            results_by_id[chunk_id].model_copy(
                update={
                    "score": fused_scores[chunk_id],
                }
            )
            for chunk_id in ranked_ids
        ]
