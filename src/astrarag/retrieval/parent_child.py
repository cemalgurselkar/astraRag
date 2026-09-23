from astrarag.schemas import RetrievalResult


class ParentChildRetriever:
    """Retrieve small child chunks and return their larger parent context."""

    def __init__(
        self,
        child_retriever,
        candidate_k: int = 20,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than 0")

        self.child_retriever = child_retriever
        self.candidate_k = candidate_k

    def retriever(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        child_results = self.child_retriever.retriever(
            query=query,
            top_k=max(self.candidate_k, top_k),
        )

        results: list[RetrievalResult] = []
        seen_parent_ids: set[str] = set()

        for child in child_results:
            parent_id = child.metadata.get("parent_id")
            parent_text = child.metadata.get("parent_text")

            if not isinstance(parent_id, str):
                continue

            if not isinstance(parent_text, str):
                continue

            if parent_id in seen_parent_ids:
                continue

            results.append(
                RetrievalResult(
                    chunk_id=parent_id,
                    document_id=child.document_id,
                    text=parent_text,
                    score=child.score,
                    page_numbers=child.page_numbers,
                    metadata={
                        **child.metadata,
                        "matched_child_id": child.chunk_id,
                    },
                )
            )

            seen_parent_ids.add(parent_id)

            if len(results) >= top_k:
                break

        return results
