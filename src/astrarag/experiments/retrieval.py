"""Retrieval strategies that are not part of the production query path."""

from google import genai
from google.genai import types

from astrarag.embedding import EmbeddingEncoder
from astrarag.indexes import DenseVectorIndex
from astrarag.models import RetrievalResult
from astrarag.retrieval import Retriever


class ParentChildRetriever:
    def __init__(self, child_retriever: Retriever, candidate_k: int = 20) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than 0")
        self.child_retriever = child_retriever
        self.candidate_k = candidate_k

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        children = self.child_retriever.retrieve(query, max(self.candidate_k, top_k))
        results: list[RetrievalResult] = []
        seen: set[str] = set()
        for child in children:
            parent_id = child.metadata.get("parent_id")
            parent_text = child.metadata.get("parent_text")
            if not isinstance(parent_id, str) or not isinstance(parent_text, str):
                continue
            if parent_id in seen:
                continue
            results.append(
                RetrievalResult(
                    chunk_id=parent_id,
                    document_id=child.document_id,
                    text=parent_text,
                    score=child.score,
                    page_numbers=child.page_numbers,
                    metadata={**child.metadata, "matched_child_id": child.chunk_id},
                )
            )
            seen.add(parent_id)
            if len(results) >= top_k:
                break
        return results


class GeminiQueryTransformer:
    def __init__(self, api_key: str, model: str) -> None:
        self.model = model
        self._client = genai.Client(api_key=api_key)

    def generate_queries(self, query: str, count: int = 3) -> list[str]:
        if not query.strip():
            raise ValueError("query cannot be empty")
        if count <= 0:
            raise ValueError("count must be greater than 0")

        response = self._client.models.generate_content(
            model=self.model,
            contents=(
                f"Generate {count} alternative search queries for the following "
                "academic literature retrieval query. Preserve the original meaning. "
                "Return only one query per line, without numbering.\n\n"
                f"Query: {query}"
            ),
            config=types.GenerateContentConfig(temperature=0.0),
        )
        if not response.text:
            raise RuntimeError("Gemini returned an empty query transformation")
        return [line.strip() for line in response.text.splitlines() if line.strip()][
            :count
        ]

    def generate_hypothetical_document(self, query: str) -> str:
        if not query.strip():
            raise ValueError("query cannot be empty")
        response = self._client.models.generate_content(
            model=self.model,
            contents=(
                "Write a short hypothetical scientific passage that directly answers "
                "the following question. Return only the passage.\n\n"
                f"Question: {query}"
            ),
            config=types.GenerateContentConfig(temperature=0.0),
        )
        if not response.text or not response.text.strip():
            raise RuntimeError("Gemini returned an empty hypothetical document")
        return response.text.strip()

    def close(self) -> None:
        self._client.close()


class MultiQueryRetriever:
    def __init__(
        self,
        base: Retriever,
        transformer: GeminiQueryTransformer,
        query_count: int = 3,
        candidate_k: int = 20,
        rrf_k: int = 60,
    ) -> None:
        if min(query_count, candidate_k, rrf_k) <= 0:
            raise ValueError("query_count, candidate_k and rrf_k must be positive")
        self.base = base
        self.transformer = transformer
        self.query_count = query_count
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        queries = [query]
        for item in self.transformer.generate_queries(query, self.query_count):
            if item not in queries:
                queries.append(item)

        scores: dict[str, float] = {}
        results: dict[str, RetrievalResult] = {}
        for search_query in queries:
            for rank, result in enumerate(
                self.base.retrieve(search_query, self.candidate_k), start=1
            ):
                results[result.chunk_id] = result
                scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (
                    self.rrf_k + rank
                )

        ranked_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
        return [
            results[item].model_copy(update={"score": scores[item]})
            for item in ranked_ids
        ]


class HyDERetriever:
    def __init__(
        self,
        encoder: EmbeddingEncoder,
        index: DenseVectorIndex,
        transformer: GeminiQueryTransformer,
    ) -> None:
        self.encoder = encoder
        self.index = index
        self.transformer = transformer

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        hypothetical = self.transformer.generate_hypothetical_document(query)
        return self.index.search(self.encoder.encode_query(hypothetical), top_k)
