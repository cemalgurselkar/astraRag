"""Deterministic API runtime built on the real pipeline contracts."""

from astrarag.api.app import create_app
from astrarag.models import (
    ContextItem,
    ContextPackage,
    GeneratedAnswer,
    PipelineResult,
    QueryProfile,
    RetrievalResult,
    RetrievalRoute,
)

_RETRIEVAL_LATENCY_MS = 1.25
_TOTAL_LATENCY_MS = 3.5

_QUERIES: dict[str, tuple[RetrievalRoute, list[tuple[str, str, list[int]]]]] = {
    "multi": (
        RetrievalRoute.MEDIUM,
        [
            ("chunk-1", "document-1", [2]),
            ("chunk-2", "document-1", [3, 4]),
            ("chunk-3", "document-2", [7]),
        ],
    ),
}

_DEFAULT_SOURCES: list[tuple[str, str, list[int]]] = [
    ("chunk-1", "document-1", [2]),
]


def _profile() -> QueryProfile:
    return QueryProfile(
        word_count=4,
        has_numbers=False,
        has_comparison=False,
        has_multi_hop_signal=False,
        has_exact_phrase=False,
    )


def _retrieval_results(specs: list[tuple[str, str, list[int]]]) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id=chunk_id,
            document_id=document_id,
            text=f"Evidence from {chunk_id}.",
            score=1.0,
            page_numbers=pages,
            metadata={"filename": "sample.pdf"},
        )
        for chunk_id, document_id, pages in specs
    ]


def _context(
    specs: list[tuple[str, str, list[int]]],
) -> ContextPackage:
    items = [
        ContextItem(
            chunk_id=chunk_id,
            document_id=document_id,
            text=f"Evidence from {chunk_id}.",
            score=1.0,
            page_numbers=pages,
            metadata={"filename": "sample.pdf"},
        )
        for chunk_id, document_id, pages in specs
    ]
    sections = [
        f"[SOURCE {position}]\nContent:\n{item.text}"
        for position, item in enumerate(items, start=1)
    ]
    text = "\n\n".join(sections)
    return ContextPackage(items=items, text=text, total_characters=len(text))


class FakeRAG:
    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def query(self, query: str) -> PipelineResult:
        if not query.strip():
            raise ValueError("query must not be empty")
        if query == "invalid":
            raise ValueError("invalid query")
        if query == "unavailable":
            raise RuntimeError("service unavailable")

        route, specs = _QUERIES.get(query, (RetrievalRoute.CHEAP, _DEFAULT_SOURCES))
        retrieval_results = _retrieval_results(specs)
        context = _context(specs)

        return PipelineResult(
            query=query.strip(),
            route=route,
            query_profile=_profile(),
            retrieval_results=retrieval_results,
            context=context,
            generation=GeneratedAnswer(
                answer="Grounded answer [SOURCE 1].",
                model="fake-model",
                source_ids=[item.chunk_id for item in context.items],
            ),
            retrieval_latency_ms=_RETRIEVAL_LATENCY_MS,
            total_latency_ms=_TOTAL_LATENCY_MS,
        )


app = create_app(FakeRAG)
