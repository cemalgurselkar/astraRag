"""Serve AstraRAG health checks and queries through FastAPI."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from astrarag.api.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceResponse,
)


def _create_rag() -> Any:
    from astrarag import AstraRAG

    return AstraRAG()


def create_app(rag_factory: Callable[[], Any] = _create_rag) -> FastAPI:
    """Create the HTTP application with an overridable runtime factory."""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        rag = rag_factory()
        rag.start()
        application.state.rag = rag
        try:
            yield
        finally:
            rag.close()

    application = FastAPI(
        title="AstraRAG API",
        description=(
            "Performance-oriented adaptive RAG engine "
            "for retrieval and context experimentation."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    application.add_api_route(
        "/health",
        health,
        methods=["GET"],
        response_model=HealthResponse,
        tags=["System"],
    )
    application.add_api_route(
        "/query",
        query,
        methods=["POST"],
        response_model=QueryResponse,
        tags=["RAG"],
    )
    return application


def health() -> HealthResponse:
    return HealthResponse(status="ok")


def query(payload: QueryRequest, request: Request) -> QueryResponse:
    rag = request.app.state.rag

    try:
        result = rag.query(payload.query)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return QueryResponse(
        answer=result.generation.answer,
        route=result.route.value,
        sources=[
            SourceResponse(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                pages=item.page_numbers,
            )
            for item in result.context.items
        ],
        retrieval_latency_ms=result.retrieval_latency_ms,
        total_latency_ms=result.total_latency_ms,
    )


app = create_app()
