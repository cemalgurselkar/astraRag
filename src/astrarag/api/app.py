"""Serve AstraRAG health checks and retrieval-augmented queries via FastAPI."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from astrarag import AstraRAG
from astrarag.api.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    rag = AstraRAG()
    rag.start()

    app.state.rag = rag

    try:
        yield
    finally:
        rag.close()


app = FastAPI(
    title="AstraRAG API",
    description=(
        "Performance-oriented adaptive RAG engine "
        "for retrieval and context experimentation."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["RAG"],
)
def query(
    payload: QueryRequest,
    request: Request,
) -> QueryResponse:
    rag: AstraRAG = request.app.state.rag

    try:
        result = rag.query(payload.query)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    sources = [
        SourceResponse(
            chunk_id=item.chunk_id,
            document_id=item.document_id,
            pages=item.page_numbers,
        )
        for item in result.context.items
    ]

    return QueryResponse(
        answer=result.generation.answer,
        route=result.route.value,
        sources=sources,
        retrieval_latency_ms=result.retrieval_latency_ms,
        total_latency_ms=result.total_latency_ms,
    )
