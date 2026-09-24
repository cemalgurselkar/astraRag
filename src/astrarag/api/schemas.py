"""Define the request and response models exposed by the HTTP API."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)


class SourceResponse(BaseModel):
    chunk_id: str
    document_id: str
    pages: list[int]


class QueryResponse(BaseModel):
    answer: str
    route: str

    sources: list[SourceResponse]

    retrieval_latency_ms: float
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: str
