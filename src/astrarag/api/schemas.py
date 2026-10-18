"""HTTP API'nin doğrulanmış istek ve cevap modellerini tanımlar.

Sözlük:
- `QueryRequest`: Kullanıcının sorgu metni.
- `SourceResponse`: Cevabı destekleyen belge parçası ve sayfaları.
- `QueryResponse`: Cevap, route, kaynak ve gecikme bilgileri.
- `HealthResponse`: Servis sağlık durumunu taşıyan cevap.
"""

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
