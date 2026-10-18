"""AstraRAG katmanları arasında kullanılan doğrulanmış veri sözleşmelerini tanımlar.

Sözlük:
- `Page`, `Document`, `Chunk`: Kaynak PDF'in sayfa, belge ve parça modelleri.
- `RetrievalResult`: Retriever tarafından döndürülen sıralı arama sonucu.
- `ContextItem`, `ContextPackage`: Üretim modeline gönderilecek kanıt bağlamı.
- `GeneratedAnswer`: Model cevabı ve kullanılan kaynak kimlikleri.
- `RetrievalRoute`, `QueryProfile`: Sorgunun özellikleri ve seçilen arama yolu.
- `PipelineResult`: Tüm RAG çalışmasının cevabı, kaynakları ve süreleri.
- `QueryType`, `Evidence`, `EvaluationQuery`: Değerlendirme veri seti modelleri.
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Page(BaseModel):
    number: int = Field(ge=1)
    text: str = Field(min_length=1)


class Document(BaseModel):
    id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    pages: list[Page] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    index: int = Field(ge=0)
    page_numbers: list[int] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    page_numbers: list[int] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)


class ContextItem(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    page_numbers: list[int] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)


class ContextPackage(BaseModel):
    items: list[ContextItem]
    text: str
    total_characters: int


class GeneratedAnswer(BaseModel):
    answer: str = Field(min_length=1)
    model: str = Field(min_length=1)
    source_ids: list[str] = Field(default_factory=list)


class RetrievalRoute(StrEnum):
    CHEAP = "cheap"
    MEDIUM = "medium"


class QueryProfile(BaseModel):
    word_count: int = Field(ge=0)
    has_numbers: bool
    has_comparison: bool
    has_multi_hop_signal: bool
    has_exact_phrase: bool


class PipelineResult(BaseModel):
    query: str = Field(min_length=1)
    route: RetrievalRoute
    query_profile: QueryProfile
    retrieval_results: list[RetrievalResult]
    context: ContextPackage
    generation: GeneratedAnswer
    retrieval_latency_ms: float = Field(ge=0.0)
    total_latency_ms: float = Field(ge=0.0)


class QueryType(StrEnum):
    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    NUMERICAL = "numerical"
    COMPARISON = "comparison"
    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"
    GLOBAL = "global"
    UNANSWERABLE = "unanswerable"


class Evidence(BaseModel):
    document: str = Field(min_length=1)
    text: str = Field(min_length=1)


class EvaluationQuery(BaseModel):
    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    query_type: QueryType
    evidence: list[Evidence] = Field(min_length=1)
