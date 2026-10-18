"""Çalışma ayarlarını ortam değişkenlerinden ve `.env` dosyasından yükler.

Sözlük:
- `Settings`: Model, Qdrant, chunk, retrieval ve API ayarlarının tek kaynağı.
- `Settings.index_metadata`: İndeksi üreten model ve HNSW yapılandırmasını özetler.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str | None = Field(default=None, min_length=1)
    gemini_model: str = "gemini-3.6-flash"

    documents_path: Path = Path("data/documents")
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_path: Path = Path("data/qdrant")
    evaluation_path: Path = Path("data/eval")
    qdrant_collection: str = "astrarag-corpus"
    qdrant_timeout_seconds: int = Field(default=30, gt=0)
    qdrant_hnsw_m: int = Field(default=16, gt=0)
    qdrant_hnsw_ef_construct: int = Field(default=100, gt=0)
    qdrant_hnsw_ef_search: int = Field(default=128, gt=0)
    qdrant_full_scan_threshold_kb: int = Field(default=10, ge=10)
    qdrant_indexing_threshold_kb: int = Field(default=1, ge=0)

    embedding_url: str = "http://127.0.0.1:8081"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = Field(default=384, gt=0)
    embedding_timeout_seconds: float = Field(default=60.0, gt=0)
    embedding_batch_size: int = Field(default=32, gt=0)
    embedding_tokenizer_path: Path = Path("models/all-MiniLM-L6-v2/tokenizer.json")
    embedding_model_revision: str = "ae6369f25f3393dcca54ff7b41157ffd2e43e2e6"
    embedding_model_sha256: str = (
        "7266519c8c9b860e99d561d2cb849cf546b248eb23bcb23787be5b11180daec3"
    )
    embedding_tokenizer_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    embedding_tokenizer_sha256: str = (
        "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"
    )

    chunk_size_tokens: int = Field(default=254, gt=0, le=254)
    chunk_overlap_tokens: int = Field(default=32, ge=0)

    # Retrieval
    retrieval_top_k: int = Field(default=10, gt=0)
    hybrid_rrf_k: int = Field(default=20, gt=0)

    # Context
    context_max_items: int = Field(default=10, gt=0)
    context_max_char: int = Field(default=12_000, gt=0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def index_metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "embedding_model_revision": self.embedding_model_revision,
            "embedding_model_sha256": self.embedding_model_sha256,
            "embedding_tokenizer_revision": self.embedding_tokenizer_revision,
            "embedding_tokenizer_sha256": self.embedding_tokenizer_sha256,
            "chunk_size_tokens": self.chunk_size_tokens,
            "chunk_overlap_tokens": self.chunk_overlap_tokens,
            "hnsw_m": self.qdrant_hnsw_m,
            "hnsw_ef_construct": self.qdrant_hnsw_ef_construct,
            "hnsw_full_scan_threshold_kb": self.qdrant_full_scan_threshold_kb,
            "hnsw_indexing_threshold_kb": self.qdrant_indexing_threshold_kb,
        }
