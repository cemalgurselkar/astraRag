"""Load AstraRAG runtime settings from environment variables and dotenv files."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    gemini_api_key: str = Field(min_length=1)
    gemini_model: str = "gemini-3.6-flash"

    documents_path: Path = Path("data/documents")
    qdrant_path: Path = Path("data/qdrant")
    evaluation_path: Path = Path("data/eval")
    qdrant_collection: str = "astrarag-corpus"

    embedding_model_path: Path = Path("models/all-MiniLM-L6-v2")
    reranker_model_path: Path = Path("models/ms-marco-MiniLM-L6-v2")

    # Retrieval
    retrieval_top_k: int = Field(default=10, gt=0)
    hybrid_rrf_k: int = Field(default=20, gt=0)
    reranker_candidate_k: int = Field(default=20, gt=0)

    # Context
    context_max_items: int = Field(default=10, gt=0)
    context_max_char: int = Field(default=12_000, gt=0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
