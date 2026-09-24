"""Define retrieval route levels and the query signals used to select them."""

from enum import StrEnum

from pydantic import BaseModel, Field


class RetrievalRoute(StrEnum):
    CHEAP = "cheap"
    MEDIUM = "medium"
    EXPENSIVE = "expensive"


class QueryProfile(BaseModel):
    word_count: int = Field(ge=0)
    has_numbers: bool
    has_comparison: bool
    has_multi_hop_signal: bool
    has_exact_phrase: bool
