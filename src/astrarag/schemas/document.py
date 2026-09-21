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