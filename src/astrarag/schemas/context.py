from pydantic import BaseModel, Field


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
