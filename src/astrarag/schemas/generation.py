from pydantic import BaseModel, Field


class GeneratedAnswer(BaseModel):
    answer: str = Field(min_length=1)
    model: str = Field(min_length=1)
    source_ids: list[str] = Field(default_factory=list)
