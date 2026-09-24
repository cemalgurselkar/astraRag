"""Define evaluation query types, evidence annotations, and dataset records."""

from enum import StrEnum

from pydantic import BaseModel, Field


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
