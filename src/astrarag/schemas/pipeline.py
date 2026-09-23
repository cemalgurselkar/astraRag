from pydantic import BaseModel, Field

from astrarag.schemas.context import ContextPackage
from astrarag.schemas.generation import GeneratedAnswer
from astrarag.schemas.retrieval import RetrievalResult
from astrarag.schemas.routing import QueryProfile, RetrievalRoute


class PipelineResult(BaseModel):
    query: str = Field(min_length=1)
    route: RetrievalRoute
    query_profile: QueryProfile
    retrieval_results: list[RetrievalResult]
    context: ContextPackage
    generation: GeneratedAnswer
    retrieval_latency_ms: float = Field(ge=0.0)
    total_latency_ms: float = Field(ge=0.0)
