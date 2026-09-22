from astrarag.reranking import CrossEncoderReRanker
from astrarag.schemas import RetrievalResult

class RerankedRetriever:
    
    def __init__(self,
            base_retriever,
            reranker: CrossEncoderReRanker,
            candidate_k: int = 20) -> None:
        
        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than 0")
        
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.candidate_k = candidate_k
        
    def retriever(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        
        candidate_k = max(self.candidate_k, top_k)
        
        candidates = self.base_retriever.retriever(
            query=query,
            top_k=candidate_k
        )
        
        return self.reranker.rerank(
            query=query, candidates=candidates, top_k=top_k
        )