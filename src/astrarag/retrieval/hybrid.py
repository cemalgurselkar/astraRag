from astrarag.retrieval.dense import DenseRetriever
from astrarag.retrieval.sparse import BM25Retriever
from astrarag.schemas import RetrievalResult

class HybridRetriever:
    def __init__(self, dense_retriever: DenseRetriever, sparse_retriever: BM25Retriever, rrf_k: int = 60) -> None:
        self.dense_retri = dense_retriever
        self.sparse_retri = sparse_retriever
        self.rrf_k = rrf_k
    
    def retriever(self,
                  query: str,
                  top_k: int= 5,
                  candidate_k: int=20) -> list[RetrievalResult]:
        
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        
        if candidate_k < top_k:
            raise ValueError("candidate must be >= top_k")
        
        dense_results = self.dense_retri.retriever(
            query=query,
            top_k=candidate_k)
        
        sparse_results = self.sparse_retri.retriever(
            query=query,
            top_k=candidate_k
        )
        
        fused_scores: dict[str, float] = {}
        results_by_id: dict[str, RetrievalResult] = {}
        
        for results in (dense_results, sparse_results):
            for rank, result in enumerate(results, start=1):
                results_by_id[result.chunk_id] = result
                
                fused_scores[result.chunk_id] = (
                    fused_scores.get(result.chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
                )
        
        ranked_ids = sorted(
            fused_scores,
            key=fused_scores.__getitem__,
            reverse=True
        )[:top_k]
        
        return [
            results_by_id[chunk_id].model_copy(
                update={"score": fused_scores[chunk_id]}
            ) for chunk_id in ranked_ids
        ]