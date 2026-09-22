from collections.abc import Sequence

from sentence_transformers import CrossEncoder

from astrarag.schemas import RetrievalResult


class CrossEncoderReRanker:
    
    def __init__(self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",) ->None:
        
        self.model_name = model_name
        self._model = CrossEncoder(model_name)
    
    def rerank(self,
               query: str,
               candidates: Sequence[RetrievalResult],
               top_k: int = 10) -> list[RetrievalResult]:
        
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        
        if not candidates:
            return []
        
        pairs = [
            (query, candidate.text) for candidate in candidates
        ]
        
        scores = self._model.predict(
            pairs,
            show_progress_bar=False,
        )
        
        ranked = sorted(
            zip(candidates, scores, strict=True),
            key= lambda item: float(item[1]),
            reverse=True
        )
        
        return [
            candidate.model_copy(
                update={"scores": float(score)}
            ) for candidate, score in ranked[:top_k]
        ]