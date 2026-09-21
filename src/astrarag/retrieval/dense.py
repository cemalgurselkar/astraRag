from astrarag.embedding import EmbeddingEncoder
from astrarag.index import DenseVectorIndex
from astrarag.schemas import RetrievalResult


class DenseRetriever:
    def __init__(self,
            encoder: EmbeddingEncoder,
            index: DenseVectorIndex) -> None:
        
        self.encoder = encoder
        self.index = index
    
    def retriever(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        
        if not query.strip():
            raise ValueError("query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        
        query_vector = self.encoder.encode_query(query)
        
        return self.index.search(
            query_vector=query_vector,
            limit=top_k
        )