from astrarag.embedding import EmbeddingEncoder
from astrarag.index import DenseVectorIndex
from astrarag.retrieval.query_transformer import GeminiQueryTransformer
from astrarag.schemas import RetrievalResult


class HyDERetriever:
    def __init__(
        self,
        encoder: EmbeddingEncoder,
        index: DenseVectorIndex,
        transformer: GeminiQueryTransformer,
    ) -> None:
        self.encoder = encoder
        self.index = index
        self.transformer = transformer

    def retriever(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        hypothetical_document = (
            self.transformer.generate_hypothetical_document(
                query=query,
            )
        )

        query_vector = self.encoder.encode_query(
            hypothetical_document
        )

        return self.index.search(
            query_vector=query_vector,
            limit=top_k,
        )