from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer


class EmbeddingEncoder:
    def __init__(
        self,
        model_name: str | Path = ("models/all-MiniLM-L6-v2"),
    ) -> None:

        self.model_name = model_name
        self._model = SentenceTransformer(str(model_name), local_files_only=True)

    @property
    def dimension(self) -> int:
        dimension = self._model.get_embedding_dimension()

        if dimension is None:
            raise RuntimeError(
                f"Could not determine embedding dimension for {self.model_name}"
            )

        return dimension

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            raise ValueError("texts cannot be empty")

        embeddings = self._model.encode(
            list(texts), convert_to_numpy=True, normalize_embeddings=True
        )

        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> NDArray[np.float32]:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        embeddings = self.encode([query])

        return embeddings[0]
