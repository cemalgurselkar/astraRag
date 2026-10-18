"""Metin embedding'lerini yerel llama.cpp servisinden alır ve doğrular.

Sözlük:
- `EmbeddingEncoder.encode`: Metin grubunu gönderir, vektörleri sıralar ve normalize eder.
- `EmbeddingEncoder.encode_query`: Tek bir sorgu için embedding üretir.
- `EmbeddingEncoder.check_health`: llama.cpp servisinin erişilebilirliğini denetler.
- `EmbeddingEncoder.close`: Kalıcı HTTP bağlantılarını kapatır.
"""

from collections.abc import Sequence

import httpx
import numpy as np
from numpy.typing import NDArray


class EmbeddingEncoder:
    """Call llama.cpp's OpenAI-compatible embedding endpoint."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8081",
        model: str = "all-MiniLM-L6-v2",
        dimension: int = 384,
        timeout_seconds: float = 60.0,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than 0")
        self.model = model
        self._dimension = dimension
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            raise ValueError("texts cannot be empty")
        try:
            response = self._client.post(
                "/v1/embeddings",
                json={
                    "model": self.model,
                    "input": list(texts),
                    "encoding_format": "float",
                },
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise RuntimeError(f"llama.cpp embedding request failed: {exc}") from exc

        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise RuntimeError(  # noqa: TRY004 - external service contract failure
                "llama.cpp returned an invalid embedding response"
            )
        try:
            data = sorted(payload["data"], key=lambda item: item["index"])
            vectors = [item["embedding"] for item in data]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                "llama.cpp returned an invalid embedding response"
            ) from exc
        if len(data) != len(texts):
            raise RuntimeError("llama.cpp returned an unexpected number of embeddings")

        embeddings = np.asarray(vectors, dtype=np.float32)
        expected_shape = (len(texts), self.dimension)
        if embeddings.shape != expected_shape:
            raise RuntimeError(
                f"Expected embedding shape {expected_shape}, got {embeddings.shape}"
            )
        if not np.isfinite(embeddings).all():
            raise RuntimeError("llama.cpp returned a non-finite embedding")

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise RuntimeError("llama.cpp returned a zero-length embedding")
        return np.asarray(embeddings / norms, dtype=np.float32)

    def encode_query(self, query: str) -> NDArray[np.float32]:
        if not query.strip():
            raise ValueError("query cannot be empty")
        return self.encode([query])[0]

    def check_health(self) -> None:
        try:
            response = self._client.get("/health")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"llama.cpp embedding service is unavailable: {exc}"
            ) from exc

    def close(self) -> None:
        self._client.close()
