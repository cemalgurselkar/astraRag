"""Verify the llama.cpp embedding response contract."""

import httpx
import numpy as np
import pytest

from astrarag.embedding import EmbeddingEncoder


def encoder_with_response(payload: dict[str, object]) -> EmbeddingEncoder:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        assert request.url.path == "/v1/embeddings"
        return httpx.Response(200, json=payload)

    encoder = EmbeddingEncoder(dimension=3)
    encoder._client.close()
    encoder._client = httpx.Client(
        base_url="http://embedding.test",
        transport=httpx.MockTransport(handler),
    )
    return encoder


def test_encoder_orders_and_normalizes_llama_embeddings() -> None:
    encoder = encoder_with_response(
        {
            "data": [
                {"index": 1, "embedding": [0.0, 3.0, 4.0]},
                {"index": 0, "embedding": [2.0, 0.0, 0.0]},
            ]
        }
    )

    try:
        embeddings = encoder.encode(["first", "second"])
    finally:
        encoder.close()

    assert embeddings.dtype == np.float32
    assert np.allclose(
        embeddings,
        [[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]],
    )


def test_encoder_rejects_the_wrong_dimension() -> None:
    encoder = encoder_with_response({"data": [{"index": 0, "embedding": [1.0, 2.0]}]})

    try:
        with pytest.raises(RuntimeError, match="Expected embedding shape"):
            encoder.encode(["text"])
    finally:
        encoder.close()


@pytest.mark.parametrize("payload", [{}, {"data": [{}]}, {"data": "invalid"}])
def test_encoder_rejects_malformed_responses(payload: dict[str, object]) -> None:
    encoder = encoder_with_response(payload)

    try:
        with pytest.raises(RuntimeError, match="invalid embedding response"):
            encoder.encode(["text"])
    finally:
        encoder.close()


def test_encoder_checks_llama_server_health() -> None:
    encoder = encoder_with_response({"data": []})

    try:
        encoder.check_health()
    finally:
        encoder.close()
