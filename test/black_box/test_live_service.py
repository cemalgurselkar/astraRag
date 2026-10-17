"""Optional live smoke test using local models, Qdrant, and Gemini."""

import os

import httpx
import pytest

from astrarag.api.app import app

QUERY_RESPONSE_KEYS = {
    "answer",
    "route",
    "sources",
    "retrieval_latency_ms",
    "total_latency_ms",
}

SOURCE_KEYS = {"chunk_id", "document_id", "pages"}


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("RUN_LIVE_E2E") != "1",
    reason="set RUN_LIVE_E2E=1 to call the real Gemini-backed service",
)
async def test_live_query_returns_a_valid_contract() -> None:
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            timeout=60,
        ) as client,
    ):
        response = await client.post(
            "/query",
            json={"query": "What is the main idea behind Sentence-BERT?"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == QUERY_RESPONSE_KEYS

    assert isinstance(payload["answer"], str)
    assert payload["answer"].strip()
    assert payload["route"] in {"cheap", "medium", "expensive"}

    sources = payload["sources"]
    assert isinstance(sources, list)
    assert sources
    for source in sources:
        assert set(source) == SOURCE_KEYS
        assert isinstance(source["chunk_id"], str) and source["chunk_id"]
        assert isinstance(source["document_id"], str) and source["document_id"]
        assert isinstance(source["pages"], list)
        assert all(
            isinstance(page, int) and page >= 1 for page in source["pages"]
        )

    assert isinstance(payload["retrieval_latency_ms"], (int, float))
    assert payload["retrieval_latency_ms"] >= 0
    assert payload["total_latency_ms"] >= payload["retrieval_latency_ms"]


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("RUN_LIVE_E2E") != "1",
    reason="set RUN_LIVE_E2E=1 to call the real Gemini-backed service",
)
async def test_live_query_rejects_an_empty_query() -> None:
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client,
    ):
        response = await client.post("/query", json={"query": ""})

    assert response.status_code == 422
