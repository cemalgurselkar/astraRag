"""Optional live smoke test using local models, Qdrant, and Gemini."""

import os

import httpx
import pytest

from astrarag.api.app import app


@pytest.mark.live
@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("RUN_LIVE_E2E") != "1",
    reason="set RUN_LIVE_E2E=1 to call the real Gemini-backed service",
)
async def test_live_query_returns_an_answer_and_sources() -> None:
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client,
    ):
        response = await client.post(
            "/query",
            json={"query": "What is the main idea behind Sentence-BERT?"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert payload["route"] in {"cheap", "medium", "expensive"}
    assert payload["sources"]
