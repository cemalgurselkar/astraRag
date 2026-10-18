"""Exercise the API through a real HTTP server process."""

from pathlib import Path

from astrarag.models import PipelineResult
from test.black_box.fake_app import FakeRAG, app
from test.black_box.server import request_json, run_failing_server


def test_openapi_exposes_the_public_routes() -> None:
    schema = app.openapi()

    assert set(schema["paths"]) == {"/health", "/query"}
    assert schema["paths"]["/health"]["get"]["responses"]["200"]
    assert schema["paths"]["/query"]["post"]["responses"]["200"]


def test_fake_runtime_matches_the_real_pipeline_contract() -> None:
    result = FakeRAG().query("What is attention?")

    assert isinstance(result, PipelineResult)


def test_health_endpoint(api_server: str) -> None:
    status, payload = request_json(f"{api_server}/health")

    assert (status, payload) == (200, {"status": "ok"})


def test_query_success_payload(api_server: str) -> None:
    status, payload = request_json(
        f"{api_server}/query",
        payload={"query": "What is attention?"},
    )

    assert status == 200
    assert set(payload) == {
        "answer",
        "route",
        "sources",
        "retrieval_latency_ms",
        "total_latency_ms",
    }
    assert payload["answer"] == "Grounded answer [SOURCE 1]."
    assert payload["route"] == "cheap"
    assert payload["sources"] == [
        {
            "chunk_id": "chunk-1",
            "document_id": "document-1",
            "pages": [2],
        }
    ]
    assert payload["retrieval_latency_ms"] == 1.25
    assert payload["total_latency_ms"] == 3.5
    assert payload["total_latency_ms"] >= payload["retrieval_latency_ms"]


def test_query_preserves_source_order(api_server: str) -> None:
    status, payload = request_json(
        f"{api_server}/query",
        payload={"query": "multi"},
    )

    assert status == 200
    assert payload["route"] == "medium"
    assert [source["chunk_id"] for source in payload["sources"]] == [
        "chunk-1",
        "chunk-2",
        "chunk-3",
    ]
    assert [source["pages"] for source in payload["sources"]] == [
        [2],
        [3, 4],
        [7],
    ]


def test_query_rejects_an_empty_string_with_422(api_server: str) -> None:
    status, _ = request_json(
        f"{api_server}/query",
        payload={"query": ""},
    )

    assert status == 422


def test_query_maps_a_whitespace_query_to_400(api_server: str) -> None:
    status, payload = request_json(
        f"{api_server}/query",
        payload={"query": "   "},
    )

    assert status == 400
    assert "query" in payload["detail"]


def test_query_maps_value_errors_to_400(api_server: str) -> None:
    status, payload = request_json(
        f"{api_server}/query",
        payload={"query": "invalid"},
    )

    assert (status, payload) == (400, {"detail": "invalid query"})


def test_query_maps_runtime_errors_to_503(api_server: str) -> None:
    status, payload = request_json(
        f"{api_server}/query",
        payload={"query": "unavailable"},
    )

    assert (status, payload) == (503, {"detail": "service unavailable"})


def test_query_ignores_unknown_request_fields(api_server: str) -> None:
    status, payload = request_json(
        f"{api_server}/query",
        payload={"query": "What is attention?", "unrelated": 42},
    )

    assert status == 200
    assert payload["answer"] == "Grounded answer [SOURCE 1]."


def test_query_rejects_a_missing_query_field(api_server: str) -> None:
    status, _ = request_json(f"{api_server}/query", payload={})

    assert status == 422


def test_query_rejects_malformed_json(api_server: str) -> None:
    status, _ = request_json(f"{api_server}/query", raw=b"{not json")

    assert status == 422


def test_unknown_route_returns_404(api_server: str) -> None:
    status, payload = request_json(f"{api_server}/does-not-exist")

    assert status == 404
    assert "detail" in payload


def test_query_rejects_get_method(api_server: str) -> None:
    status, payload = request_json(f"{api_server}/query", method="GET")

    assert status == 405
    assert "detail" in payload


def test_server_fails_without_a_gemini_key(tmp_path: Path) -> None:
    result = run_failing_server("test.black_box.bootstrap_app:app", cwd=tmp_path)

    assert result.returncode != 0, result.stdout
    assert "GEMINI_API_KEY is required to start AstraRAG" in result.stderr


def test_server_fails_with_an_empty_corpus(tmp_path: Path) -> None:
    result = run_failing_server(
        "test.black_box.bootstrap_app:app",
        environment={
            "GEMINI_API_KEY": "startup-test-dummy-key",
            "QDRANT_PATH": str(tmp_path / "qdrant"),
        },
        cwd=tmp_path,
    )

    assert result.returncode != 0, result.stdout
    assert "Index documents before starting" in result.stderr
