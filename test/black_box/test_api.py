"""Exercise the API through a real HTTP server process."""

import json
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest

from test.black_box.fake_app import app

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _local_sockets_available() -> bool:
    try:
        with socket.socket():
            return True
    except OSError:
        return False


@contextmanager
def running_server() -> Iterator[str]:
    port = _free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "test.black_box.fake_app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"

    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                raise RuntimeError(f"API failed to start:\n{stdout}\n{stderr}")
            try:
                with urlopen(f"{base_url}/health", timeout=0.5):
                    break
            except URLError:
                time.sleep(0.1)
        else:
            raise TimeoutError("API did not become ready")

        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def request_json(
    url: str,
    *,
    payload: dict[str, str] | None = None,
) -> tuple[int, dict[str, object]]:
    data = json.dumps(payload).encode() if payload is not None else None
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_openapi_exposes_the_public_routes() -> None:
    schema = app.openapi()

    assert set(schema["paths"]) == {"/health", "/query"}
    assert schema["paths"]["/health"]["get"]["responses"]["200"]
    assert schema["paths"]["/query"]["post"]["responses"]["200"]


@pytest.mark.skipif(
    not _local_sockets_available(),
    reason="local socket binding is unavailable in this sandbox",
)
def test_http_contract() -> None:
    with running_server() as base_url:
        health = request_json(f"{base_url}/health")
        success = request_json(
            f"{base_url}/query",
            payload={"query": "What is attention?"},
        )
        empty = request_json(f"{base_url}/query", payload={"query": ""})
        invalid = request_json(f"{base_url}/query", payload={"query": "invalid"})
        unavailable = request_json(
            f"{base_url}/query", payload={"query": "unavailable"}
        )

    assert health == (200, {"status": "ok"})
    assert success == (
        200,
        {
            "answer": "Grounded answer [SOURCE 1].",
            "route": "cheap",
            "sources": [
                {
                    "chunk_id": "chunk-1",
                    "document_id": "document-1",
                    "pages": [2],
                }
            ],
            "retrieval_latency_ms": 1.25,
            "total_latency_ms": 3.5,
        },
    )
    assert empty[0] == 422
    assert invalid == (400, {"detail": "invalid query"})
    assert unavailable == (503, {"detail": "service unavailable"})
