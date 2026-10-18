"""Process-boundary helpers shared by black-box tests."""

import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDING_MODEL = PROJECT_ROOT / "models/all-minilm-l6-v2_f16.gguf"
EMBEDDING_TOKENIZER = PROJECT_ROOT / "models/all-MiniLM-L6-v2/tokenizer.json"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def local_sockets_available() -> bool:
    try:
        with socket.socket():
            return True
    except OSError:
        return False


def _prepare_environment(
    environment: dict[str, str] | None,
    cwd: Path,
) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("GEMINI_API_KEY", None)
    if environment:
        env.update(environment)

    parts = [str(PROJECT_ROOT)]
    existing = env.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)

    if cwd == PROJECT_ROOT:
        env.pop("PYTHONPATH", None)
    return env


@contextmanager
def running_server(
    module_name: str,
    *,
    environment: dict[str, str] | None = None,
    cwd: Path = PROJECT_ROOT,
) -> Iterator[str]:
    port = free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            module_name,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=cwd,
        env=_prepare_environment(environment, cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"

    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                raise RuntimeError(
                    f"Server exited before becoming ready:\n{stdout}\n{stderr}"
                )
            try:
                with urlopen(f"{base_url}/health", timeout=0.5):
                    break
            except URLError:
                time.sleep(0.1)
        else:
            raise TimeoutError("Server did not become ready")

        yield base_url
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def run_failing_server(
    module_name: str,
    *,
    environment: dict[str, str] | None = None,
    cwd: Path = PROJECT_ROOT,
) -> subprocess.CompletedProcess[str]:
    """Run a server that is expected to exit on its own during startup."""
    port = free_port()
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            module_name,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=cwd,
        env=_prepare_environment(environment, cwd),
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def request_json(
    url: str,
    *,
    payload: dict[str, object] | None = None,
    raw: bytes | None = None,
    method: str | None = None,
) -> tuple[int, Any]:
    data = raw
    if data is None and payload is not None:
        data = json.dumps(payload).encode()
    request = Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except HTTPError as exc:
        body = exc.read()
        try:
            return exc.code, json.loads(body)
        except ValueError:
            return exc.code, body.decode(errors="replace")
