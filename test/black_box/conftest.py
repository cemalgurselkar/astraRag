"""Shared black-box fixtures for process-boundary tests."""

from collections.abc import Iterator

import pytest

from test.black_box.server import local_sockets_available, running_server


@pytest.fixture(scope="module")
def api_server() -> Iterator[str]:
    if not local_sockets_available():
        pytest.skip("local socket binding is unavailable in this sandbox")

    with running_server("test.black_box.fake_app:app") as base_url:
        yield base_url
