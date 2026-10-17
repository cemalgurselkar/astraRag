"""Deterministic API runtime used by black-box server tests."""

from types import SimpleNamespace

from astrarag.api.app import create_app


class FakeRAG:
    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def query(self, query: str) -> SimpleNamespace:
        if query == "invalid":
            raise ValueError("invalid query")
        if query == "unavailable":
            raise RuntimeError("service unavailable")

        source = SimpleNamespace(
            chunk_id="chunk-1",
            document_id="document-1",
            page_numbers=[2],
        )
        return SimpleNamespace(
            generation=SimpleNamespace(answer="Grounded answer [SOURCE 1]."),
            route=SimpleNamespace(value="cheap"),
            context=SimpleNamespace(items=[source]),
            retrieval_latency_ms=1.25,
            total_latency_ms=3.5,
        )


app = create_app(FakeRAG)
