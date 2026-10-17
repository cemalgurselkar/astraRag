"""Expose the high-level AstraRAG application interface."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from astrarag.app import AstraRAG

__all__ = ["AstraRAG"]


def __getattr__(name: str) -> Any:
    if name == "AstraRAG":
        from astrarag.app import AstraRAG

        return AstraRAG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
