"""Verify that hierarchical chunking preserves parent-child relationships."""

from astrarag.experiments import chunk_parent_child
from astrarag.models import Document, Page


def test_parent_child_relationship() -> None:
    document = Document(
        id="doc-1",
        source="test.pdf",
        pages=[
            Page(
                number=1,
                text="A" * 2000,
            )
        ],
        metadata={
            "filename": "test.pdf",
        },
    )

    parents, children = chunk_parent_child(
        document,
        parent_size=1000,
        child_size=300,
        child_overlap=50,
    )

    assert len(parents) == 2
    assert len(children) > len(parents)

    parent_ids = {parent.id for parent in parents}

    for child in children:
        assert child.metadata["chunk_type"] == "child"
        assert child.metadata["parent_id"] in parent_ids

    for parent in parents:
        assert parent.metadata["chunk_type"] == "parent"
