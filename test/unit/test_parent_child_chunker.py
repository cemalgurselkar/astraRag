from astrarag.ingestion import ParentChildChunker
from astrarag.schemas import Document, Page


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

    chunker = ParentChildChunker(
        parent_size=1000,
        child_size=300,
        child_overlap=50,
    )

    parents, children = chunker.chunk(document)

    assert len(parents) == 2
    assert len(children) > len(parents)

    parent_ids = {parent.id for parent in parents}

    for child in children:
        assert child.metadata["chunk_type"] == "child"
        assert child.metadata["parent_id"] in parent_ids

    for parent in parents:
        assert parent.metadata["chunk_type"] == "parent"
