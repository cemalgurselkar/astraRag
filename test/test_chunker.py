from astrarag.ingestion import FixedSizeChunker
from astrarag.schemas import Document, Page


def test_chunker_splits_document_with_overlap() -> None:
    document = Document(
        id="doc-1",
        source="test.pdf",
        pages=[
            Page(
                number=1,
                text="A" * 2500,
            )
        ],
    )

    chunker = FixedSizeChunker(
        chunk_size=1000,
        overlap=200,
    )

    chunks = chunker.chunk(document)

    assert len(chunks) == 3

    assert chunks[0].metadata["start_char"] == 0
    assert chunks[0].metadata["end_char"] == 1000

    assert chunks[1].metadata["start_char"] == 800
    assert chunks[1].metadata["end_char"] == 1800

    assert chunks[2].metadata["start_char"] == 1600
    assert chunks[2].metadata["end_char"] == 2500


def test_chunker_preserves_page_numbers() -> None:
    document = Document(
        id="doc-1",
        source="test.pdf",
        pages=[
            Page(number=1, text="A" * 1200),
            Page(number=2, text="B" * 1200),
        ],
    )

    chunker = FixedSizeChunker(
        chunk_size=1000,
        overlap=200,
    )

    chunks = chunker.chunk(document)

    assert chunks[0].page_numbers == [1]
    assert chunks[1].page_numbers == [1]
    assert chunks[2].page_numbers == [2]
    assert chunks[3].page_numbers == [2]