from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from astrarag.documents import chunk_document, chunk_document_by_tokens
from astrarag.models import Document, Page


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

    chunks = chunk_document(document, chunk_size=1000, overlap=200)

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

    chunks = chunk_document(document, chunk_size=1000, overlap=200)

    assert chunks[0].page_numbers == [1]
    assert chunks[1].page_numbers == [1]
    assert chunks[2].page_numbers == [2]
    assert chunks[3].page_numbers == [2]


def test_token_chunker_respects_embedding_limit_and_offsets() -> None:
    tokenizer = Tokenizer(
        WordLevel(
            vocab={"[UNK]": 0, "one": 1, "two": 2, "three": 3, "four": 4},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    document = Document(
        id="doc-1",
        source="test.pdf",
        pages=[Page(number=1, text="one two three four")],
    )

    chunks = chunk_document_by_tokens(
        document,
        tokenizer,
        chunk_size=3,
        overlap=1,
    )

    assert [chunk.text for chunk in chunks] == ["one two three", "three four"]
    assert chunks[0].metadata["start_token"] == 0
    assert chunks[0].metadata["end_token"] == 3
    assert chunks[1].metadata["start_char"] == 8
    assert chunks[1].metadata["end_char"] == 18
