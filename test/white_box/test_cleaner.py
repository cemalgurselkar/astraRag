"""Verify normalization and artifact repair performed by the text cleaner."""

from astrarag.documents import clean_text


def test_cleaner_removes_pdf_artifacts() -> None:
    text = "masked lan\ufffeguage model"
    assert clean_text(text) == "masked language model"


def test_cleaner_joins_hyphenated_line_breaks() -> None:
    text = "bidirec-\ntional representations"
    assert clean_text(text) == "bidirectional representations"


def test_cleaner_normalizes_line_breaks() -> None:
    text = "Dense Passage\nRetrieval   for\nQuestion Answering"
    assert clean_text(text) == "Dense Passage Retrieval for Question Answering"


def test_cleaner_handles_empty_text() -> None:
    assert clean_text("") == ""
