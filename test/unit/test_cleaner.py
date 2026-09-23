from astrarag.ingestion import TextCleaner


def test_cleaner_removes_pdf_artifacts() -> None:
    cleaner = TextCleaner()

    text = "masked lan\ufffeguage model"

    assert cleaner.clean(text) == "masked language model"


def test_cleaner_joins_hyphenated_line_breaks() -> None:
    cleaner = TextCleaner()

    text = "bidirec-\ntional representations"

    assert cleaner.clean(text) == "bidirectional representations"


def test_cleaner_normalizes_line_breaks() -> None:
    cleaner = TextCleaner()

    text = "Dense Passage\nRetrieval   for\nQuestion Answering"

    assert cleaner.clean(text) == "Dense Passage Retrieval for Question Answering"


def test_cleaner_handles_empty_text() -> None:
    cleaner = TextCleaner()

    assert cleaner.clean("") == ""
