from pathlib import Path

from astrarag.ingestion import PDFParser


def test_parser_extracts_text_from_pdf() -> None:
    pdf_path = Path("data/raw/attention.pdf")

    parser = PDFParser()
    document = parser.parse(pdf_path)

    assert document.pages
    assert document.metadata["page_count"] == 15
    assert document.metadata["text_page_count"] == 15
    assert document.pages[0].number == 1
    assert "Attention Is All You Need" in document.pages[0].text