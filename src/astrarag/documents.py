"""Kaynak PDF'leri okur, metni temizler ve aranabilir parçalara böler.

Sözlük:
- `clean_text`: PDF çıkarımından kalan boşluk ve karakter bozukluklarını düzeltir.
- `parse_pdf`: PDF sayfalarını doğrulanmış bir `Document` modeline dönüştürür.
- `chunk_document`: Deneyler için karakter tabanlı parçalar üretir.
- `chunk_document_by_tokens`: Production için tokenizer tabanlı parçalar üretir.
- `_document_id`: Dosya içeriğinden kararlı belge kimliği oluşturur.
"""

import re
import unicodedata
from hashlib import sha256
from pathlib import Path

import pymupdf
from tokenizers import Tokenizer

from astrarag.models import Chunk, Document, Page


def clean_text(text: str) -> str:
    """Normalize common artifacts produced by PDF text extraction."""
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "").replace("\ufffd", "").replace("\ufffe", "")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def parse_pdf(file_path: str | Path) -> Document:
    """Parse a text-based PDF into a validated document."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")

    pages: list[Page] = []
    with pymupdf.open(path) as pdf:
        page_count = len(pdf)
        for page_index, page in enumerate(pdf):
            text = clean_text(page.get_text("text"))
            if text:
                pages.append(Page(number=page_index + 1, text=text))

    if not pages:
        raise ValueError(f"No extractable text found in PDF: {path}")

    return Document(
        id=_document_id(path),
        source=str(path),
        pages=pages,
        metadata={
            "filename": path.name,
            "page_count": page_count,
            "text_page_count": len(pages),
        },
    )


def _document_id(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chunk_document(
    document: Document,
    chunk_size: int = 1800,
    overlap: int = 200,
) -> list[Chunk]:
    """Split every page into overlapping, fixed-size chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[Chunk] = []
    chunk_index = 0

    for page in document.pages:
        start = 0
        while start < len(page.text):
            end = min(start + chunk_size, len(page.text))
            text = page.text[start:end].strip()
            if text:
                chunks.append(
                    Chunk(
                        id=f"{document.id}:{chunk_index}",
                        document_id=document.id,
                        text=text,
                        index=chunk_index,
                        page_numbers=[page.number],
                        metadata={
                            **document.metadata,
                            "start_char": start,
                            "end_char": end,
                        },
                    )
                )
                chunk_index += 1
            if end == len(page.text):
                break
            start = end - overlap

    return chunks


def chunk_document_by_tokens(
    document: Document,
    tokenizer: Tokenizer,
    chunk_size: int = 254,
    overlap: int = 32,
) -> list[Chunk]:
    """Split pages on embedding-token boundaries without losing source offsets."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")

    chunks: list[Chunk] = []
    chunk_index = 0
    step = chunk_size - overlap

    for page in document.pages:
        offsets = tokenizer.encode(
            page.text,
            add_special_tokens=False,
        ).offsets
        for token_start in range(0, len(offsets), step):
            token_end = min(token_start + chunk_size, len(offsets))
            start_char = offsets[token_start][0]
            end_char = offsets[token_end - 1][1]
            text = page.text[start_char:end_char].strip()
            if text:
                chunks.append(
                    Chunk(
                        id=f"{document.id}:{chunk_index}",
                        document_id=document.id,
                        text=text,
                        index=chunk_index,
                        page_numbers=[page.number],
                        metadata={
                            **document.metadata,
                            "start_char": start_char,
                            "end_char": end_char,
                            "start_token": token_start,
                            "end_token": token_end,
                        },
                    )
                )
                chunk_index += 1
            if token_end == len(offsets):
                break

    return chunks
