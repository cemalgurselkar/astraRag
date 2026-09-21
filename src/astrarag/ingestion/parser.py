from hashlib import sha256
from pathlib import Path

import pymupdf

from astrarag.schemas import Document, Page


class PDFParser:
    """Parse text-based PDF files into AstraRAG documents."""

    def parse(self, file_path: Path) -> Document:
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {file_path.suffix}")

        docuemnt_id = self._compute_document_id(file_path)
        pages: list[Page] = []
        
        with pymupdf.open(file_path) as pdf:
            page_count = len(pdf)
            
            for page_index, page in enumerate(pdf):
                text = page.get_text("text").strip()

                if text:
                    pages.append(
                        Page(
                            number=page_index + 1,
                            text=text,
                        )
                    )

        if not pages:
            raise ValueError(f"No extractable text found in PDF: {file_path}")
        
        return Document(
            id=docuemnt_id,
            source=str(file_path),
            pages=pages,
            metadata={
                "filename": file_path.name,
                "page_count": page_count,
                "text_page_count": len(pages)
            },
        )
    
    @staticmethod
    def _compute_document_id(file_path: Path) -> str:
        hasher = sha256()
        
        with file_path.open("rb") as file:
            for block in iter(lambda: file.read(1024*1024), b""):
                hasher.update(block)
        
        return hasher.hexdigest()