"""Print fixed-size chunks from a sample PDF for quick manual inspection."""

from pathlib import Path

from astrarag.ingestion import FixedSizeChunker, PDFParser

pdf_path = Path("data/raw/attention.pdf")

parser = PDFParser()
document = parser.parse(pdf_path)

chunker = FixedSizeChunker(
    chunk_size=1000,
    overlap=200,
)
chunks = chunker.chunk(document)

for chunk in chunks:
    print("=" * 100)
    print(f"Chunk ID: {chunk.id}")
    print(f"Pages: {chunk.page_numbers}")
    print("=" * 100)
    print(chunk.text)
    print()
