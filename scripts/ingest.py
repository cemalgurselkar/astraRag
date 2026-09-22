from pathlib import Path
from time import perf_counter

from astrarag.embedding import EmbeddingEncoder
from astrarag.index import DenseVectorIndex
from astrarag.ingestion import FixedSizeChunker, PDFParser


RAW_DATA_DIR = Path("data/raw")
QDRANT_PATH = Path("data/qdrant")
COLLECTION_NAME = "astrarag-corpus"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def main() -> None:
    pdf_paths = sorted(RAW_DATA_DIR.glob("*.pdf"))

    if not pdf_paths:
        raise RuntimeError(f"No PDF files found in {RAW_DATA_DIR}")

    parser = PDFParser()
    chunker = FixedSizeChunker(
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )
    encoder = EmbeddingEncoder()

    index = DenseVectorIndex(
        path=QDRANT_PATH,
        collection_name=COLLECTION_NAME,
        dimension=encoder.dimension,
    )

    total_chunks = 0
    start = perf_counter()

    print(f"Found {len(pdf_paths)} PDF files.\n")

    for pdf_path in pdf_paths:
        document = parser.parse(pdf_path)
        chunks = chunker.chunk(document)

        embeddings = encoder.encode(
            [chunk.text for chunk in chunks]
        )

        index.add(
            chunks=chunks,
            embeddings=embeddings,
        )

        total_chunks += len(chunks)

        print(
            f"{pdf_path.name:<30} "
            f"pages={len(document.pages):<4} "
            f"chunks={len(chunks):<4} "
            f"document={document.id[:12]}"
        )

    elapsed = perf_counter() - start

    print("\nIngestion complete")
    print("-" * 50)
    print(f"Documents: {len(pdf_paths)}")
    print(f"Chunks:    {total_chunks}")
    print(f"Time:      {elapsed:.2f}s")
    print(f"Collection: {COLLECTION_NAME}")


if __name__ == "__main__":
    main()