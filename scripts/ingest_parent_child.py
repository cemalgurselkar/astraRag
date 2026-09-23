from pathlib import Path

from astrarag.embedding import EmbeddingEncoder
from astrarag.index import DenseVectorIndex
from astrarag.ingestion import PDFParser, ParentChildChunker


RAW_DIR = Path("data/documents")
QDRANT_PATH = Path("data/qdrant")

COLLECTION_NAME = "astrarag-parent-child"

PARENT_SIZE = 2000
CHILD_SIZE = 500
CHILD_OVERLAP = 100


def main() -> None:
    parser = PDFParser()

    chunker = ParentChildChunker(
        parent_size=PARENT_SIZE,
        child_size=CHILD_SIZE,
        child_overlap=CHILD_OVERLAP,
    )

    encoder = EmbeddingEncoder()

    index = DenseVectorIndex(
        path=QDRANT_PATH,
        collection_name=COLLECTION_NAME,
        dimension=encoder.dimension,
    )

    pdf_paths = sorted(RAW_DIR.glob("*.pdf"))

    if not pdf_paths:
        raise RuntimeError(f"No PDF files found in {RAW_DIR}")

    all_children = []
    parent_count = 0

    for pdf_path in pdf_paths:
        document = parser.parse(pdf_path)

        parents, children = chunker.chunk(document)

        parent_count += len(parents)
        all_children.extend(children)

        print(
            f"{pdf_path.name}: "
            f"{len(parents)} parents, "
            f"{len(children)} children"
        )

    print(f"\nParents: {parent_count}")
    print(f"Children: {len(all_children)}")

    texts = [
        child.text
        for child in all_children
    ]

    embeddings = encoder.encode(texts)

    index.add(
        chunks=all_children,
        embeddings=embeddings,
    )

    print(
        f"Indexed {len(all_children)} child chunks "
        f"into '{COLLECTION_NAME}'"
    )


if __name__ == "__main__":
    main()