"""Orchestrate corpus parsing, embedding, and indexing."""

from collections.abc import Iterable
from pathlib import Path

from astrarag.documents import chunk_document, parse_pdf


def index_directory(
    directory: str | Path,
    *,
    qdrant_path: str | Path,
    embedding_model_path: str | Path,
    collection_name: str = "astrarag-corpus",
) -> int:
    """Index every PDF under a directory into the production collection."""
    path = Path(directory).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Document directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Expected a directory: {path}")

    files = sorted(item for item in path.rglob("*.pdf") if item.is_file())
    if not files:
        raise ValueError(f"No PDF documents found in: {path}")

    _index_files(
        files,
        qdrant_path=Path(qdrant_path),
        embedding_model_path=Path(embedding_model_path),
        collection_name=collection_name,
    )
    return len(files)


def _index_files(
    files: Iterable[Path],
    *,
    qdrant_path: Path,
    embedding_model_path: Path,
    collection_name: str,
) -> None:
    chunks = []
    for pdf_path in files:
        document_chunks = chunk_document(parse_pdf(pdf_path))
        chunks.extend(document_chunks)
        print(f"{pdf_path.name}: {len(document_chunks)} chunks")

    if not chunks:
        raise RuntimeError("No chunks were produced from the supplied documents")

    from qdrant_client import QdrantClient

    from astrarag.embedding import EmbeddingEncoder
    from astrarag.indexes import DenseVectorIndex

    encoder = EmbeddingEncoder(embedding_model_path)
    client = QdrantClient(path=str(qdrant_path))

    try:
        index = DenseVectorIndex(
            path=qdrant_path,
            collection_name=collection_name,
            dimension=encoder.dimension,
            client=client,
        )
        embeddings = encoder.encode([chunk.text for chunk in chunks])
        index.add(chunks, embeddings)
        print(f"Indexed {len(chunks)} chunks into '{collection_name}'.")
    finally:
        client.close()
