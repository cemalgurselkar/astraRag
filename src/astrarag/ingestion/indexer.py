"""Orchestrate PDF parsing, chunking, embedding, and Qdrant indexing."""

from collections.abc import Iterable
from pathlib import Path

from qdrant_client import QdrantClient

from astrarag.embedding import EmbeddingEncoder
from astrarag.index import DenseVectorIndex
from astrarag.ingestion.chunker import FixedSizeChunker
from astrarag.ingestion.paren_child_chunker import ParentChildChunker
from astrarag.ingestion.parser import PDFParser


class CorpusIndexer:
    """Builds AstraRAG's persistent vector indexes from PDF documents."""

    def __init__(
        self,
        qdrant_path: str | Path,
        embedding_model_path: str | Path,
        collection_name: str = "astrarag-corpus",
        parent_child_collection_name: str = "astrarag-parent-child",
    ) -> None:
        self.qdrant_path = Path(qdrant_path)
        self.embedding_model_path = Path(embedding_model_path)
        self.collection_name = collection_name
        self.parent_child_collection_name = parent_child_collection_name

    def index_directory(self, directory: str | Path) -> int:
        directory = Path(directory).expanduser().resolve()

        if not directory.exists():
            raise FileNotFoundError(f"Document directory does not exist: {directory}")

        if not directory.is_dir():
            raise NotADirectoryError(f"Expected a directory: {directory}")

        pdf_files = sorted(path for path in directory.rglob("*.pdf") if path.is_file())

        if not pdf_files:
            raise ValueError(f"No PDF documents found in: {directory}")

        return self.index_files(pdf_files)

    def index_files(self, files: Iterable[str | Path]) -> int:
        pdf_files = [Path(path) for path in files]

        if not pdf_files:
            raise ValueError("No PDF files provided.")

        for path in pdf_files:
            if not path.exists():
                raise FileNotFoundError(f"PDF file does not exist: {path}")

            if path.suffix.casefold() != ".pdf":
                raise ValueError(f"Unsupported document type: {path}")

        return self._index_pdf_files(pdf_files)

    def _index_pdf_files(self, pdf_files: list[Path]) -> int:
        parser = PDFParser()

        encoder = EmbeddingEncoder(
            model_name=self.embedding_model_path,
        )

        client = QdrantClient(path=str(self.qdrant_path))

        dense_index = DenseVectorIndex(
            path=self.qdrant_path,
            collection_name=self.collection_name,
            dimension=encoder.dimension,
            client=client,
        )

        parent_child_index = DenseVectorIndex(
            path=self.qdrant_path,
            collection_name=self.parent_child_collection_name,
            dimension=encoder.dimension,
            client=client,
        )

        dense_chunks = []
        parent_child_chunks = []

        dense_chunker = FixedSizeChunker()

        parent_child_chunker = ParentChildChunker(
            parent_size=2000,
            child_size=500,
            child_overlap=100,
        )

        print(f"Found {len(pdf_files)} PDF document(s).")

        for pdf_path in pdf_files:
            print(f"Processing: {pdf_path.name}")

            document = parser.parse(pdf_path)

            chunks = dense_chunker.chunk(document)
            dense_chunks.extend(chunks)

            parents, children = parent_child_chunker.chunk(document)
            parent_child_chunks.extend(children)

            print(
                f"  dense chunks: {len(chunks)}, "
                f"parents: {len(parents)}, "
                f"children: {len(children)}"
            )

        if not dense_chunks:
            raise RuntimeError("No chunks were produced from the supplied documents.")

        print(f"\nEmbedding {len(dense_chunks)} dense chunks...")

        dense_embeddings = encoder.encode([chunk.text for chunk in dense_chunks])

        dense_index.add(
            chunks=dense_chunks,
            embeddings=dense_embeddings,
        )

        print(f"Indexed {len(dense_chunks)} chunks into '{self.collection_name}'.")

        if parent_child_chunks:
            print(f"Embedding {len(parent_child_chunks)} parent-child chunks...")

            parent_child_embeddings = encoder.encode(
                [chunk.text for chunk in parent_child_chunks]
            )

            parent_child_index.add(
                chunks=parent_child_chunks,
                embeddings=parent_child_embeddings,
            )

            print(
                f"Indexed {len(parent_child_chunks)} child chunks "
                f"into '{self.parent_child_collection_name}'."
            )

        print("\nAstraRAG corpus indexing complete.")
        client.close()

        return len(pdf_files)
