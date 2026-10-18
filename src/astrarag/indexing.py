"""PDF corpusunu indeksler ve yeni Qdrant koleksiyonunu atomik olarak yayımlar.

Sözlük:
- `index_directory`: Dizindeki PDF'leri doğrular ve tam indekslemeyi başlatır.
- `_index_files`: Belgeleri parçalar, embedding üretir ve geçici koleksiyona yazar.
- `_publish_collection`: Hazır koleksiyonu alias değişimiyle canlıya alır.
"""

from pathlib import Path
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ApiException
from qdrant_client.models import (
    CreateAlias,
    CreateAliasOperation,
    DeleteAlias,
    DeleteAliasOperation,
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    VectorParams,
)
from tokenizers import Tokenizer

from astrarag.config import Settings
from astrarag.documents import chunk_document_by_tokens, parse_pdf
from astrarag.embedding import EmbeddingEncoder
from astrarag.indexes import DenseVectorIndex, create_qdrant_client


def index_directory(
    directory: str | Path,
    *,
    settings: Settings,
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

    _index_files(files, settings)
    return len(files)


def _index_files(files: list[Path], settings: Settings) -> None:
    tokenizer_path = settings.embedding_tokenizer_path.expanduser().resolve()
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Embedding tokenizer not found: {tokenizer_path}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    encoder = EmbeddingEncoder(
        base_url=settings.embedding_url,
        model=settings.embedding_model,
        dimension=settings.embedding_dimension,
        timeout_seconds=settings.embedding_timeout_seconds,
    )
    client = create_qdrant_client(
        url=settings.qdrant_url,
        path=settings.qdrant_path,
        api_key=settings.qdrant_api_key,
        timeout=settings.qdrant_timeout_seconds,
    )
    physical_collection = f"{settings.qdrant_collection}__{uuid4().hex[:12]}"
    published = False
    try:
        client.create_collection(
            collection_name=physical_collection,
            vectors_config=VectorParams(
                size=settings.embedding_dimension,
                distance=Distance.COSINE,
            ),
            hnsw_config=HnswConfigDiff(
                m=settings.qdrant_hnsw_m,
                ef_construct=settings.qdrant_hnsw_ef_construct,
                full_scan_threshold=settings.qdrant_full_scan_threshold_kb,
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=settings.qdrant_indexing_threshold_kb,
            ),
            metadata=settings.index_metadata,
        )
        index = DenseVectorIndex(
            path=settings.qdrant_path,
            collection_name=physical_collection,
            dimension=encoder.dimension,
            hnsw_ef_search=settings.qdrant_hnsw_ef_search,
            expected_metadata=settings.index_metadata,
            validate_hnsw=settings.qdrant_url is not None,
            client=client,
        )

        chunk_count = 0
        for pdf_path in files:
            chunks = chunk_document_by_tokens(
                parse_pdf(pdf_path),
                tokenizer,
                chunk_size=settings.chunk_size_tokens,
                overlap=settings.chunk_overlap_tokens,
            )
            for start in range(0, len(chunks), settings.embedding_batch_size):
                batch = chunks[start : start + settings.embedding_batch_size]
                embeddings = encoder.encode([chunk.text for chunk in batch])
                index.add(batch, embeddings)
            chunk_count += len(chunks)
            print(f"{pdf_path.name}: {len(chunks)} chunks")

        if chunk_count == 0:
            raise RuntimeError("No chunks were produced from the supplied documents")

        previous_collection = _publish_collection(
            client,
            alias=settings.qdrant_collection,
            collection=physical_collection,
        )
        published = True
        if previous_collection:
            try:
                client.delete_collection(previous_collection)
            except (ApiException, OSError, RuntimeError) as exc:
                print(
                    f"Warning: could not remove previous collection "
                    f"'{previous_collection}': {exc}"
                )
        print(
            f"Indexed {chunk_count} chunks into "
            f"'{settings.qdrant_collection}' with HNSW enabled."
        )
    except Exception:
        if not published and client.collection_exists(physical_collection):
            client.delete_collection(physical_collection)
        raise
    finally:
        encoder.close()
        client.close()


def _publish_collection(
    client: QdrantClient,
    *,
    alias: str,
    collection: str,
) -> str | None:
    aliases = {
        item.alias_name: item.collection_name for item in client.get_aliases().aliases
    }
    previous = aliases.get(alias)

    operations: list[DeleteAliasOperation | CreateAliasOperation] = []
    if previous:
        operations.append(
            DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=alias))
        )
    else:
        collection_names = {item.name for item in client.get_collections().collections}
        if alias in collection_names:
            client.delete_collection(alias)

    operations.append(
        CreateAliasOperation(
            create_alias=CreateAlias(
                collection_name=collection,
                alias_name=alias,
            )
        )
    )
    client.update_collection_aliases(operations)
    return previous if previous != collection else None
