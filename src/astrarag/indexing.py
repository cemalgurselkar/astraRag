"""PDF corpusunu artımlı indeksler ve Qdrant collection'ını atomik yayımlar.

Sözlük:
- `SourceFile`: Bir PDF'in değişiklik ve yeniden kullanım bilgilerini taşır.
- `index_directory`: Dizindeki PDF'leri tarar ve indekslemeyi başlatır.
- `_load_source_manifest`: Mevcut chunk payload'larından belge durumunu çıkarır.
- `_classify_sources`: Yeni, değişmiş ve tekrar kullanılabilir belgeleri ayırır.
- `_copy_document_points`: Değişmeyen belge vektörlerini yeni collection'a taşır.
- `_publish_collection`: Hazır collection'ı alias değişimiyle canlıya alır.
"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ApiException
from qdrant_client.models import (
    CreateAlias,
    CreateAliasOperation,
    DeleteAlias,
    DeleteAliasOperation,
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchValue,
    OptimizersConfigDiff,
    PointStruct,
    VectorParams,
)
from tokenizers import Tokenizer

from astrarag.config import Settings
from astrarag.documents import chunk_document_by_tokens, file_sha256, parse_pdf
from astrarag.embedding import EmbeddingEncoder
from astrarag.indexes import DenseVectorIndex, create_qdrant_client


@dataclass(frozen=True)
class SourceFile:
    path: Path
    key: str
    content_sha256: str
    size: int
    mtime_ns: int
    reusable_document_id: str | None
    metadata_changed: bool


def index_directory(
    directory: str | Path,
    *,
    settings: Settings,
) -> int:
    """Incrementally index every PDF under a directory."""
    root = Path(directory).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Document directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Expected a directory: {root}")

    files = sorted(item for item in root.rglob("*.pdf") if item.is_file())
    if not files:
        raise ValueError(f"No PDF documents found in: {root}")

    _index_files(files, root, settings)
    return len(files)


def _index_files(files: list[Path], root: Path, settings: Settings) -> None:
    client = create_qdrant_client(
        url=settings.qdrant_url,
        path=settings.qdrant_path,
        api_key=settings.qdrant_api_key,
        timeout=settings.qdrant_timeout_seconds,
    )
    encoder: EmbeddingEncoder | None = None
    physical_collection = f"{settings.qdrant_collection}__{uuid4().hex[:12]}"
    published = False

    try:
        active_collection = _active_collection(client, settings.qdrant_collection)
        compatible = bool(
            active_collection
            and _collection_is_compatible(client, active_collection, settings)
        )
        old_sources = (
            _load_source_manifest(client, active_collection)
            if active_collection and compatible
            else {}
        )
        sources = _classify_sources(files, root, old_sources)
        current_keys = {source.key for source in sources}

        if (
            compatible
            and set(old_sources) == current_keys
            and all(
                source.reusable_document_id and not source.metadata_changed
                for source in sources
            )
        ):
            print(
                f"Index is already up to date: {len(sources)} documents, "
                f"{_count_collection(client, active_collection)} chunks."
            )
            return

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
            dimension=settings.embedding_dimension,
            hnsw_ef_search=settings.qdrant_hnsw_ef_search,
            expected_metadata=settings.index_metadata,
            validate_hnsw=settings.qdrant_url is not None,
            client=client,
        )

        changed_sources = [
            source for source in sources if source.reusable_document_id is None
        ]
        tokenizer: Tokenizer | None = None
        if changed_sources:
            tokenizer_path = settings.embedding_tokenizer_path.expanduser().resolve()
            if not tokenizer_path.is_file():
                raise FileNotFoundError(
                    f"Embedding tokenizer not found: {tokenizer_path}"
                )
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            tokenizer.no_padding()
            tokenizer.no_truncation()

        reused_chunks = 0
        embedded_chunks = 0
        for source in sources:
            metadata: dict[str, str | int] = {
                "source_key": source.key,
                "content_sha256": source.content_sha256,
                "file_size": source.size,
                "mtime_ns": source.mtime_ns,
            }
            if source.reusable_document_id and active_collection:
                count = _copy_document_points(
                    client,
                    source_collection=active_collection,
                    target_collection=physical_collection,
                    document_id=source.reusable_document_id,
                    metadata_updates=metadata,
                )
                if count == 0:
                    raise RuntimeError(
                        f"No reusable chunks found for document: {source.key}"
                    )
                reused_chunks += count
                print(f"{source.key}: reused {count} chunks")
                continue

            if tokenizer is None:
                raise RuntimeError("Embedding tokenizer was not initialized")
            document = parse_pdf(
                source.path,
                document_id=_source_document_id(source),
                extra_metadata=metadata,
            )
            chunks = chunk_document_by_tokens(
                document,
                tokenizer,
                chunk_size=settings.chunk_size_tokens,
                overlap=settings.chunk_overlap_tokens,
            )
            if encoder is None:
                encoder = EmbeddingEncoder(
                    base_url=settings.embedding_url,
                    model=settings.embedding_model,
                    dimension=settings.embedding_dimension,
                    timeout_seconds=settings.embedding_timeout_seconds,
                )
                encoder.check_health()
            for start in range(0, len(chunks), settings.embedding_batch_size):
                batch = chunks[start : start + settings.embedding_batch_size]
                embeddings = encoder.encode([chunk.text for chunk in batch])
                index.add(batch, embeddings)
            embedded_chunks += len(chunks)
            print(f"{source.key}: embedded {len(chunks)} chunks")

        chunk_count = reused_chunks + embedded_chunks
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
            f"Indexed {chunk_count} chunks into '{settings.qdrant_collection}': "
            f"{reused_chunks} reused, {embedded_chunks} embedded."
        )
    except Exception:
        if not published and client.collection_exists(physical_collection):
            client.delete_collection(physical_collection)
        raise
    finally:
        if encoder is not None:
            encoder.close()
        client.close()


def _active_collection(client: QdrantClient, alias: str) -> str | None:
    aliases = {
        item.alias_name: item.collection_name for item in client.get_aliases().aliases
    }
    if alias in aliases:
        return aliases[alias]
    if client.collection_exists(alias):
        return alias
    return None


def _collection_is_compatible(
    client: QdrantClient,
    collection: str,
    settings: Settings,
) -> bool:
    info = client.get_collection(collection)
    vectors = info.config.params.vectors
    if getattr(vectors, "size", None) != settings.embedding_dimension:
        return False
    metadata = info.config.metadata or {}
    return all(
        metadata.get(key) == value for key, value in settings.index_metadata.items()
    )


def _load_source_manifest(
    client: QdrantClient,
    collection: str,
) -> dict[str, dict[str, str | int]]:
    manifest: dict[str, dict[str, str | int]] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            metadata = payload.get("metadata")
            document_id = payload.get("document_id")
            if not isinstance(metadata, dict) or not isinstance(document_id, str):
                continue
            source_key = metadata.get("source_key")
            content_sha256 = metadata.get("content_sha256")
            file_size = metadata.get("file_size")
            mtime_ns = metadata.get("mtime_ns")
            if (
                isinstance(source_key, str)
                and isinstance(content_sha256, str)
                and isinstance(file_size, int)
                and isinstance(mtime_ns, int)
            ):
                manifest[source_key] = {
                    "document_id": document_id,
                    "content_sha256": content_sha256,
                    "file_size": file_size,
                    "mtime_ns": mtime_ns,
                }
        if offset is None:
            return manifest


def _classify_sources(
    files: list[Path],
    root: Path,
    old_sources: dict[str, dict[str, str | int]],
) -> list[SourceFile]:
    sources: list[SourceFile] = []
    for path in files:
        key = path.relative_to(root).as_posix()
        stat = path.stat()
        old = old_sources.get(key)
        metadata_changed = not old or (
            old.get("file_size") != stat.st_size
            or old.get("mtime_ns") != stat.st_mtime_ns
        )

        if old and not metadata_changed:
            content_sha256 = str(old["content_sha256"])
        else:
            content_sha256 = file_sha256(path)

        reusable_document_id = None
        if old and old.get("content_sha256") == content_sha256:
            reusable_document_id = str(old["document_id"])

        sources.append(
            SourceFile(
                path=path,
                key=key,
                content_sha256=content_sha256,
                size=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                reusable_document_id=reusable_document_id,
                metadata_changed=metadata_changed,
            )
        )
    return sources


def _source_document_id(source: SourceFile) -> str:
    """Keep equal-content files distinct while retaining stable chunk IDs."""
    identity = f"{source.key}\0{source.content_sha256}".encode()
    return sha256(identity).hexdigest()


def _copy_document_points(
    client: QdrantClient,
    *,
    source_collection: str,
    target_collection: str,
    document_id: str,
    metadata_updates: dict[str, str | int],
) -> int:
    copied = 0
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=source_collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        batch: list[PointStruct] = []
        for point in points:
            if point.vector is None:
                raise RuntimeError(f"Stored vector is missing for point {point.id}")
            payload = dict(point.payload or {})
            metadata = dict(payload.get("metadata", {}))
            metadata.update(metadata_updates)
            payload["metadata"] = metadata
            batch.append(
                PointStruct(
                    id=point.id,
                    vector=cast(Any, point.vector),
                    payload=payload,
                )
            )
        if batch:
            client.upsert(
                collection_name=target_collection,
                points=batch,
                wait=True,
            )
            copied += len(batch)
        if offset is None:
            return copied


def _count_collection(client: QdrantClient, collection: str | None) -> int:
    if collection is None:
        return 0
    return client.count(collection_name=collection, exact=True).count


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
