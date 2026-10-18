"""Verify PDF-to-Qdrant indexing through the installed CLI."""

import os
import subprocess
import sys
from pathlib import Path

import pymupdf
import pytest
from qdrant_client import QdrantClient

from test.black_box.server import EMBEDDING_MODEL, EMBEDDING_TOKENIZER

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_index(
    documents: Path,
    tmp_path: Path,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("GEMINI_API_KEY", None)
    environment["EMBEDDING_TOKENIZER_PATH"] = str(EMBEDDING_TOKENIZER)
    environment["QDRANT_PATH"] = str(tmp_path / "qdrant")

    return subprocess.run(
        [sys.executable, "-m", "astrarag.cli", "index", str(documents)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def write_pdf(path: Path, text: str) -> None:
    pdf = pymupdf.open()
    page = pdf.new_page()
    if text:
        page.insert_text((72, 72), text)
    pdf.save(path)
    pdf.close()


@pytest.mark.slow
@pytest.mark.skipif(
    os.getenv("RUN_SLOW_TESTS") != "1"
    or not EMBEDDING_MODEL.exists()
    or not EMBEDDING_TOKENIZER.exists(),
    reason="set RUN_SLOW_TESTS=1 and install the local embedding model",
)
def test_cli_indexes_a_small_pdf_without_a_gemini_key(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    write_pdf(documents / "sample.pdf", "AstraRAG black-box indexing test document.")

    result = run_index(documents, tmp_path)

    assert result.returncode == 0, result.stderr
    assert "Successfully indexed 1 document(s)" in result.stdout
    assert (tmp_path / "qdrant").exists()

    client = QdrantClient(path=str(tmp_path / "qdrant"))
    try:
        collection = client.get_collection("astrarag-corpus")
        point_count = client.count(
            collection_name="astrarag-corpus",
            exact=True,
        ).count
        points, _ = client.scroll(
            collection_name="astrarag-corpus",
            limit=10,
            with_payload=True,
        )
        aliases = client.get_aliases().aliases
    finally:
        client.close()

    assert point_count >= 1
    assert collection.config.hnsw_config.m == 16
    assert collection.config.hnsw_config.ef_construct == 100
    assert collection.config.metadata["embedding_model"] == "all-MiniLM-L6-v2"
    assert collection.config.metadata["hnsw_indexing_threshold_kb"] == 1
    assert any(alias.alias_name == "astrarag-corpus" for alias in aliases)
    payloads = [point.payload or {} for point in points]
    assert payloads
    assert all(
        "AstraRAG black-box indexing test" in str(payload["text"])
        for payload in payloads
    )


@pytest.mark.slow
@pytest.mark.skipif(
    os.getenv("RUN_SLOW_TESTS") != "1"
    or not EMBEDDING_MODEL.exists()
    or not EMBEDDING_TOKENIZER.exists(),
    reason="set RUN_SLOW_TESTS=1 and install the local embedding model",
)
def test_cli_indexes_multiple_pdfs_into_one_collection(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    write_pdf(documents / "first.pdf", "First document about attention mechanisms.")
    write_pdf(documents / "second.pdf", "Second document about retrieval models.")

    result = run_index(documents, tmp_path)

    assert result.returncode == 0, result.stderr
    assert "Successfully indexed 2 document(s)" in result.stdout

    client = QdrantClient(path=str(tmp_path / "qdrant"))
    try:
        point_count = client.count(
            collection_name="astrarag-corpus",
            exact=True,
        ).count
    finally:
        client.close()

    assert point_count >= 2


@pytest.mark.slow
@pytest.mark.skipif(
    os.getenv("RUN_SLOW_TESTS") != "1"
    or not EMBEDDING_MODEL.exists()
    or not EMBEDDING_TOKENIZER.exists(),
    reason="set RUN_SLOW_TESTS=1 and start the local embedding service",
)
def test_reindex_replaces_old_document_chunks(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    pdf_path = documents / "sample.pdf"
    write_pdf(pdf_path, "Content that must disappear after the rebuild.")
    first_result = run_index(documents, tmp_path)
    assert first_result.returncode == 0, first_result.stderr

    pdf_path.unlink()
    write_pdf(pdf_path, "Replacement content in the new index.")
    second_result = run_index(documents, tmp_path)
    assert second_result.returncode == 0, second_result.stderr

    client = QdrantClient(path=str(tmp_path / "qdrant"))
    try:
        points, _ = client.scroll(
            collection_name="astrarag-corpus",
            limit=10,
            with_payload=True,
        )
        collections = client.get_collections().collections
    finally:
        client.close()

    texts = [str((point.payload or {})["text"]) for point in points]
    assert any("Replacement content" in text for text in texts)
    assert all("must disappear" not in text for text in texts)
    assert len(collections) == 1


def test_cli_rejects_a_pdf_without_extractable_text(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    write_pdf(documents / "blank.pdf", "")

    result = run_index(documents, tmp_path)

    assert result.returncode != 0
    assert "No extractable text found in PDF" in result.stderr
