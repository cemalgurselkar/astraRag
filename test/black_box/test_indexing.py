"""Verify PDF-to-Qdrant indexing through the installed CLI."""

import os
import subprocess
import sys
from pathlib import Path

import pymupdf
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMBEDDING_MODEL = PROJECT_ROOT / "models/all-MiniLM-L6-v2"


@pytest.mark.slow
@pytest.mark.skipif(
    os.getenv("RUN_SLOW_TESTS") != "1" or not EMBEDDING_MODEL.exists(),
    reason="set RUN_SLOW_TESTS=1 and install the local embedding model",
)
def test_cli_indexes_a_small_pdf_without_a_gemini_key(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    pdf_path = documents / "sample.pdf"

    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "AstraRAG black-box indexing test document.")
    pdf.save(pdf_path)
    pdf.close()

    environment = os.environ.copy()
    environment.pop("GEMINI_API_KEY", None)
    environment["EMBEDDING_MODEL_PATH"] = str(EMBEDDING_MODEL)
    environment["QDRANT_PATH"] = str(tmp_path / "qdrant")

    result = subprocess.run(
        [sys.executable, "-m", "astrarag.cli", "index", str(documents)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert "Successfully indexed 1 document(s)" in result.stdout
    assert (tmp_path / "qdrant").exists()
