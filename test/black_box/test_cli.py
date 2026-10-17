"""Exercise CLI behavior in a separate Python process."""

import os
import subprocess
import sys
from pathlib import Path


def run_cli(
    *arguments: str, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("GEMINI_API_KEY", None)
    return subprocess.run(
        [sys.executable, "-m", "astrarag.cli", *arguments],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_help_lists_public_commands() -> None:
    result = run_cli("--help")

    assert result.returncode == 0
    assert "index" in result.stdout
    assert "serve" in result.stdout
    assert "evaluate" in result.stdout


def test_index_help_documents_the_path_argument() -> None:
    result = run_cli("index", "--help")

    assert result.returncode == 0
    assert "PDF directory" in result.stdout


def test_unknown_commands_fail() -> None:
    result = run_cli("explode")

    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_index_without_a_path_uses_the_default_document_directory(
    tmp_path: Path,
) -> None:
    result = run_cli("index", cwd=tmp_path)

    assert result.returncode != 0
    assert "Document directory does not exist" in result.stderr


def test_index_reports_a_missing_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    result = run_cli("index", str(missing), cwd=tmp_path)

    assert result.returncode != 0
    assert "Document directory does not exist" in result.stderr


def test_index_rejects_a_file_instead_of_a_directory(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 stub")
    result = run_cli("index", str(pdf_path), cwd=tmp_path)

    assert result.returncode != 0
    assert "Expected a directory" in result.stderr


def test_index_reports_an_empty_directory(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    result = run_cli("index", str(documents), cwd=tmp_path)

    assert result.returncode != 0
    assert "No PDF documents found" in result.stderr


def test_evaluate_reports_the_dataset_and_exits_successfully(
    tmp_path: Path,
) -> None:
    """Pins the current evaluate contract until the command is implemented."""
    dataset = tmp_path / "evaluation.json"
    dataset.write_text("[]", encoding="utf-8")
    result = run_cli("evaluate", str(dataset), cwd=tmp_path)

    assert result.returncode == 0
    assert "Evaluation dataset:" in result.stdout
