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


def test_index_reports_a_missing_directory(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    result = run_cli("index", str(missing), cwd=tmp_path)

    assert result.returncode != 0
    assert "Document directory does not exist" in result.stderr


def test_index_reports_an_empty_directory(tmp_path: Path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    result = run_cli("index", str(documents), cwd=tmp_path)

    assert result.returncode != 0
    assert "No PDF documents found" in result.stderr
