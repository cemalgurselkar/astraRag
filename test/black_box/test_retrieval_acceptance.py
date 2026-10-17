"""Optional quality gate around the executable retrieval benchmark."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.live
@pytest.mark.skipif(
    os.getenv("RUN_RETRIEVAL_ACCEPTANCE") != "1",
    reason="set RUN_RETRIEVAL_ACCEPTANCE=1 to run the full local benchmark",
)
def test_hybrid_retrieval_keeps_its_quality_floor() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/benchmark_retrieval.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )

    assert result.returncode == 0, result.stderr
    row = re.search(r"^Hybrid RRF\s+([0-9.]+)", result.stdout, re.MULTILINE)
    assert row, result.stdout
    assert float(row.group(1)) >= 0.75
