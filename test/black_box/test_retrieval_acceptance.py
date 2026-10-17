"""Optional quality gate around the executable retrieval benchmark."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MINIMUM_HIT_AT_5 = {
    "Dense": 0.55,
    "BM25": 0.70,
    "Hybrid RRF": 0.75,
}


def hit_at_5_scores(output: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    for line in output.splitlines():
        match = re.match(r"^(Dense|BM25|Hybrid RRF)\s+([0-9.]+)", line)
        if match:
            scores[match.group(1)] = float(match.group(2))
    return scores


@pytest.mark.live
@pytest.mark.skipif(
    os.getenv("RUN_RETRIEVAL_ACCEPTANCE") != "1",
    reason="set RUN_RETRIEVAL_ACCEPTANCE=1 to run the full local benchmark",
)
def test_retrieval_strategies_keep_their_quality_floors() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/benchmark_retrieval.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )

    assert result.returncode == 0, result.stderr
    scores = hit_at_5_scores(result.stdout)
    assert set(scores) == set(MINIMUM_HIT_AT_5), result.stdout

    for strategy, floor in MINIMUM_HIT_AT_5.items():
        assert scores[strategy] >= floor, (
            f"{strategy} Hit@5 {scores[strategy]} fell below the {floor} floor"
        )
