"""Load and validate retrieval evaluation queries from JSON datasets."""

import json
from pathlib import Path

from pydantic import TypeAdapter

from astrarag.schemas import EvaluationQuery


def load_evaluation_dataset(
    path: Path,
) -> list[EvaluationQuery]:

    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    adapter = TypeAdapter(list[EvaluationQuery])

    return adapter.validate_python(data)
