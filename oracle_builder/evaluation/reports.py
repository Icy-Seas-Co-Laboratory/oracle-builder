from __future__ import annotations

from pathlib import Path
from typing import Any

from oracle_builder.data.sqlite_dataset import load_arrays
from oracle_builder.evaluation.classification import evaluate_classification
from oracle_builder.evaluation.segmentation import evaluate_segmentation


def evaluate_run_model(model, config: dict[str, Any], input_path: str | Path, run_dir: str | Path, split: str = "test") -> dict[str, Any]:
    try:
        x, y, records = load_arrays(input_path, config, split=split)
    except ValueError:
        x, y, records = load_arrays(input_path, config, split="validation")
    if config["run"]["task"] == "classification":
        return evaluate_classification(model, x, y, records, run_dir)
    return evaluate_segmentation(model, x, y, records, run_dir)

