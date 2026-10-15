"""Optional classifier head attached to a trained classification bundle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_posthoc_head(run_dir: str | Path) -> Any | None:
    """Load a head only when its completed report and estimator are present."""
    root = Path(run_dir) / "model" / "posthoc"
    report_path = root / "report.json"
    estimator_path = root / "estimator.joblib"
    if not report_path.is_file() or not estimator_path.is_file():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "complete":
        return None
    import joblib

    return joblib.load(estimator_path)


def posthoc_output(estimator: Any, embedding: np.ndarray, num_classes: int | None = None) -> dict[str, Any]:
    """Return the fitted head's own class distribution and vocabulary."""
    values = np.asarray(embedding, dtype="float32").reshape(1, -1)
    raw = np.asarray(estimator.predict_proba(values)[0], dtype="float64")
    if num_classes is None:
        num_classes = max(int(value) for value in estimator.classes_) + 1
    probabilities = np.zeros(num_classes, dtype="float64")
    for class_index, probability in zip(estimator.classes_, raw, strict=True):
        index = int(class_index)
        if index < 0 or index >= num_classes:
            raise ValueError(f"Post-hoc class index {index} is outside the model output")
        probabilities[index] = probability
    labels = getattr(estimator, "oracle_posthoc_labels_", {})
    selected = int(np.argmax(probabilities))
    return {
        "type": "classification",
        "representation": "fused_embedding",
        "class_index": selected,
        "label_name": labels.get(selected) or labels.get(str(selected)),
        "classes": [int(value) for value in estimator.classes_],
        "probabilities": probabilities.tolist(),
    }
