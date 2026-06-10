from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tensorflow import keras


def save_model_artifacts(model: keras.Model, run_dir: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    model_path = Path(run_dir) / "model"
    model_path.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "final_keras_saved": False,
        "weights_saved": False,
        "savedmodel_exported": False,
        "errors": [],
    }
    try:
        model.save(model_path / "final.keras")
        report["final_keras_saved"] = True
    except Exception as exc:
        report["errors"].append({"artifact": "final.keras", "error": str(exc)})
    try:
        model.save_weights(model_path / "weights.weights.h5")
        report["weights_saved"] = True
    except Exception as exc:
        report["errors"].append({"artifact": "weights.weights.h5", "error": str(exc)})
    if config.get("output", {}).get("export_savedmodel", True):
        try:
            export_dir = model_path / "export_savedmodel"
            if hasattr(model, "export"):
                model.export(export_dir)
            else:
                model.save(export_dir)
            report["savedmodel_exported"] = True
        except Exception as exc:
            report["errors"].append({"artifact": "export_savedmodel", "error": str(exc)})
    return report


def write_load_test_report(run_dir: str | Path, report: dict[str, Any]) -> None:
    Path(run_dir, "model", "load_test_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

