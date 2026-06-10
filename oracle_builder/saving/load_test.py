from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from oracle_builder.training.train import build_and_compile_model


def _synthetic_input(config: dict[str, Any]) -> np.ndarray:
    return np.zeros((1, *config["data"]["input_shape"]), dtype="float32")


def run_load_tests(run_dir: str | Path, config: dict[str, Any], initial_report: dict[str, Any] | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "final_keras_saved": False,
        "final_keras_reloaded": False,
        "weights_saved": False,
        "weights_reloaded_into_rebuilt_model": False,
        "savedmodel_exported": False,
        "savedmodel_reload_checked": False,
        "prediction_test_passed": False,
        "errors": [],
    }
    if initial_report:
        report.update({key: value for key, value in initial_report.items() if key != "errors"})
        report["errors"].extend(initial_report.get("errors", []))

    model_path = Path(run_dir) / "model"
    sample = _synthetic_input(config)
    prediction_ok = False
    if (model_path / "final.keras").exists():
        report["final_keras_saved"] = True
        try:
            loaded = keras.models.load_model(model_path / "final.keras")
            loaded.predict(sample, verbose=0)
            report["final_keras_reloaded"] = True
            prediction_ok = True
        except Exception as exc:
            report["errors"].append({"artifact": "final.keras", "error": str(exc)})

    if (model_path / "weights.weights.h5").exists():
        report["weights_saved"] = True
        try:
            rebuilt = build_and_compile_model(config)
            rebuilt.load_weights(model_path / "weights.weights.h5")
            rebuilt.predict(sample, verbose=0)
            report["weights_reloaded_into_rebuilt_model"] = True
            prediction_ok = True
        except Exception as exc:
            report["errors"].append({"artifact": "weights.weights.h5", "error": str(exc)})

    export_dir = model_path / "export_savedmodel"
    if export_dir.exists():
        report["savedmodel_exported"] = True
        try:
            loaded_export = tf.saved_model.load(str(export_dir))
            infer = loaded_export.signatures.get("serving_default")
            if infer:
                input_name = next(iter(infer.structured_input_signature[1]))
                infer(**{input_name: tf.constant(sample)})
            report["savedmodel_reload_checked"] = True
            prediction_ok = True
        except Exception as exc:
            report["errors"].append({"artifact": "export_savedmodel", "error": str(exc)})

    report["prediction_test_passed"] = prediction_ok
    return report


def load_model_for_run(run_dir: str | Path, config: dict[str, Any]):
    model_path = Path(run_dir) / "model"
    errors = []
    try:
        return keras.models.load_model(model_path / "final.keras")
    except Exception as exc:
        errors.append(f"final.keras: {exc}")
    try:
        model = build_and_compile_model(config)
        model.load_weights(model_path / "weights.weights.h5")
        return model
    except Exception as exc:
        errors.append(f"weights.weights.h5: {exc}")
    try:
        return SavedModelPredictor(model_path / "export_savedmodel")
    except Exception as exc:
        errors.append(f"export_savedmodel: {exc}")
    raise RuntimeError("No loadable Keras model found. " + " | ".join(errors))


class SavedModelPredictor:
    def __init__(self, export_dir: str | Path):
        export_dir = Path(export_dir)
        if not export_dir.exists():
            raise FileNotFoundError(export_dir)
        self.loaded = tf.saved_model.load(str(export_dir))
        self.infer = self.loaded.signatures.get("serving_default")
        if self.infer is None:
            raise RuntimeError("SavedModel has no serving_default signature")
        self.input_name = next(iter(self.infer.structured_input_signature[1]))

    def predict(self, x, verbose: int = 0):
        del verbose
        output = self.infer(**{self.input_name: tf.constant(x)})
        first = next(iter(output.values()))
        return first.numpy()
