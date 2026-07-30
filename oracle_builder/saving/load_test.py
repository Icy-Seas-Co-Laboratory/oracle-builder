from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from oracle_builder.classification.features import L2Normalization
from oracle_builder.training.losses import WeightedSparseCategoricalCrossentropy
from oracle_builder.training.train import build_and_compile_model


def _synthetic_input(config: dict[str, Any]) -> np.ndarray:
    return np.zeros((1, *config["data"]["input_shape"]), dtype="float32")


def _load_keras_model(path: str | Path):
    return keras.models.load_model(
        path,
        compile=False,
        custom_objects={
            "L2Normalization": L2Normalization,
            "oracle_builder>L2Normalization": L2Normalization,
            "WeightedSparseCategoricalCrossentropy": WeightedSparseCategoricalCrossentropy,
            "oracle_builder>WeightedSparseCategoricalCrossentropy": WeightedSparseCategoricalCrossentropy,
        },
    )


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
            loaded = _load_keras_model(model_path / "final.keras")
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
    from oracle_builder.artifacts.layout import RunLayout
    from oracle_builder.artifacts.run import validate_run_artifact

    layout = RunLayout(run_dir)
    if layout.manifest.exists():
        validation = validate_run_artifact(layout.root)
        if not validation["valid"]:
            raise RuntimeError(
                "Run artifact integrity validation failed: "
                + "; ".join(validation["errors"])
            )
    model_path = Path(run_dir) / "model"
    errors = []
    try:
        return _load_keras_model(model_path / "final.keras")
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
        self.embed = self.loaded.signatures.get("embed")

    def predict(self, x, verbose: int = 0):
        del verbose
        output = self.infer(**{self.input_name: tf.constant(x)})
        probabilities = output.get("probabilities")
        if probabilities is None:
            probabilities = next(iter(output.values()))
        return probabilities.numpy()

    def predict_features(self, x, verbose: int = 0):
        del verbose
        if self.embed is None:
            raise RuntimeError("SavedModel has no embed signature")
        input_name = next(iter(self.embed.structured_input_signature[1]))
        output = self.embed(**{input_name: tf.constant(x)})
        return output["features"].numpy()

    def predict_outputs(self, x, verbose: int = 0):
        del verbose
        output = self.infer(**{self.input_name: tf.constant(x)})
        probabilities = output.get("probabilities")
        if probabilities is None:
            probabilities = next(iter(output.values()))
        logits = output.get("logits")
        if logits is None:
            values = np.asarray(probabilities.numpy())
            logits_value = np.log(np.clip(values, 1e-7, 1.0))
            logits_source = "derived_log_probability"
        else:
            logits_value = logits.numpy()
            logits_source = "model"
        features = None
        if "features" in output:
            features = output["features"].numpy()
        elif self.embed is not None:
            features = self.predict_features(x)
        return {
            "logits": logits_value,
            "probabilities": probabilities.numpy(),
            "features": features,
            "logits_source": logits_source,
        }
