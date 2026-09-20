from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from oracle_builder.classification.features import L2Normalization, build_embedding_model
from oracle_builder.products.ingest import (
    ConvertChannels,
    InvertIntensity,
    ProbabilityToLogits,
)
from oracle_builder.training.losses import WeightedSparseCategoricalCrossentropy
from oracle_builder.training.train import build_and_compile_model


def _synthetic_input(config: dict[str, Any]) -> np.ndarray | dict[str, np.ndarray]:
    image = np.zeros((1, *config["data"]["input_shape"]), dtype="float32")
    specs = config.get("model", {}).get("auxiliary_features_fitted", [])
    if specs:
        return {
            "image": image,
            "metadata": np.zeros((1, len(specs)), dtype="float32"),
        }
    return image


def _load_keras_model(path: str | Path):
    return keras.models.load_model(
        path,
        compile=False,
        custom_objects={
            "L2Normalization": L2Normalization,
            "oracle_builder>L2Normalization": L2Normalization,
            "WeightedSparseCategoricalCrossentropy": WeightedSparseCategoricalCrossentropy,
            "oracle_builder>WeightedSparseCategoricalCrossentropy": WeightedSparseCategoricalCrossentropy,
            "ProbabilityToLogits": ProbabilityToLogits,
            "oracle_builder>ProbabilityToLogits": ProbabilityToLogits,
            "InvertIntensity": InvertIntensity,
            "oracle_builder>InvertIntensity": InvertIntensity,
            "ConvertChannels": ConvertChannels,
            "oracle_builder>ConvertChannels": ConvertChannels,
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
            if config.get("run", {}).get("task") == "embedding":
                rebuilt = build_embedding_model(rebuilt)
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
                expected = infer.structured_input_signature[1]
                if isinstance(sample, dict):
                    infer(**{name: tf.constant(sample[name]) for name in expected})
                else:
                    input_name = next(iter(expected))
                    infer(**{input_name: tf.constant(sample)})
            report["savedmodel_reload_checked"] = True
            prediction_ok = True
        except Exception as exc:
            report["errors"].append({"artifact": "export_savedmodel", "error": str(exc)})

    report["prediction_test_passed"] = prediction_ok
    return report


def load_model_for_run(
    run_dir: str | Path,
    config: dict[str, Any],
    *,
    prefer_savedmodel: bool = False,
):
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
    return load_model_from_dir(Path(run_dir) / "model", config, prefer_savedmodel=prefer_savedmodel)


def load_model_from_dir(
    model_path: str | Path,
    config: dict[str, Any],
    *,
    prefer_savedmodel: bool = False,
):
    """Load a model from a normal artifact directory.

    Unlike :func:`load_model_for_run`, this is intentionally not coupled to a
    run-root integrity check.  Stratified children live under their parent
    artifact and are validated by the parent bundle manifest.
    """
    model_path = Path(model_path)
    errors = []
    if prefer_savedmodel:
        try:
            return SavedModelPredictor(model_path / "export_savedmodel")
        except Exception as exc:
            errors.append(f"export_savedmodel: {exc}")
    try:
        return _load_keras_model(model_path / "final.keras")
    except Exception as exc:
        errors.append(f"final.keras: {exc}")
    try:
        model = build_and_compile_model(config)
        if config.get("run", {}).get("task") == "embedding":
            model = build_embedding_model(model)
        model.load_weights(model_path / "weights.weights.h5")
        return model
    except Exception as exc:
        errors.append(f"weights.weights.h5: {exc}")
    if not prefer_savedmodel:
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
        self.input_names = tuple(self.infer.structured_input_signature[1])
        self.input_name = self.input_names[0]
        self.embed = self.loaded.signatures.get("embed")

    def predict(self, x, verbose: int = 0):
        del verbose
        output = self.infer(**self._inputs(self.infer, x))
        probabilities = output.get("probabilities")
        if probabilities is None:
            probabilities = next(iter(output.values()))
        return probabilities.numpy()

    def predict_features(self, x, verbose: int = 0):
        del verbose
        if self.embed is None:
            raise RuntimeError("SavedModel has no embed signature")
        output = self.embed(**self._inputs(self.embed, x))
        return output["features"].numpy()

    def predict_embedding(self, x, verbose: int = 0):
        del verbose
        if self.embed is None:
            raise RuntimeError("SavedModel has no embed signature")
        output = self.embed(**self._inputs(self.embed, x))
        values = output.get("embedding")
        if values is None:
            values = output.get("features")
        if values is None:
            raise RuntimeError("SavedModel embed signature has no embedding output")
        return values.numpy()

    def predict_outputs(self, x, verbose: int = 0):
        del verbose
        output = self.infer(**self._inputs(self.infer, x))
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

    @staticmethod
    def _inputs(signature, value):
        """Convert array or named image/metadata inputs for SavedModel calls."""
        expected = signature.structured_input_signature[1]
        if isinstance(value, dict):
            missing = set(expected) - set(value)
            extra = set(value) - set(expected)
            if missing or extra:
                raise ValueError(
                    "SavedModel inputs do not match signature; "
                    f"missing={sorted(missing)}, extra={sorted(extra)}"
                )
            return {name: tf.constant(np.asarray(value[name], dtype="float32")) for name in expected}
        if len(expected) != 1:
            raise ValueError("This SavedModel requires named multi-input data")
        return {next(iter(expected)): tf.constant(np.asarray(value, dtype="float32"))}
