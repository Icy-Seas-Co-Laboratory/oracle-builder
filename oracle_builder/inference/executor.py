"""Resident TensorFlow execution adapters for portable inference bundles."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from oracle_builder.classification.features import build_embedding_model, build_feature_model


def execution_device_diagnostics() -> dict[str, Any]:
    """Describe the TensorFlow device that the resident inference will use."""
    try:
        import tensorflow as tf

        logical_gpus = tf.config.list_logical_devices("GPU")
        if logical_gpus:
            return {
                "gpu_accelerated": True,
                "accelerator": "gpu",
                "device_type": "GPU",
                "device_name": logical_gpus[0].name,
                "device_count": len(logical_gpus),
                "source": "tensorflow_logical_devices",
            }
        logical_cpus = tf.config.list_logical_devices("CPU")
        return {
            "gpu_accelerated": False,
            "accelerator": "cpu",
            "device_type": "CPU",
            "device_name": logical_cpus[0].name if logical_cpus else "/device:CPU:0",
            "device_count": 0,
            "source": "tensorflow_logical_devices",
        }
    except Exception as exc:  # pragma: no cover - TensorFlow is a runtime dependency
        return {
            "gpu_accelerated": False,
            "accelerator": "cpu",
            "device_type": "CPU",
            "device_name": "/device:CPU:0",
            "device_count": 0,
            "source": "tensorflow_unavailable",
            "diagnostic_error": f"{type(exc).__name__}: {exc}",
        }


class ClassificationExecutor:
    """One stable named-output classifier callable for the lifetime of a bundle."""

    def __init__(self, model: Any, input_shape: tuple[int, ...]):
        self.model = model
        self.input_shape = tuple(int(value) for value in input_shape)
        self.metadata_count = len(getattr(model, "inputs", [])) > 1
        if self.metadata_count:
            self.metadata_count = int(model.inputs[1].shape[-1])
        self.execution_diagnostics = execution_device_diagnostics()
        self.runtime = "adapter"
        self._call = self._build()

    def _build(self):
        # SavedModelPredictor and externally supplied adapters already expose a
        # stable named-output invocation contract.
        if hasattr(self.model, "predict_outputs"):
            self.runtime = (
                "savedmodel_signature"
                if type(self.model).__name__ == "SavedModelPredictor"
                else "named_output_adapter"
            )
            return lambda values: self.model.predict_outputs(values, verbose=0)
        try:
            import tensorflow as tf

            feature_model = build_feature_model(self.model)

            signature = (
                {"image": tf.TensorSpec([None, *self.input_shape], tf.float32, name="image"),
                 "metadata": tf.TensorSpec([None, self.metadata_count], tf.float32, name="metadata")}
                if self.metadata_count else tf.TensorSpec([None, *self.input_shape], tf.float32, name="inputs")
            )
            @tf.function(input_signature=[signature], reduce_retracing=True)
            def infer(inputs):
                return feature_model(inputs, training=False)

            self.runtime = "compiled_keras"
            return infer
        except (AttributeError, ValueError):
            self.runtime = "legacy_predict_adapter"
            return lambda values: self.model.predict(values, verbose=0)

    def predict(self, values: Any) -> dict[str, np.ndarray | None | str]:
        batch = ({key: np.asarray(value, dtype="float32") for key, value in values.items()}
                 if isinstance(values, dict) else np.asarray(values, dtype="float32"))
        outputs = self._call(batch)
        if isinstance(outputs, dict) and {"logits", "probabilities"}.issubset(outputs):
            return {
                "logits": np.asarray(outputs["logits"]),
                "probabilities": np.asarray(outputs["probabilities"]),
                "features": (
                    np.asarray(outputs["features"])
                    if outputs.get("features") is not None
                    else None
                ),
                "logits_source": str(outputs.get("logits_source", "model")),
            }
        probabilities = np.asarray(outputs)
        features = (
            np.asarray(self.model.predict_features(batch, verbose=0))
            if hasattr(self.model, "predict_features")
            else None
        )
        return {
            "logits": np.log(np.clip(probabilities, 1e-7, 1.0)).astype("float32"),
            "probabilities": probabilities,
            "features": features,
            "logits_source": "derived_log_probability",
        }

    def warm(self, batch_size: int) -> dict[str, Any]:
        started = time.perf_counter()
        values: Any = np.zeros((batch_size, *self.input_shape), dtype="float32")
        if self.metadata_count:
            values = {"image": values, "metadata": np.zeros((batch_size, self.metadata_count), dtype="float32")}
        self.predict(values)
        return {
            "runtime": self.runtime,
            "execution": self.execution_diagnostics,
            "batch_size": int(batch_size),
            "duration_ms": (time.perf_counter() - started) * 1000.0,
        }


class EmbeddingExecutor:
    """One stable embedding callable for representation-only products."""

    def __init__(self, model: Any, input_shape: tuple[int, ...]):
        self.model = model
        self.input_shape = tuple(int(value) for value in input_shape)
        self.execution_diagnostics = execution_device_diagnostics()
        self.runtime = "adapter"
        self._call = self._build()

    def _build(self):
        if hasattr(self.model, "predict_embedding"):
            self.runtime = "savedmodel_signature"
            return lambda values: self.model.predict_embedding(values, verbose=0)
        try:
            import tensorflow as tf

            embedding_model = build_embedding_model(self.model)

            @tf.function(
                input_signature=[
                    tf.TensorSpec([None, *self.input_shape], tf.float32, name="inputs")
                ],
                reduce_retracing=True,
            )
            def infer(inputs):
                return embedding_model(inputs, training=False)

            self.runtime = "compiled_keras"
            return infer
        except (AttributeError, ValueError):
            self.runtime = "legacy_predict_adapter"
            return lambda values: self.model.predict(values, verbose=0)

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(self._call(np.asarray(values, dtype="float32")))

    def warm(self, batch_size: int) -> dict[str, Any]:
        started = time.perf_counter()
        self.predict(np.zeros((batch_size, *self.input_shape), dtype="float32"))
        return {
            "runtime": self.runtime,
            "execution": self.execution_diagnostics,
            "batch_size": int(batch_size),
            "duration_ms": (time.perf_counter() - started) * 1000.0,
        }
