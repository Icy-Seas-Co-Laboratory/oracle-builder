"""Resident TensorFlow execution adapters for portable inference bundles."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from oracle_builder.classification.features import build_feature_model


class ClassificationExecutor:
    """One stable named-output classifier callable for the lifetime of a bundle."""

    def __init__(self, model: Any, input_shape: tuple[int, ...]):
        self.model = model
        self.input_shape = tuple(int(value) for value in input_shape)
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

            @tf.function(
                input_signature=[
                    tf.TensorSpec([None, *self.input_shape], tf.float32, name="inputs")
                ],
                reduce_retracing=True,
            )
            def infer(inputs):
                return feature_model(inputs, training=False)

            self.runtime = "compiled_keras"
            return infer
        except (AttributeError, ValueError):
            self.runtime = "legacy_predict_adapter"
            return lambda values: self.model.predict(values, verbose=0)

    def predict(self, values: np.ndarray) -> dict[str, np.ndarray | None | str]:
        batch = np.asarray(values, dtype="float32")
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
        self.predict(np.zeros((batch_size, *self.input_shape), dtype="float32"))
        return {
            "runtime": self.runtime,
            "batch_size": int(batch_size),
            "duration_ms": (time.perf_counter() - started) * 1000.0,
        }
