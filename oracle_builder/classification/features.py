from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


FEATURE_LAYER_NAME = "features"
LOGITS_LAYER_NAME = "logits"
DEFAULT_EMBEDDING_DIM = 256


@keras.utils.register_keras_serializable(package="oracle_builder")
class L2Normalization(layers.Layer):
    """Normalize feature vectors, including a deterministic zero-vector fallback."""

    def call(self, inputs):
        norm = tf.norm(inputs, axis=-1, keepdims=True)
        normalized = tf.math.divide_no_nan(inputs, norm)
        fallback = tf.one_hot(
            tf.zeros(tf.shape(inputs)[:-1], dtype=tf.int32),
            depth=tf.shape(inputs)[-1],
            dtype=inputs.dtype,
        )
        return tf.where(norm > 0, normalized, fallback)


def classification_head(
    x,
    num_classes: int,
    config: dict[str, Any],
    *,
    dropout_default: float = 0.0,
    normalize_default: bool = True,
):
    """Attach the standard fixed-size embedding and classification head."""
    model_config = config.get("model", {})
    embedding_dim = int(model_config.get("embedding_dim", DEFAULT_EMBEDDING_DIM))
    normalize = bool(model_config.get("normalize_embeddings", normalize_default))
    dropout = float(model_config.get("dropout", dropout_default))
    if embedding_dim < 1:
        raise ValueError("model.embedding_dim must be a positive integer")
    if not 0 <= dropout < 1:
        raise ValueError("model.dropout must be in [0, 1)")

    x = layers.Dense(embedding_dim, name="embedding_projection")(x)
    if normalize:
        features = L2Normalization(name=FEATURE_LAYER_NAME)(x)
    else:
        features = layers.Activation("linear", name=FEATURE_LAYER_NAME)(x)
    x = features
    if dropout:
        x = layers.Dropout(dropout, name="classifier_dropout")(x)
    logits = layers.Dense(num_classes, name=LOGITS_LAYER_NAME)(x)
    return layers.Activation("softmax", name="predictions")(logits)


def build_feature_model(model: keras.Model) -> keras.Model:
    """Return an inference view containing logits, probabilities, and embeddings."""
    features = model.get_layer(FEATURE_LAYER_NAME).output
    logits = model.get_layer(LOGITS_LAYER_NAME).output
    return keras.Model(
        model.input,
        {
            "logits": logits,
            "probabilities": model.output,
            "features": features,
        },
        name=f"{model.name}_with_features",
    )


def build_embedding_model(model: keras.Model) -> keras.Model:
    """Return the shared classifier backbone through its fixed-size features."""
    inputs = model.inputs[0] if getattr(model, "inputs", None) else model.input
    return keras.Model(
        inputs,
        model.get_layer(FEATURE_LAYER_NAME).output,
        name=f"{model.name}_encoder",
    )


def build_self_supervised_embedding_model(model: keras.Model) -> keras.Model:
    """Return the classifier encoder before serving-time feature normalization.

    Classifier exports intentionally use ``features`` so downstream consumers get
    the configured (usually L2-normalized) embedding contract.  SSL projection
    heads instead need access to the learned embedding values before that
    normalization, both to preserve magnitude information and to make variance
    regularization meaningful.
    """
    inputs = model.inputs[0] if getattr(model, "inputs", None) else model.input
    return keras.Model(
        inputs,
        model.get_layer("embedding_projection").output,
        name=f"{model.name}_self_supervised_encoder",
    )


# Legacy import compatibility.
build_pretraining_embedding_model = build_self_supervised_embedding_model


def predict_with_features(model, x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Predict probabilities and features, tolerating legacy/non-Keras predictors."""
    if hasattr(model, "predict_features"):
        return (
            np.asarray(model.predict(x, verbose=0)),
            np.asarray(model.predict_features(x, verbose=0)),
        )
    try:
        outputs = build_feature_model(model).predict(x, verbose=0)
    except (AttributeError, ValueError):
        return np.asarray(model.predict(x, verbose=0)), None
    return np.asarray(outputs["probabilities"]), np.asarray(outputs["features"])


def predict_classification_outputs(
    model,
    x: np.ndarray,
    *,
    batch_size: int | None = None,
) -> dict[str, np.ndarray | None]:
    """Return the complete classification inference contract for a batch."""
    if hasattr(model, "predict_outputs"):
        outputs = model.predict_outputs(x, verbose=0)
        return {
            "logits": np.asarray(outputs["logits"]),
            "probabilities": np.asarray(outputs["probabilities"]),
            "features": (
                np.asarray(outputs["features"])
                if outputs.get("features") is not None
                else None
            ),
            "logits_source": outputs.get("logits_source", "model"),
        }
    # Promoted external products expose the standard tensors as a Keras output
    # dictionary, avoiding a dependence on their original internal graph.
    predict_options = {"verbose": 0}
    if batch_size is not None:
        predict_options["batch_size"] = batch_size
    direct = model.predict(x, **predict_options)
    if isinstance(direct, dict) and {"logits", "probabilities"}.issubset(direct):
        return {
            "logits": np.asarray(direct["logits"]),
            "probabilities": np.asarray(direct["probabilities"]),
            "features": (
                np.asarray(direct["features"])
                if direct.get("features") is not None
                else None
            ),
            "logits_source": "model",
        }
    try:
        outputs = build_feature_model(model).predict(
            x, **predict_options
        )
        return {
            "logits": np.asarray(outputs["logits"]),
            "probabilities": np.asarray(outputs["probabilities"]),
            "features": np.asarray(outputs["features"]),
            "logits_source": "model",
        }
    except (AttributeError, ValueError):
        probabilities = np.asarray(
            model.predict(x, **predict_options)
        )
        features = (
            np.asarray(model.predict_features(x, verbose=0))
            if hasattr(model, "predict_features")
            else None
        )
        # Softmax logits are identifiable only up to an additive constant.
        # log(p) is the stable canonical representative for probability-only
        # external or historical predictors.
        logits = np.log(np.clip(probabilities, 1e-7, 1.0))
        return {
            "logits": logits.astype("float32"),
            "probabilities": probabilities,
            "features": features,
            "logits_source": "derived_log_probability",
        }
