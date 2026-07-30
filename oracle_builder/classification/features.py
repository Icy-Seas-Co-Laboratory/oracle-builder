from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


FEATURE_LAYER_NAME = "features"
DEFAULT_EMBEDDING_DIM = 256


@keras.utils.register_keras_serializable(package="oracle_builder")
class L2Normalization(layers.Layer):
    """Normalize nonzero feature vectors to exact unit L2 length."""

    def call(self, inputs):
        return tf.math.divide_no_nan(inputs, tf.norm(inputs, axis=-1, keepdims=True))


def classification_head(
    x,
    num_classes: int,
    config: dict[str, Any],
    *,
    dropout_default: float = 0.0,
):
    """Attach the standard fixed-size embedding and classification head."""
    model_config = config.get("model", {})
    embedding_dim = int(model_config.get("embedding_dim", DEFAULT_EMBEDDING_DIM))
    normalize = bool(model_config.get("normalize_embeddings", True))
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
    return layers.Dense(num_classes, activation="softmax", name="predictions")(x)


def build_feature_model(model: keras.Model) -> keras.Model:
    """Return an inference view containing probabilities and embeddings."""
    features = model.get_layer(FEATURE_LAYER_NAME).output
    return keras.Model(
        model.input,
        {"probabilities": model.output, "features": features},
        name=f"{model.name}_with_features",
    )


def build_embedding_model(model: keras.Model) -> keras.Model:
    """Return the shared classifier backbone through its fixed-size features."""
    return keras.Model(
        model.input,
        model.get_layer(FEATURE_LAYER_NAME).output,
        name=f"{model.name}_encoder",
    )


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
