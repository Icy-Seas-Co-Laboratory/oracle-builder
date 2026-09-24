"""Composable, serializable building blocks for image classification models.

This module deliberately contains no model-family code.  An encoder supplies a
spatial ``feature_map`` and the helpers below assemble pooling, representation,
metadata fusion and prediction layers around it.  The names used here are the
v2 representation contract and are safe to use in exported Keras models.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


OUTPUT_NAMES = (
    "feature_map",
    "image_embedding",
    "metadata_embedding",
    "fused_embedding",
    "projection_embedding",
    "logits",
    "probabilities",
)


def _positive(value: int, field: str) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


@keras.utils.register_keras_serializable(package="oracle_builder")
class L2Normalize(layers.Layer):
    """L2 normalize vectors, returning zeros for an all-zero input."""

    def __init__(self, epsilon: float = 1e-12, **kwargs: Any):
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.math.l2_normalize(inputs, axis=-1, epsilon=self.epsilon)

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "epsilon": self.epsilon}


@keras.utils.register_keras_serializable(package="oracle_builder")
class ZeroLike(layers.Layer):
    """Serializable zero representation used for absent optional metadata."""

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.zeros_like(inputs)


@keras.utils.register_keras_serializable(package="oracle_builder")
class GeMPooling2D(layers.Layer):
    """Generalized-mean pooling with an optionally trainable positive exponent."""

    def __init__(
        self, p: float = 3.0, epsilon: float = 1e-6, trainable_p: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if p <= 0:
            raise ValueError("GeM p must be positive")
        if epsilon <= 0:
            raise ValueError("GeM epsilon must be positive")
        self.p = float(p)
        self.epsilon = float(epsilon)
        self.trainable_p = bool(trainable_p)

    def build(self, input_shape: tf.TensorShape) -> None:
        # softplus(raw_p) + epsilon keeps p valid without clipping gradients.
        target = max(self.p - self.epsilon, self.epsilon)
        raw_p = float(tf.math.log(tf.math.expm1(target)).numpy())
        self._raw_p = self.add_weight(
            name="raw_p", shape=(), initializer=keras.initializers.Constant(raw_p),
            trainable=self.trainable_p,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        p = tf.nn.softplus(self._raw_p) + tf.cast(self.epsilon, inputs.dtype)
        x = tf.maximum(tf.cast(inputs, self.compute_dtype), tf.cast(self.epsilon, self.compute_dtype))
        return tf.pow(tf.reduce_mean(tf.pow(x, p), axis=[1, 2]), 1.0 / p)

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "p": self.p, "epsilon": self.epsilon,
                "trainable_p": self.trainable_p}


@keras.utils.register_keras_serializable(package="oracle_builder")
class CosineClassifier(layers.Layer):
    """A normalized linear classifier, optionally scaled by a fixed logit scale."""

    def __init__(self, num_classes: int, scale: float = 16.0, use_bias: bool = False, **kwargs: Any):
        super().__init__(**kwargs)
        self.num_classes = _positive(num_classes, "num_classes")
        self.scale = float(scale)
        self.use_bias = bool(use_bias)

    def build(self, input_shape: tf.TensorShape) -> None:
        width = _positive(input_shape[-1], "CosineClassifier input width")
        self.kernel = self.add_weight(
            name="kernel", shape=(width, self.num_classes), initializer="glorot_uniform", trainable=True,
        )
        self.bias = self.add_weight(name="bias", shape=(self.num_classes,), initializer="zeros", trainable=True) if self.use_bias else None
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        logits = self.scale * tf.matmul(
            tf.math.l2_normalize(inputs, axis=-1), tf.math.l2_normalize(self.kernel, axis=0)
        )
        return tf.nn.bias_add(logits, self.bias) if self.bias is not None else logits

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "num_classes": self.num_classes, "scale": self.scale,
                "use_bias": self.use_bias}


@keras.utils.register_keras_serializable(package="oracle_builder")
class PrototypeClassifier(layers.Layer):
    """Learn one or more class prototypes in cosine or Euclidean space."""

    def __init__(
        self, num_classes: int, prototypes_per_class: int = 1,
        metric: str = "cosine", scale: float = 16.0, **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.num_classes = _positive(num_classes, "num_classes")
        self.prototypes_per_class = _positive(prototypes_per_class, "prototypes_per_class")
        self.metric = str(metric).lower()
        if self.metric not in {"cosine", "euclidean"}:
            raise ValueError("PrototypeClassifier metric must be cosine or euclidean")
        self.scale = float(scale)

    def build(self, input_shape: tf.TensorShape) -> None:
        width = _positive(input_shape[-1], "PrototypeClassifier input width")
        self.prototypes = self.add_weight(
            name="prototypes",
            shape=(self.num_classes, self.prototypes_per_class, width),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        if self.metric == "cosine":
            values = tf.math.l2_normalize(inputs, axis=-1)
            prototypes = tf.math.l2_normalize(self.prototypes, axis=-1)
            scores = tf.einsum("bd,cpd->bcp", values, prototypes)
        else:
            values = inputs[:, tf.newaxis, tf.newaxis, :]
            scores = -tf.reduce_sum(tf.square(values - self.prototypes[tf.newaxis, ...]), axis=-1)
        # Max assignment makes multiple prototypes useful without adding a
        # latent routing system; one prototype is the standard prototype head.
        return tf.cast(self.scale, scores.dtype) * tf.reduce_max(scores, axis=-1)

    def get_config(self) -> dict[str, Any]:
        return {
            **super().get_config(),
            "num_classes": self.num_classes,
            "prototypes_per_class": self.prototypes_per_class,
            "metric": self.metric,
            "scale": self.scale,
        }


def build_pooler(feature_map: tf.Tensor, kind: str = "avg", *, gem_p: float = 3.0,
                 gem_trainable_p: bool = False, name: str = "image_pooling") -> tf.Tensor:
    """Pool a rank-4 encoder map using avg, max, avg_max, or gem."""
    kind = str(kind).lower()
    if kind == "avg":
        return layers.GlobalAveragePooling2D(name=name)(feature_map)
    if kind == "max":
        return layers.GlobalMaxPooling2D(name=name)(feature_map)
    if kind == "avg_max":
        average = layers.GlobalAveragePooling2D(name=f"{name}_avg")(feature_map)
        maximum = layers.GlobalMaxPooling2D(name=f"{name}_max")(feature_map)
        return layers.Concatenate(name=name)([average, maximum])
    if kind == "gem":
        return GeMPooling2D(p=gem_p, trainable_p=gem_trainable_p, name=name)(feature_map)
    raise ValueError("pooler kind must be one of: avg, max, avg_max, gem")


def build_image_projection(inputs: tf.Tensor, dimension: int, *, activation: str | None = None,
                           dropout: float = 0.0, l2_normalize: bool = False,
                           name: str = "image_embedding") -> tf.Tensor:
    """Project pooled image features into the image representation space."""
    _positive(dimension, "image projection dimension")
    if not 0 <= float(dropout) < 1:
        raise ValueError("image projection dropout must be in [0, 1)")
    x = layers.Dense(dimension, activation=activation, name=f"{name}_dense")(inputs)
    if dropout:
        x = layers.Dropout(float(dropout), name=f"{name}_dropout")(x)
    return L2Normalize(name=name)(x) if l2_normalize else layers.Activation("linear", name=name)(x)


def build_metadata_encoder(metadata: tf.Tensor, *, kind: str = "direct", dimension: int | None = None,
                           hidden_units: tuple[int, ...] | list[int] = (), activation: str = "relu",
                           dropout: float = 0.0, name: str = "metadata_embedding") -> tf.Tensor:
    """Return direct metadata or a configurable MLP embedding."""
    kind = str(kind).lower()
    if kind not in {"direct", "mlp"}:
        raise ValueError("metadata encoder kind must be 'direct' or 'mlp'")
    if not 0 <= float(dropout) < 1:
        raise ValueError("metadata encoder dropout must be in [0, 1)")
    if kind == "direct":
        if dimension is not None:
            return layers.Dense(_positive(dimension, "metadata dimension"), name=name)(metadata)
        return layers.Activation("linear", name=name)(metadata)
    x = metadata
    for index, units in enumerate(hidden_units):
        x = layers.Dense(_positive(units, "metadata hidden_units"), activation=activation,
                         name=f"{name}_hidden_{index + 1}")(x)
        if dropout:
            x = layers.Dropout(float(dropout), name=f"{name}_dropout_{index + 1}")(x)
    if dimension is None:
        raise ValueError("metadata MLP requires a dimension")
    return layers.Dense(_positive(dimension, "metadata dimension"), name=name)(x)


def build_fusion(image_embedding: tf.Tensor, metadata_embedding: tf.Tensor | None = None, *,
                 kind: str = "concat", dimension: int | None = None, activation: str | None = None,
                 name: str = "fused_embedding") -> tf.Tensor:
    """Fuse image and optional metadata representations without implicit inputs."""
    kind = str(kind).lower()
    if kind not in {"concat", "projected"}:
        raise ValueError("fusion kind must be 'concat' or 'projected'")
    inputs = [image_embedding] + ([metadata_embedding] if metadata_embedding is not None else [])
    if len(inputs) == 1:
        joined = layers.Activation("linear", name=f"{name}_identity")(image_embedding)
    else:
        joined = layers.Concatenate(name=f"{name}_concat")(inputs)
    if kind == "projected":
        if dimension is None:
            raise ValueError("projected fusion requires a dimension")
        return layers.Dense(_positive(dimension, "fusion dimension"), activation=activation, name=name)(joined)
    return layers.Activation("linear", name=name)(joined)


def build_classifier_head(inputs: tf.Tensor, num_classes: int, *, kind: str = "linear",
                          hidden_units: tuple[int, ...] | list[int] = (), activation: str = "relu",
                          dropout: float = 0.0, cosine_scale: float = 16.0,
                          prototypes_per_class: int = 1, prototype_metric: str = "cosine",
                          name: str = "logits") -> tf.Tensor:
    """Build a linear, MLP, cosine, or prototype supervised logits head."""
    _positive(num_classes, "num_classes")
    kind = str(kind).lower()
    if kind not in {"linear", "mlp", "cosine", "prototype"}:
        raise ValueError("classifier head kind must be linear, mlp, cosine, or prototype")
    if not 0 <= float(dropout) < 1:
        raise ValueError("classifier dropout must be in [0, 1)")
    if kind == "cosine":
        return CosineClassifier(num_classes, scale=cosine_scale, name=name)(inputs)
    if kind == "prototype":
        return PrototypeClassifier(
            num_classes,
            prototypes_per_class=prototypes_per_class,
            metric=prototype_metric,
            scale=cosine_scale,
            name=name,
        )(inputs)
    x = inputs
    if kind == "mlp":
        for index, units in enumerate(hidden_units):
            x = layers.Dense(_positive(units, "classifier hidden_units"), activation=activation,
                             name=f"{name}_hidden_{index + 1}")(x)
            if dropout:
                x = layers.Dropout(float(dropout), name=f"{name}_dropout_{index + 1}")(x)
    return layers.Dense(num_classes, name=name)(x)


def build_named_output_model(inputs: keras.KerasTensor | list[keras.KerasTensor], *,
                             feature_map: tf.Tensor, image_embedding: tf.Tensor,
                             fused_embedding: tf.Tensor, projection_embedding: tf.Tensor,
                             logits: tf.Tensor, metadata_embedding: tf.Tensor | None = None,
                             name: str = "composable_classifier") -> keras.Model:
    """Create the versioned v2 named-output inference view.

    Metadata is optional.  For image-only models its output aliases the image
    representation, retaining one stable tensor dictionary without fabricating
    an empty metadata tensor.
    """
    if metadata_embedding is None:
        metadata_embedding = ZeroLike(name="metadata_embedding_empty")(image_embedding)
    probabilities = layers.Activation("softmax", name="probabilities")(logits)
    outputs = {
        "feature_map": feature_map,
        "image_embedding": image_embedding,
        "metadata_embedding": metadata_embedding,
        "fused_embedding": fused_embedding,
        "projection_embedding": projection_embedding,
        "logits": logits,
        "probabilities": probabilities,
    }
    return keras.Model(inputs=inputs, outputs=outputs, name=name)


def build_composable_classifier(*, image: keras.KerasTensor, feature_map: tf.Tensor,
                                num_classes: int, metadata: keras.KerasTensor | None = None,
                                pooler: Mapping[str, Any] | None = None,
                                image_projection: Mapping[str, Any] | None = None,
                                metadata_encoder: Mapping[str, Any] | None = None,
                                fusion: Mapping[str, Any] | None = None,
                                classifier_head: Mapping[str, Any] | None = None,
                                name: str = "composable_classifier") -> keras.Model:
    """Assemble a complete classifier around an already-built encoder map."""
    pooled = build_pooler(feature_map, **dict(pooler or {}))
    image_embedding = build_image_projection(pooled, **dict(image_projection or {"dimension": int(pooled.shape[-1])}))
    metadata_embedding = build_metadata_encoder(metadata, **dict(metadata_encoder or {})) if metadata is not None else None
    fused_embedding = build_fusion(image_embedding, metadata_embedding, **dict(fusion or {}))
    # Projection is an explicit stage even when no additional transform is wanted.
    projection_embedding = layers.Activation("linear", name="projection_embedding")(fused_embedding)
    logits = build_classifier_head(projection_embedding, num_classes, **dict(classifier_head or {}))
    inputs: keras.KerasTensor | list[keras.KerasTensor] = [image] + ([metadata] if metadata is not None else [])
    return build_named_output_model(
        inputs, feature_map=feature_map, image_embedding=image_embedding,
        metadata_embedding=metadata_embedding, fused_embedding=fused_embedding,
        projection_embedding=projection_embedding, logits=logits, name=name,
    )
