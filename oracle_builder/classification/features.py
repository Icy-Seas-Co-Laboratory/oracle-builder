from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


FEATURE_LAYER_NAME = "features"
LOGITS_LAYER_NAME = "logits"
DEFAULT_EMBEDDING_DIM = 256
STRATUM_DIMENSION_INPUT_NAME = "stratum_dimension"

# V2 representation-contract layer names.  ``features`` remains the legacy
# serving alias so existing bundles and downstream callers continue to load.
FEATURE_MAP_LAYER_NAME = "feature_map"
IMAGE_EMBEDDING_LAYER_NAME = "image_embedding"
METADATA_EMBEDDING_LAYER_NAME = "metadata_embedding"
FUSED_EMBEDDING_LAYER_NAME = "fused_embedding"
PROJECTION_EMBEDDING_LAYER_NAME = "projection_embedding"


def classifier_normalization(config: dict[str, Any], channels: int, name: str):
    """Return a V2 normalization layer with legacy stratification fallback."""
    component = config.get("normalization", {})
    settings = config.get("classification", {}).get("stratification", {})
    mode = str(
        component.get("type") if isinstance(component, dict) and component.get("type") is not None
        else settings.get("normalization", "batch")
    ).lower()
    if mode == "group":
        requested_value = (
            component.get("groups", settings.get("group_norm_groups", 8))
            if isinstance(component, dict) else settings.get("group_norm_groups", 8)
        )
        requested = 8 if str(requested_value).lower() == "auto" else int(requested_value)
        if requested <= 0:
            requested = 8
        groups = max(
            group
            for group in range(min(requested, int(channels)), 0, -1)
            if int(channels) % group == 0
        )
        return layers.GroupNormalization(groups=groups, epsilon=1e-3, name=name)
    if mode in {"layer", "layernorm", "layer_norm"}:
        return layers.LayerNormalization(axis=-1, epsilon=1e-5, name=name)
    if mode in {"none", "identity", "off"}:
        return layers.Activation("linear", name=name)
    if mode != "batch":
        raise ValueError("normalization.type must be batch, group, layer, or none")
    return layers.BatchNormalization(momentum=0.99, epsilon=1e-3, name=name)


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


def classifier_inputs(input_shape: tuple[int, ...], config: dict[str, Any]):
    """Create the stable named inputs used by native image classifiers."""
    image = keras.Input(shape=input_shape, name="image")
    count = len(config.get("model", {}).get("auxiliary_features_fitted", config.get("model", {}).get("auxiliary_features", [])))
    metadata = keras.Input(shape=(count,), name="metadata") if count else None
    return image, metadata


@keras.utils.register_keras_serializable(package="oracle_builder")
class StratumDimensionIndex(layers.Layer):
    """Convert configured resolution values (for example 32) to embedding IDs."""

    def __init__(self, dimensions: list[int], **kwargs):
        super().__init__(**kwargs)
        self.dimensions = tuple(int(value) for value in dimensions)

    def call(self, inputs):
        values = tf.cast(tf.reshape(inputs, [-1, 1]), tf.int32)
        dimensions = tf.constant(self.dimensions, dtype=tf.int32)
        matches = tf.equal(values, dimensions[tf.newaxis, :])
        tf.debugging.assert_equal(
            tf.reduce_any(matches, axis=1),
            tf.ones(tf.shape(values)[0], dtype=tf.bool),
            message="stratum_dimension is not configured for this model",
        )
        return tf.argmax(tf.cast(matches, tf.int32), axis=1, output_type=tf.int32)

    def get_config(self):
        return {**super().get_config(), "dimensions": list(self.dimensions)}


def stratum_conditioning_input(config: dict[str, Any]):
    """Create the runtime resolution input for an explicitly conditioned bundle."""
    settings = config.get("classification", {}).get("stratification", {})
    conditioning = settings.get("conditioning", {}) if isinstance(settings, dict) else {}
    if not (settings.get("enabled", False) and conditioning.get("enabled", False)):
        return None
    return keras.Input(shape=(1,), dtype="int32", name=STRATUM_DIMENSION_INPUT_NAME)


def join_stratum_conditioning(x, stratum_dimension, config: dict[str, Any]):
    """Add a zero-initialized, resolution-specific residual adapter to ``x``.

    The adapter begins as an identity mapping. This makes enabling conditioning
    conservative: shared visual features remain the initial solution while each
    resolution can learn a small correction during mixed-resolution training.
    """
    if stratum_dimension is None:
        return x
    settings = config["classification"]["stratification"]
    conditioning = settings["conditioning"]
    dimensions = [int(value) for value in settings["dimensions"]]
    index = StratumDimensionIndex(dimensions, name="stratum_dimension_index")(stratum_dimension)
    embedding_dim = int(conditioning.get("embedding_dim", 16))
    embedding = layers.Embedding(
        len(dimensions), embedding_dim, name="stratum_embedding"
    )(index)
    combined = layers.Concatenate(name="stratum_adapter_inputs")([x, embedding])
    adapter = layers.Dense(
        int(x.shape[-1]), kernel_initializer="zeros", bias_initializer="zeros",
        name="stratum_adapter",
    )(combined)
    return layers.Add(name="stratum_conditioned_features")([x, adapter])


def join_auxiliary_features(x, metadata):
    """Join standardized scalar features after global image pooling."""
    if metadata is None:
        return x
    return layers.Concatenate(name="image_metadata_features")([x, metadata])


def classification_head(
    x,
    num_classes: int,
    config: dict[str, Any],
    *,
    dropout_default: float = 0.0,
    normalize_default: bool = True,
    stratum_dimension=None,
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

    x = join_stratum_conditioning(x, stratum_dimension, config)
    x = layers.Dense(embedding_dim, name="embedding_projection")(x)
    if normalize:
        features = L2Normalization(name=FEATURE_LAYER_NAME)(x)
    else:
        features = layers.Activation("linear", name=FEATURE_LAYER_NAME)(x)
    x = features
    if dropout:
        x = layers.Dropout(dropout, name="classifier_dropout")(x)
    # Keep the public ``logits`` tensor aligned with the probabilities.  A
    # conditioned model adds its per-stratum bias after the common classifier.
    logits = layers.Dense(
        num_classes,
        name="shared_logits" if stratum_dimension is not None else LOGITS_LAYER_NAME,
    )(x)
    if stratum_dimension is not None:
        settings = config["classification"]["stratification"]
        index = StratumDimensionIndex(
            [int(value) for value in settings["dimensions"]], name="stratum_bias_index"
        )(stratum_dimension)
        bias = layers.Embedding(
            len(settings["dimensions"]), num_classes,
            embeddings_initializer="zeros", name="stratum_classifier_bias",
        )(index)
        logits = layers.Add(name=LOGITS_LAYER_NAME)([logits, bias])
    return layers.Activation("softmax", name="predictions")(logits)


def uses_composable_graph(config: dict[str, Any]) -> bool:
    """Whether a model should use the V2 assembly graph.

    V1 configurations retain their exact graph for reproducible reloads. New
    resolved recipes default to ``[architecture] version = 2``; recording the
    version keeps the graph format visible in each sealed run configuration.
    """
    architecture = config.get("architecture", {})
    return isinstance(architecture, dict) and int(architecture.get("version", 1)) >= 2


def build_composable_classification_model(
    *,
    image,
    feature_map,
    metadata,
    num_classes: int,
    config: dict[str, Any],
    name: str,
    stratum_dimension=None,
):
    """Attach V2 pooler/embedding/fusion/head components to an encoder map.

    The training model retains its historical probability output.  The named
    intermediate layers are exposed through :func:`build_feature_model`, so a
    single model supports supervised fitting, SSL, inference, and caching.
    """
    from oracle_builder.models.components import (
        L2Normalize,
        build_classifier_head,
        build_fusion,
        build_image_projection,
        build_metadata_encoder,
        build_pooler,
    )

    map_value = layers.Activation("linear", name=FEATURE_MAP_LAYER_NAME)(feature_map)
    pooling = dict(config.get("pooling", {}))
    pooler = {
        "kind": pooling.get("type", "avg"),
        "gem_p": float(pooling.get("p", 3.0)),
        "gem_trainable_p": bool(pooling.get("trainable_p", False)),
        "name": "image_pooling",
    }
    pooled = build_pooler(map_value, **pooler)
    projection = config.get("image_embedding", {}).get("projection", {})
    if not isinstance(projection, dict):
        raise ValueError("image_embedding.projection must be a table/object")
    projection_kind = str(projection.get("type", "identity")).lower()
    if projection_kind not in {"identity", "linear", "mlp"}:
        raise ValueError("image_embedding.projection.type must be identity, linear, or mlp")
    pooled_width = int(pooled.shape[-1])
    image_dim = int(projection.get("output_dim", pooled_width))
    if projection_kind == "identity":
        if image_dim != pooled_width:
            raise ValueError("identity image projection cannot change embedding dimension")
        image_embedding = layers.Activation("linear", name=IMAGE_EMBEDDING_LAYER_NAME)(pooled)
        if bool(projection.get("l2_normalize", False)):
            image_embedding = L2Normalization(name=f"{IMAGE_EMBEDDING_LAYER_NAME}_l2")(image_embedding)
    elif projection_kind == "mlp":
        hidden_units = projection.get("hidden_units", [])
        if not isinstance(hidden_units, list) or any(int(value) < 1 for value in hidden_units):
            raise ValueError("image_embedding.projection.hidden_units must be positive integers")
        value = pooled
        for index, units in enumerate(hidden_units):
            value = layers.Dense(
                int(units), activation=projection.get("activation", "gelu"),
                name=f"{IMAGE_EMBEDDING_LAYER_NAME}_hidden_{index + 1}",
            )(value)
        image_embedding = build_image_projection(
            value, image_dim, dropout=float(projection.get("dropout", 0.0)),
            l2_normalize=bool(projection.get("l2_normalize", False)),
            name=IMAGE_EMBEDDING_LAYER_NAME,
        )
    else:
        image_embedding = build_image_projection(
            pooled,
            image_dim,
            dropout=float(projection.get("dropout", 0.0)),
            l2_normalize=bool(projection.get("l2_normalize", False)),
            name=IMAGE_EMBEDDING_LAYER_NAME,
        )
    metadata_embedding = None
    if metadata is not None:
        metadata_settings = config.get("metadata", {})
        encoder = metadata_settings.get("encoder", {}) if isinstance(metadata_settings, dict) else {}
        if not isinstance(encoder, dict):
            raise ValueError("metadata.encoder must be a table/object")
        metadata_embedding = build_metadata_encoder(
            metadata,
            kind=encoder.get("type", "direct"),
            dimension=encoder.get("output_dim"),
            hidden_units=encoder.get("hidden_units", ()),
            activation=encoder.get("activation", "relu"),
            dropout=float(encoder.get("dropout", 0.0)),
            name=METADATA_EMBEDDING_LAYER_NAME,
        )
    fusion_settings = dict(config.get("fusion", {}))
    fusion_kind = fusion_settings.get("type", "concat")
    fused_embedding = build_fusion(
        image_embedding,
        metadata_embedding,
        kind=fusion_kind,
        dimension=fusion_settings.get("output_dim"),
        activation=fusion_settings.get("activation"),
        name=FUSED_EMBEDDING_LAYER_NAME,
    )
    classifier_input = join_stratum_conditioning(
        fused_embedding, stratum_dimension, config
    )
    # Legacy aliases keep existing SSL/export clients operational while V2
    # consumers select the exact representation stage above.
    projected = layers.Activation("linear", name=PROJECTION_EMBEDDING_LAYER_NAME)(classifier_input)
    projected = layers.Activation("linear", name="embedding_projection")(projected)
    normalize = bool(projection.get("normalize", config.get("model", {}).get("normalize_embeddings", True)))
    features = L2Normalize(name=FEATURE_LAYER_NAME)(projected) if normalize else layers.Activation("linear", name=FEATURE_LAYER_NAME)(projected)
    head = config.get("classifier", {})
    if not isinstance(head, dict):
        raise ValueError("classifier must be a table/object")
    logits = build_classifier_head(
        features,
        num_classes,
        kind=head.get("type", "linear"),
        hidden_units=head.get("hidden_units", ()),
        activation=head.get("activation", "relu"),
        dropout=float(head.get("dropout", config.get("model", {}).get("dropout", 0.0))),
        cosine_scale=float(head.get("cosine_scale", 16.0)),
        prototypes_per_class=int(head.get("prototypes_per_class", 1)),
        prototype_metric=head.get("prototype_metric", "cosine"),
        name=LOGITS_LAYER_NAME,
    )
    probabilities = layers.Activation("softmax", name="predictions")(logits)
    model_inputs = [image] + ([metadata] if metadata is not None else []) + ([stratum_dimension] if stratum_dimension is not None else [])
    return keras.Model(model_inputs if len(model_inputs) > 1 else image, probabilities, name=name)


def _optional_layer_output(model: keras.Model, *names: str):
    """Return the first available named tensor without penalizing V1 models."""
    for name in names:
        try:
            return model.get_layer(name).output
        except ValueError:
            continue
    return None


def build_feature_model(model: keras.Model) -> keras.Model:
    """Return the versioned representation contract plus the legacy alias.

    V1 classifiers did not name intermediate representations.  Their adapter
    exposes the strongest faithful approximation: pooled image features,
    direct metadata input where present, the concatenated pre-projection
    tensor, and the historical normalized ``features`` output.  New assembled
    models provide exact named stages.  Optional stages are intentionally
    omitted from the Keras output mapping rather than populated with fake
    tensors.
    """
    features = model.get_layer(FEATURE_LAYER_NAME).output
    logits = model.get_layer(LOGITS_LAYER_NAME).output
    image_embedding = _optional_layer_output(
        model, IMAGE_EMBEDDING_LAYER_NAME, "global_pool", FEATURE_LAYER_NAME
    )
    fused_embedding = _optional_layer_output(
        model, FUSED_EMBEDDING_LAYER_NAME, "image_metadata_features"
    )
    if fused_embedding is None:
        fused_embedding = image_embedding
    projection_embedding = _optional_layer_output(
        model, PROJECTION_EMBEDDING_LAYER_NAME, "embedding_projection", FEATURE_LAYER_NAME
    )
    outputs = {
        "logits": logits,
        "probabilities": model.output,
        # Legacy stable key. It remains the normal downstream representation
        # until clients explicitly select a V2 source.
        "features": features,
        "image_embedding": image_embedding,
        "fused_embedding": fused_embedding,
        "projection_embedding": projection_embedding,
    }
    feature_map = _optional_layer_output(model, FEATURE_MAP_LAYER_NAME)
    if feature_map is not None:
        outputs["feature_map"] = feature_map
    metadata_embedding = _optional_layer_output(model, METADATA_EMBEDDING_LAYER_NAME)
    if metadata_embedding is None:
        metadata_embedding = _optional_layer_output(model, "metadata")
    if metadata_embedding is not None:
        outputs["metadata_embedding"] = metadata_embedding
    return keras.Model(
        model.input,
        outputs,
        name=f"{model.name}_with_features",
    )


def build_embedding_model(model: keras.Model) -> keras.Model:
    """Return the shared classifier backbone through its fixed-size features."""
    inputs = model.inputs if getattr(model, "inputs", None) and len(model.inputs) > 1 else model.inputs[0] if getattr(model, "inputs", None) else model.input
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
    inputs = model.inputs if getattr(model, "inputs", None) and len(model.inputs) > 1 else model.inputs[0] if getattr(model, "inputs", None) else model.input
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
    representation_keys = (
        "features",
        "feature_map",
        "image_embedding",
        "metadata_embedding",
        "fused_embedding",
        "projection_embedding",
    )

    def normalized(outputs: dict[str, Any], *, logits_source: str) -> dict[str, Any]:
        result: dict[str, Any] = {
            "logits": np.asarray(outputs["logits"]),
            "probabilities": np.asarray(outputs["probabilities"]),
            "logits_source": logits_source,
        }
        for key in representation_keys:
            value = outputs.get(key)
            result[key] = np.asarray(value) if value is not None else None
        return result

    if hasattr(model, "predict_outputs"):
        outputs = model.predict_outputs(x, verbose=0)
        return normalized(outputs, logits_source=outputs.get("logits_source", "model"))
    # Promoted external products expose the standard tensors as a Keras output
    # dictionary, avoiding a dependence on their original internal graph.
    predict_options = {"verbose": 0}
    if batch_size is not None:
        predict_options["batch_size"] = batch_size
    direct = model.predict(x, **predict_options)
    if isinstance(direct, dict) and {"logits", "probabilities"}.issubset(direct):
        return normalized(direct, logits_source="model")
    try:
        outputs = build_feature_model(model).predict(
            x, **predict_options
        )
        return normalized(outputs, logits_source="model")
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
        result = {
            "logits": logits.astype("float32"),
            "probabilities": probabilities,
            "features": features,
            "logits_source": "derived_log_probability",
        }
        for key in representation_keys:
            result.setdefault(key, None)
        return result
