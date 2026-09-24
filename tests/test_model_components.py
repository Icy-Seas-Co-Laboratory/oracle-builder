"""Focused tests for the v2 composable model representation contract."""

import numpy as np
import tensorflow as tf
from tensorflow import keras

from oracle_builder.models.components import (
    OUTPUT_NAMES, GeMPooling2D, PrototypeClassifier, build_composable_classifier, build_pooler,
)


def test_poolers_have_expected_widths():
    features = keras.Input((4, 4, 3))
    assert build_pooler(features, "avg").shape[-1] == 3
    assert build_pooler(features, "max").shape[-1] == 3
    assert build_pooler(features, "avg_max").shape[-1] == 6
    assert build_pooler(features, "gem").shape[-1] == 3


def test_composable_classifier_exposes_stable_named_stages_and_serializes(tmp_path):
    image = keras.Input((8, 8, 1), name="image")
    metadata = keras.Input((2,), name="metadata")
    feature_map = keras.layers.Conv2D(4, 3, padding="same", name="encoder_map")(image)
    model = build_composable_classifier(
        image=image, feature_map=feature_map, metadata=metadata, num_classes=3,
        pooler={"kind": "avg_max"}, image_projection={"dimension": 5},
        metadata_encoder={"kind": "mlp", "hidden_units": [4], "dimension": 3},
        fusion={"kind": "projected", "dimension": 6},
        classifier_head={"kind": "cosine", "cosine_scale": 8.0},
    )
    values = model({"image": np.ones((2, 8, 8, 1)), "metadata": np.ones((2, 2))})
    assert set(values) == set(OUTPUT_NAMES)
    assert values["feature_map"].shape == (2, 8, 8, 4)
    assert values["image_embedding"].shape == (2, 5)
    assert values["metadata_embedding"].shape == (2, 3)
    assert values["projection_embedding"].shape == (2, 6)
    np.testing.assert_allclose(tf.reduce_sum(values["probabilities"], axis=1), [1, 1])

    path = tmp_path / "composable.keras"
    model.save(path)
    restored = keras.models.load_model(path)
    restored_values = restored({"image": np.ones((1, 8, 8, 1)), "metadata": np.ones((1, 2))})
    assert set(restored_values) == set(OUTPUT_NAMES)


def test_gem_is_serializable_and_trainable_exponent():
    layer = GeMPooling2D(p=3.0, trainable_p=True)
    assert layer.get_config()["trainable_p"] is True
    result = layer(tf.ones((2, 3, 3, 2)))
    np.testing.assert_allclose(result, np.ones((2, 2)), rtol=1e-5)


def test_prototype_classifier_supports_multiple_prototypes_and_serialization(tmp_path):
    layer = PrototypeClassifier(3, prototypes_per_class=2, metric="cosine", scale=8.0)
    logits = layer(tf.ones((2, 4), dtype=tf.float32))
    assert logits.shape == (2, 3)
    assert layer.prototypes.shape == (3, 2, 4)

    inputs = keras.Input((4,))
    model = keras.Model(inputs, layer(inputs))
    path = tmp_path / "prototype.keras"
    model.save(path)
    restored = keras.models.load_model(path)
    assert restored(np.ones((1, 4), dtype="float32")).shape == (1, 3)
