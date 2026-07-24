from __future__ import annotations

import numpy as np
import tensorflow as tf

from oracle_builder.training.augmentation import (
    augment_batch,
    input_mask_channels,
    photometric_channels,
    signed_distance_input_channels,
    transform_input_channels,
)


def test_segmentation_defaults_protect_candidate_mask_channel_from_photometric_changes():
    config = {
        "run": {"task": "segmentation"},
        "data": {"input_shape": [4, 4, 2]},
        "augmentation": {"enabled": True, "invert": True},
    }
    x = np.zeros((1, 4, 4, 2), dtype="float32")
    x[..., 0] = 0.25
    x[..., 1] = 1.0
    y = np.zeros((1, 4, 4, 1), dtype="float32")

    augmented_x, augmented_y = augment_batch(tf.constant(x), tf.constant(y), config)

    assert np.allclose(augmented_x.numpy()[..., 0], 0.75)
    assert np.allclose(augmented_x.numpy()[..., 1], 1.0)
    assert np.array_equal(augmented_y.numpy(), y)
    assert photometric_channels(config, config["augmentation"]) == [0]
    assert input_mask_channels(config, config["augmentation"]) == [1]


def test_segmentation_geometric_augmentation_preserves_shapes_and_binary_masks():
    config = {
        "run": {"task": "segmentation"},
        "data": {"input_shape": [8, 8, 2]},
        "augmentation": {
            "enabled": True,
            "rotation": 0.05,
            "zoom": 0.10,
            "translation": [0.10, 0.10],
            "skew": 0.05,
            "mask_input_channels": [1],
            "photometric_channels": [0],
        },
    }
    x = np.zeros((2, 8, 8, 2), dtype="float32")
    x[:, 2:6, 2:6, 0] = 0.5
    x[:, 3:5, 3:5, 1] = 1.0
    y = np.zeros((2, 8, 8, 1), dtype="float32")
    y[:, 2:6, 2:6, 0] = 1.0

    augmented_x, augmented_y = augment_batch(tf.constant(x), tf.constant(y), config)

    assert augmented_x.shape == x.shape
    assert augmented_y.shape == y.shape
    assert set(np.unique(augmented_x.numpy()[..., 1]).tolist()).issubset({0.0, 1.0})
    assert set(np.unique(augmented_y.numpy()).tolist()).issubset({0.0, 1.0})


def test_classification_augmentation_keeps_label_dtype_and_shape():
    config = {
        "run": {"task": "classification"},
        "data": {"input_shape": [4, 4, 3]},
        "augmentation": {"enabled": True, "invert": True, "brightness": 0.0},
    }
    x = np.ones((3, 4, 4, 3), dtype="float32") * 0.2
    y = np.array([0, 1, 2], dtype="int64")

    augmented_x, augmented_y = augment_batch(tf.constant(x), tf.constant(y), config)

    assert augmented_x.shape == x.shape
    assert augmented_y.dtype == tf.int64
    assert np.array_equal(augmented_y.numpy(), y)
    assert np.allclose(augmented_x.numpy(), 0.8)


def test_segmentation_augmentation_transforms_spatial_weights_with_masks():
    config = {
        "run": {"task": "segmentation"},
        "data": {"input_shape": [8, 8, 1]},
        "augmentation": {"enabled": True, "translation": [0.1, 0.1]},
    }
    x = np.zeros((1, 8, 8, 1), dtype="float32")
    y = np.zeros((1, 8, 8, 1), dtype="float32")
    y[:, 2:6, 2:6] = 1.0
    weights = np.ones((1, 8, 8), dtype="float32")
    weights[:, 2:6, 2:6] = 3.0

    augmented_x, augmented_y, augmented_weights = augment_batch(
        tf.constant(x), tf.constant(y), config, tf.constant(weights)
    )

    assert augmented_x.shape == x.shape
    assert augmented_y.shape == y.shape
    assert augmented_weights.shape == weights.shape
    assert float(tf.reduce_min(augmented_weights)) >= 1.0


def test_candidate_sdf_channel_is_excluded_from_photometric_augmentation():
    config = {
        "run": {"task": "segmentation"},
        "data": {"input_shape": [4, 4, 3], "candidate_sdf": True},
        "augmentation": {"enabled": True, "invert": True},
    }
    x = np.zeros((1, 4, 4, 3), dtype="float32")
    x[..., 0] = 0.25
    x[..., 1] = 1.0
    x[..., 2] = np.linspace(-1, 1, 16).reshape(4, 4)
    y = np.zeros((1, 4, 4, 1), dtype="float32")

    augmented_x, _augmented_y = augment_batch(tf.constant(x), tf.constant(y), config)

    assert np.allclose(augmented_x.numpy()[..., 0], 0.75)
    assert np.array_equal(augmented_x.numpy()[..., 1], x[..., 1])
    assert np.allclose(augmented_x.numpy()[..., 2], x[..., 2])
    assert photometric_channels(config, config["augmentation"]) == [0]
    assert input_mask_channels(config, config["augmentation"]) == [1]


def test_candidate_sdf_augmentation_uses_continuous_field_and_negative_outside_fill():
    config = {
        "run": {"task": "segmentation"},
        "data": {"input_shape": [4, 4, 3], "candidate_sdf": True},
        "augmentation": {},
    }
    x = np.zeros((1, 4, 4, 3), dtype="float32")
    x[..., 1] = 1.0
    x[..., 2] = 0.5
    transforms = tf.constant([[1.0, 0.0, 10.0, 0.0, 1.0, 10.0, 0.0, 0.0]])

    transformed = transform_input_channels(
        tf.constant(x),
        transforms,
        fill_value=0.0,
        mask_channels=input_mask_channels(config, config["augmentation"]),
        distance_channels=signed_distance_input_channels(config, config["augmentation"]),
    ).numpy()

    assert np.all(transformed[..., 0] == 0.0)
    assert np.all(transformed[..., 1] == 0.0)
    assert np.all(transformed[..., 2] == -1.0)
    assert photometric_channels(config, config["augmentation"]) == [0]
