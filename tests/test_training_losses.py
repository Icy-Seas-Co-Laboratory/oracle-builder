from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

from oracle_builder.training.losses import (
    BinaryCrossentropySoftDice,
    BinaryCrossentropySoftTversky,
    WeightedSparseCategoricalCrossentropy,
)
from oracle_builder.training.class_weights import resolve_class_weights
from oracle_builder.training.train import build_and_compile_model


def test_bce_soft_dice_is_near_zero_for_perfect_predictions():
    target = tf.constant([[[[1.0], [0.0]], [[0.0], [1.0]]]])
    prediction = tf.constant([[[[1.0], [0.0]], [[0.0], [1.0]]]])

    loss = BinaryCrossentropySoftDice()(target, prediction)

    assert float(loss) < 1e-5


def test_bce_soft_dice_penalizes_incorrect_predictions_more():
    target = tf.constant([[[[1.0], [1.0]], [[0.0], [0.0]]]])
    good = tf.constant([[[[0.9], [0.8]], [[0.2], [0.1]]]])
    bad = 1.0 - good
    loss = BinaryCrossentropySoftDice()

    assert float(loss(target, good)) < float(loss(target, bad))


def test_tversky_with_higher_false_negative_weight_favors_expansion():
    target = tf.constant([[[[1.0], [1.0], [0.0], [0.0]]]])
    under = tf.constant([[[[0.5], [0.5], [0.0], [0.0]]]])
    over = tf.constant([[[[1.0], [1.0], [0.5], [0.5]]]])
    loss = BinaryCrossentropySoftTversky(bce_weight=0.0, alpha=0.3, beta=0.7)

    assert float(loss(target, over)) < float(loss(target, under))


def test_bce_soft_tversky_can_be_saved_and_reloaded(tmp_path):
    model = keras.Sequential([keras.layers.Input((4, 4, 1)), keras.layers.Conv2D(1, 1, activation="sigmoid")])
    model.compile(optimizer="adam", loss=BinaryCrossentropySoftTversky())
    path = tmp_path / "model.keras"
    model.save(path)

    assert isinstance(keras.models.load_model(path).loss, BinaryCrossentropySoftTversky)


def test_bce_soft_dice_returns_pixelwise_values_before_reduction():
    loss = BinaryCrossentropySoftDice(reduction=None)
    target = np.zeros((2, 4, 4, 1), dtype="float32")
    prediction = np.full_like(target, 0.25)

    values = loss(target, prediction)

    assert values.shape == (2, 4, 4)


def test_model_with_bce_soft_dice_can_be_saved_and_reloaded(tmp_path):
    model = keras.Sequential(
        [keras.layers.Input((4, 4, 1)), keras.layers.Conv2D(1, 1, activation="sigmoid")]
    )
    model.compile(optimizer="adam", loss=BinaryCrossentropySoftDice())
    path = tmp_path / "model.keras"
    model.save(path)

    loaded = keras.models.load_model(path)

    assert isinstance(loaded.loss, BinaryCrossentropySoftDice)


def test_weighted_cross_entropy_emphasizes_rare_class_errors():
    loss = WeightedSparseCategoricalCrossentropy([0.5, 2.0], reduction=None)
    targets = tf.constant([0, 1])
    predictions = tf.constant([[0.25, 0.75], [0.75, 0.25]])

    values = loss(targets, predictions).numpy()

    assert values[1] > values[0] * 3


def test_inverse_frequency_weights_have_unit_observed_mean():
    resolved = resolve_class_weights(
        [0, 0, 0, 1],
        2,
        {"mode": "inverse_frequency", "normalize": True},
    )

    assert resolved["counts"] == [3, 1]
    assert resolved["values"][1] > resolved["values"][0]
    observed = (
        3 * resolved["values"][0] + resolved["values"][1]
    ) / 4
    assert np.isclose(observed, 1.0)


def test_classification_model_trains_with_weighted_cross_entropy():
    config = {
        "run": {"task": "classification", "model": "simple_cnn"},
        "data": {"input_shape": [8, 8, 1], "num_classes": 2},
        "model": {"base_filters": 2, "dropout": 0.0, "embedding_dim": 4},
        "training": {
            "optimizer": "adam",
            "learning_rate": 0.001,
            "loss": "weighted_sparse_categorical_crossentropy",
            "class_weights": {"values": [0.5, 1.5]},
            "metrics": ["accuracy"],
        },
    }
    model = build_and_compile_model(config)

    values = model.train_on_batch(
        np.zeros((2, 8, 8, 1), dtype="float32"),
        np.array([0, 1], dtype="int64"),
    )

    assert np.all(np.isfinite(values))
