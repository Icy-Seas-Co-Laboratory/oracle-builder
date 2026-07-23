from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras

from oracle_builder.training.losses import BinaryCrossentropySoftDice


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
