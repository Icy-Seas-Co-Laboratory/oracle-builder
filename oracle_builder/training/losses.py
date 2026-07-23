from __future__ import annotations

import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package="oracle_builder")
class BinaryCrossentropySoftDice(keras.losses.Loss):
    """Pixelwise BCE plus a per-image soft Dice loss broadcast over the image."""

    def __init__(
        self,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1e-6,
        name: str = "bce_soft_dice",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        y_pred = tf.cast(y_pred, y_pred.dtype)
        bce = keras.losses.binary_crossentropy(y_true, y_pred)
        axes = tf.range(1, tf.rank(y_true))
        intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
        denominator = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred, axis=axes)
        soft_dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1.0 - soft_dice
        dice_loss = tf.reshape(dice_loss, [-1, 1, 1])
        return self.bce_weight * bce + self.dice_weight * dice_loss

    def get_config(self):
        return {
            **super().get_config(),
            "bce_weight": self.bce_weight,
            "dice_weight": self.dice_weight,
            "smooth": self.smooth,
        }
