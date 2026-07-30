from __future__ import annotations

import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package="oracle_builder")
class WeightedSparseCategoricalCrossentropy(keras.losses.Loss):
    """Sparse categorical cross entropy weighted by the true class."""

    def __init__(
        self,
        class_weights,
        from_logits: bool = False,
        name: str = "weighted_sparse_categorical_crossentropy",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        values = [float(value) for value in class_weights]
        if not values or any(value <= 0 for value in values):
            raise ValueError("class_weights must contain positive values")
        self.class_weights = values
        self.from_logits = bool(from_logits)

    def call(self, y_true, y_pred):
        labels = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        losses = keras.losses.sparse_categorical_crossentropy(
            labels,
            y_pred,
            from_logits=self.from_logits,
        )
        weights = tf.gather(
            tf.convert_to_tensor(self.class_weights, dtype=losses.dtype),
            labels,
        )
        return losses * weights

    def get_config(self):
        return {
            **super().get_config(),
            "class_weights": self.class_weights,
            "from_logits": self.from_logits,
        }


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
