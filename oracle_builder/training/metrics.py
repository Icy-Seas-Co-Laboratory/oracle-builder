from __future__ import annotations

import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package="oracle_builder")
class BinaryDice(keras.metrics.Metric):
    def __init__(self, threshold: float = 0.5, name: str = "dice", **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = float(threshold)
        self.true_positives = self.add_weight(name="true_positives", initializer="zeros")
        self.false_positives = self.add_weight(name="false_positives", initializer="zeros")
        self.false_negatives = self.add_weight(name="false_negatives", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        true_mask = tf.cast(y_true > 0.5, self.dtype)
        pred_mask = tf.cast(y_pred >= self.threshold, self.dtype)
        if sample_weight is not None:
            weight = tf.cast(sample_weight, self.dtype)
            if weight.shape.rank == true_mask.shape.rank - 1:
                weight = weight[..., None]
        else:
            weight = tf.cast(1.0, self.dtype)
        self.true_positives.assign_add(tf.reduce_sum(true_mask * pred_mask * weight))
        self.false_positives.assign_add(tf.reduce_sum((1.0 - true_mask) * pred_mask * weight))
        self.false_negatives.assign_add(tf.reduce_sum(true_mask * (1.0 - pred_mask) * weight))

    def result(self):
        denominator = 2.0 * self.true_positives + self.false_positives + self.false_negatives
        return tf.where(denominator > 0, 2.0 * self.true_positives / denominator, 1.0)

    def reset_state(self):
        for variable in self.variables:
            variable.assign(0)

    def get_config(self):
        return {**super().get_config(), "threshold": self.threshold}
