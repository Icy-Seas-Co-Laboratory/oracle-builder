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


@keras.utils.register_keras_serializable(package="oracle_builder")
class SparseCategoricalMacroF1(keras.metrics.Metric):
    """Macro F1 for exclusive classification, accumulated across an epoch.

    The confusion matrix is retained across batches so this is not an average
    of batch-level F1 values. Classes with no true or predicted samples receive
    an F1 of zero, matching the evaluation report's ``zero_division=0`` policy.
    """

    def __init__(self, num_classes: int, name: str = "macro_f1", **kwargs):
        super().__init__(name=name, **kwargs)
        if int(num_classes) < 1:
            raise ValueError("num_classes must be positive")
        self.num_classes = int(num_classes)
        self.confusion = self.add_weight(
            name="confusion",
            shape=(self.num_classes, self.num_classes),
            initializer="zeros",
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        targets = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        predictions = tf.cast(tf.reshape(tf.argmax(y_pred, axis=-1), [-1]), tf.int32)
        weights = None
        if sample_weight is not None:
            weights = tf.cast(tf.reshape(sample_weight, [-1]), self.dtype)
        matrix = tf.math.confusion_matrix(
            targets,
            predictions,
            num_classes=self.num_classes,
            weights=weights,
            dtype=self.dtype,
        )
        self.confusion.assign_add(matrix)

    def result(self):
        true_positives = tf.linalg.diag_part(self.confusion)
        predicted_totals = tf.reduce_sum(self.confusion, axis=0)
        actual_totals = tf.reduce_sum(self.confusion, axis=1)
        denominator = actual_totals + predicted_totals
        f1_by_class = tf.math.divide_no_nan(2.0 * true_positives, denominator)
        return tf.reduce_mean(f1_by_class)

    def reset_state(self):
        self.confusion.assign(tf.zeros_like(self.confusion))

    def get_config(self):
        return {**super().get_config(), "num_classes": self.num_classes}
