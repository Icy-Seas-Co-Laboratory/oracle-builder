from __future__ import annotations

import json
import copy
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from oracle_builder.classification.features import build_embedding_model
from oracle_builder.data.decoders import decode_blob
from oracle_builder.data.sqlite_dataset import resize_array_to_shape
from oracle_builder.registry import get_model_builder
from oracle_builder.training.augmentation import augment_batch


DEFAULT_VIEW_AUGMENTATION = {
    "enabled": True,
    "rotation": 0.08,
    "zoom": 0.15,
    "translation": 0.1,
    "skew": 0.05,
    "flip_horizontal": True,
    "flip_vertical": False,
    "brightness": 0.2,
    "contrast": 0.2,
    "gaussian_noise": 0.03,
    "fill_value": 0.0,
}


def make_pretraining_dataset(x: np.ndarray, config: dict[str, Any]):
    batch_size = int(config["data"].get("batch_size", 16))
    buffer_size = int(config["data"].get("shuffle_buffer", 512))
    seed = int(config["run"].get("seed", 123))
    return (
        tf.data.Dataset.from_tensor_slices(np.asarray(x, dtype="float32"))
        .shuffle(buffer_size, seed=seed)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def load_grayscale_reconstruction_dataset(database: str | Path, config: dict[str, Any]):
    """Load unlabeled grayscale images from either supported dataset type."""
    target_shape = tuple(int(value) for value in config["data"]["input_shape"][:2])
    with sqlite3.connect(database) as connection:
        kind = connection.execute("SELECT dataset_type FROM dataset WHERE singleton = 1").fetchone()[0]
        table, key = ("classification_items", "image_asset_id") if kind == "classification" else ("mask_refinement_items", "image_asset_id")
        rows = connection.execute(
            f"SELECT a.payload, a.encoding, a.shape_json FROM {table} i JOIN assets a ON a.asset_id=i.{key} ORDER BY i.item_id"
        ).fetchall()
    values = []
    for payload, encoding, shape in rows:
        image = np.asarray(decode_blob(payload, encoding, shape))
        if image.ndim == 3:
            image = image[..., 0] if image.shape[-1] == 1 else np.mean(image[..., :3], axis=-1)
        image = resize_array_to_shape(image, target_shape, mask=False).astype("float32")
        if image.max(initial=0) > 1:
            image /= float(np.iinfo(np.asarray(image).dtype).max) if np.asarray(image).dtype.kind in "ui" else float(image.max())
        values.append(image[..., None])
    if not values:
        raise ValueError("Pretraining database contains no image items")
    x = np.stack(values).astype("float32")
    return tf.data.Dataset.from_tensor_slices((x, x)).shuffle(len(x), seed=int(config["run"].get("seed", 123))).batch(int(config["pretraining"].get("batch_size", config["data"].get("batch_size", 16)))).prefetch(tf.data.AUTOTUNE)


def run_grayscale_reconstruction_pretraining(model, dataset, config, run_dir, strategy=None):
    """Pretrain the matching U-Net family as a 1-channel reconstruction model."""
    strategy = strategy or tf.distribute.get_strategy()
    pre_config = copy.deepcopy(config)
    pre_config["run"]["task"] = "segmentation"
    pre_config["data"]["input_shape"] = [*config["data"]["input_shape"][:2], 1]
    pre_config["data"]["output_shape"] = [*config["data"]["input_shape"][:2], 1]
    pre_config["data"]["candidate_sdf"] = False
    pre_config["data"]["candidate_distance"] = "none"
    with strategy.scope():
        pre_model = get_model_builder(config["run"]["model"])(pre_config)
        pre_model.compile(optimizer=keras.optimizers.Adam(float(config["pretraining"].get("learning_rate", 0.001))), loss="mse")
    history = pre_model.fit(dataset, epochs=int(config["pretraining"].get("epochs", 10)), verbose=2 if config.get("debug") else 1)
    source_convs = [layer for layer in pre_model.layers if isinstance(layer, layers.Conv2D)]
    target_convs = [layer for layer in model.layers if isinstance(layer, layers.Conv2D)]
    # Leave the reconstruction and segmentation heads independently initialized.
    for source, target in zip(source_convs[:-1], target_convs[:-1], strict=False):
        source_weights = source.get_weights()
        target_weights = target.get_weights()
        if not source_weights or not target_weights:
            continue
        if source_weights[0].shape == target_weights[0].shape:
            target.set_weights(source_weights)
        elif source_weights[0].shape[:-2] == target_weights[0].shape[:-2] and source_weights[0].shape[-2] == 1 and source_weights[0].shape[-2] < target_weights[0].shape[-2] and source_weights[0].shape[-1] == target_weights[0].shape[-1]:
            kernel = np.zeros_like(target_weights[0])
            kernel[..., 0, :] = source_weights[0][..., 0, :]
            target.set_weights([kernel, source_weights[1]])
    from oracle_builder.artifacts.layout import RunLayout
    layout = RunLayout(run_dir); layout.pretraining_metrics.mkdir(parents=True, exist_ok=True); layout.pretraining_model.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history.history).to_csv(layout.pretraining_metrics / "metrics.csv", index_label="epoch")
    (layout.pretraining_metrics / "metrics.json").write_text(json.dumps(history.history, indent=2, default=float) + "\n")
    pre_model.save_weights(layout.pretraining_model / "grayscale_reconstruction.weights.h5")
    return history


def _view_config(config: dict[str, Any]) -> dict[str, Any]:
    pretraining = config.get("pretraining", {})
    view_augmentation = dict(DEFAULT_VIEW_AUGMENTATION)
    if pretraining.get("use_training_augmentation", False):
        training_augmentation = dict(config.get("augmentation", {}))
        training_augmentation.pop("repeats_per_epoch", None)
        training_augmentation.pop("enabled", None)
        view_augmentation.update(training_augmentation)
    view_augmentation.update(pretraining.get("augmentation", {}))
    view_augmentation["enabled"] = True
    return {
        "run": {**config["run"], "task": "classification"},
        "augmentation": view_augmentation,
    }


def _projection_head(input_dim: int, hidden_dim: int, output_dim: int, name: str):
    return keras.Sequential(
        [
            keras.Input(shape=(input_dim,)),
            layers.Dense(hidden_dim, use_bias=False),
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.Dense(output_dim),
        ],
        name=name,
    )


class StudentTeacherPretrainer(keras.Model):
    """BYOL-style student predictor trained against an EMA teacher."""

    def __init__(self, classifier: keras.Model, config: dict[str, Any]):
        super().__init__(name="student_teacher_pretrainer")
        settings = config.get("pretraining", {})
        self.view_config = _view_config(config)
        self.momentum = float(settings.get("teacher_momentum", 0.99))
        embedding_dim = int(classifier.get_layer("features").output.shape[-1])
        projection_dim = int(settings.get("projection_dim", 128))
        hidden_dim = int(settings.get("projection_hidden_dim", max(embedding_dim, 256)))

        self.student_encoder = build_embedding_model(classifier)
        self.teacher_encoder = keras.models.clone_model(self.student_encoder)
        self.teacher_encoder.set_weights(self.student_encoder.get_weights())
        self.teacher_encoder.trainable = False
        self.student_projector = _projection_head(
            embedding_dim, hidden_dim, projection_dim, "student_projector"
        )
        self.teacher_projector = _projection_head(
            embedding_dim, hidden_dim, projection_dim, "teacher_projector"
        )
        self.teacher_projector.set_weights(self.student_projector.get_weights())
        self.teacher_projector.trainable = False
        self.predictor = _projection_head(
            projection_dim, hidden_dim, projection_dim, "student_predictor"
        )
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.cosine_similarity_tracker = keras.metrics.Mean(name="cosine_similarity")

    @property
    def metrics(self):
        return [self.loss_tracker, self.cosine_similarity_tracker]

    def call(self, inputs, training=False):
        """Expose the student path so Keras can build this subclassed model."""
        embeddings = self.student_encoder(inputs, training=training)
        projected = self.student_projector(embeddings, training=training)
        return self.predictor(projected, training=training)

    def _augment(self, x):
        dummy = tf.zeros([tf.shape(x)[0]], dtype=tf.int32)
        augmented, _ = augment_batch(x, dummy, self.view_config)
        return augmented

    @staticmethod
    def _cosine_loss(prediction, target):
        prediction = tf.math.l2_normalize(prediction, axis=-1)
        target = tf.math.l2_normalize(tf.stop_gradient(target), axis=-1)
        return 2.0 - 2.0 * tf.reduce_mean(tf.reduce_sum(prediction * target, axis=-1))

    def train_step(self, data):
        x = data[0] if isinstance(data, (tuple, list)) else data
        view_one = self._augment(tf.cast(x, tf.float32))
        view_two = self._augment(tf.cast(x, tf.float32))
        with tf.GradientTape() as tape:
            student_one = self.predictor(
                self.student_projector(self.student_encoder(view_one, training=True), training=True),
                training=True,
            )
            student_two = self.predictor(
                self.student_projector(self.student_encoder(view_two, training=True), training=True),
                training=True,
            )
            teacher_one = self.teacher_projector(
                self.teacher_encoder(view_one, training=False), training=False
            )
            teacher_two = self.teacher_projector(
                self.teacher_encoder(view_two, training=False), training=False
            )
            loss = 0.5 * (
                self._cosine_loss(student_one, teacher_two)
                + self._cosine_loss(student_two, teacher_one)
            )
            optimization_loss = loss / tf.cast(
                tf.distribute.get_strategy().num_replicas_in_sync,
                loss.dtype,
            )
        gradients = tape.gradient(optimization_loss, self.trainable_variables)
        self.optimizer.apply_gradients(
            (gradient, variable)
            for gradient, variable in zip(gradients, self.trainable_variables, strict=False)
            if gradient is not None
        )
        self._update_teacher()
        cosine_similarity = 1.0 - loss / 2.0
        self.loss_tracker.update_state(loss)
        self.cosine_similarity_tracker.update_state(cosine_similarity)
        return {metric.name: metric.result() for metric in self.metrics}

    def _update_teacher(self):
        for teacher, student in zip(
            self.teacher_encoder.weights,
            self.student_encoder.weights,
            strict=True,
        ):
            teacher.assign(self.momentum * teacher + (1.0 - self.momentum) * student)
        for teacher, student in zip(
            self.teacher_projector.weights,
            self.student_projector.weights,
            strict=True,
        ):
            teacher.assign(self.momentum * teacher + (1.0 - self.momentum) * student)


class SimCLRPretrainer(keras.Model):
    """SimCLR encoder trained with two augmented views and NT-Xent."""

    def __init__(self, classifier: keras.Model, config: dict[str, Any]):
        super().__init__(name="simclr_pretrainer")
        settings = config.get("pretraining", {})
        self.view_config = _view_config(config)
        self.temperature = float(settings.get("temperature", 0.1))
        embedding_dim = int(classifier.get_layer("features").output.shape[-1])
        projection_dim = int(settings.get("projection_dim", 128))
        hidden_dim = int(
            settings.get("projection_hidden_dim", max(embedding_dim, 256))
        )
        self.encoder = build_embedding_model(classifier)
        self.projector = _projection_head(
            embedding_dim,
            hidden_dim,
            projection_dim,
            "simclr_projector",
        )
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.contrastive_accuracy = keras.metrics.Mean(
            name="contrastive_accuracy"
        )

    @property
    def metrics(self):
        return [self.loss_tracker, self.contrastive_accuracy]

    def call(self, inputs, training=False):
        """Expose encoder/projector inference so Keras can build before fit()."""
        embeddings = self.encoder(inputs, training=training)
        return self.projector(embeddings, training=training)

    def _augment(self, x):
        dummy = tf.zeros([tf.shape(x)[0]], dtype=tf.int32)
        augmented, _ = augment_batch(x, dummy, self.view_config)
        return augmented

    def _nt_xent(self, first, second):
        first = tf.math.l2_normalize(first, axis=-1)
        second = tf.math.l2_normalize(second, axis=-1)
        batch_size = tf.shape(first)[0]
        representations = tf.concat([first, second], axis=0)
        logits = tf.matmul(
            representations, representations, transpose_b=True
        ) / tf.cast(self.temperature, representations.dtype)
        count = 2 * batch_size
        logits = logits - tf.eye(
            count, dtype=logits.dtype
        ) * tf.cast(1e9, logits.dtype)
        positives = tf.concat(
            [
                tf.range(batch_size, count),
                tf.range(0, batch_size),
            ],
            axis=0,
        )
        loss = tf.reduce_mean(
            keras.losses.sparse_categorical_crossentropy(
                positives, logits, from_logits=True
            )
        )
        accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(logits, axis=1, output_type=tf.int32), positives),
                tf.float32,
            )
        )
        return loss, accuracy

    def train_step(self, data):
        x = data[0] if isinstance(data, (tuple, list)) else data
        view_one = self._augment(tf.cast(x, tf.float32))
        view_two = self._augment(tf.cast(x, tf.float32))
        with tf.GradientTape() as tape:
            first = self.projector(
                self.encoder(view_one, training=True), training=True
            )
            second = self.projector(
                self.encoder(view_two, training=True), training=True
            )
            loss, accuracy = self._nt_xent(first, second)
            optimization_loss = loss / tf.cast(
                tf.distribute.get_strategy().num_replicas_in_sync,
                loss.dtype,
            )
        variables = self.encoder.trainable_variables + self.projector.trainable_variables
        gradients = tape.gradient(optimization_loss, variables)
        self.optimizer.apply_gradients(
            (gradient, variable)
            for gradient, variable in zip(gradients, variables, strict=True)
            if gradient is not None
        )
        self.loss_tracker.update_state(loss)
        self.contrastive_accuracy.update_state(accuracy)
        return {metric.name: metric.result() for metric in self.metrics}


def run_student_teacher_pretraining(
    classifier: keras.Model,
    dataset,
    config: dict[str, Any],
    run_dir: str | Path,
    strategy: tf.distribute.Strategy | None = None,
):
    settings = config.get("pretraining", {})
    method = str(settings.get("method", "byol")).lower()
    if method not in {"byol", "student_teacher", "simclr"}:
        raise ValueError(
            "pretraining.method must be byol, student_teacher, or simclr"
        )
    strategy = strategy or tf.distribute.get_strategy()
    with strategy.scope():
        pretrainer = (
            SimCLRPretrainer(classifier, config)
            if method == "simclr"
            else StudentTeacherPretrainer(classifier, config)
        )
        pretrainer.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=float(settings.get("learning_rate", 0.001))
            )
        )
        # Keras 3 may try a symbolic build before the first custom train_step.
        # Calling the explicit inference path here creates every wrapper variable
        # for all backbones, including EfficientNet.
        pretrainer(
            tf.zeros([1, *config["data"]["input_shape"]], dtype=tf.float32),
            training=False,
        )
    history = pretrainer.fit(
        dataset,
        epochs=int(settings.get("epochs", 10)),
        verbose=2 if config.get("debug") else 1,
    )
    from oracle_builder.artifacts.layout import RunLayout

    layout = RunLayout(run_dir)
    metrics_dir = layout.pretraining_metrics
    model_dir = layout.pretraining_model
    metrics_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history.history).to_csv(
        metrics_dir / "metrics.csv", index_label="epoch"
    )
    (metrics_dir / "metrics.json").write_text(
        json.dumps(history.history, indent=2, default=float) + "\n"
    )
    classifier.save_weights(model_dir / "student_pretrained.weights.h5")
    return history
