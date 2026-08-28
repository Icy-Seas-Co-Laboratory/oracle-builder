from __future__ import annotations

import json
import copy
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from oracle_builder.classification.features import build_self_supervised_embedding_model
from oracle_builder.config import self_supervised_settings
from oracle_builder.data.decoders import decode_blob
from oracle_builder.data.sqlite_dataset import resize_array_to_shape
from oracle_builder.registry import get_model_builder
from oracle_builder.training.augmentation import augment_batch
from oracle_builder.training.logging_callbacks import log_event, write_history_jsonl
from oracle_builder.training.status import RichTrainingStatusCallback


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


def _self_supervised_fit_verbose(settings: dict[str, Any]) -> int:
    """Return the Keras progress mode for self-supervised training."""
    value = settings.get("verbose", 1)
    if isinstance(value, bool) or value not in {0, 1, 2}:
        raise ValueError("self_supervised.verbose must be 0, 1, or 2")
    return int(value)


def _self_supervised_display(config: dict[str, Any], settings: dict[str, Any]) -> str:
    """Map the legacy SSL verbosity knob onto the shared display modes."""
    verbose = _self_supervised_fit_verbose(settings)
    if verbose == 0:
        return "off"
    if verbose == 2:
        return "text"
    return str(config.get("training", {}).get("display", "rich")).lower()


def make_self_supervised_dataset(x: np.ndarray, config: dict[str, Any]):
    batch_size = int(config["data"].get("batch_size", 16))
    buffer_size = int(config["data"].get("shuffle_buffer", 512))
    seed = int(config["run"].get("seed", 123))
    return (
        tf.data.Dataset.from_tensor_slices(np.asarray(x, dtype="float32"))
        .shuffle(buffer_size, seed=seed)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


# Legacy import compatibility.
make_pretraining_dataset = make_self_supervised_dataset


def load_grayscale_self_supervised_dataset(database: str | Path, config: dict[str, Any]):
    """Load unlabeled grayscale images from either supported dataset type."""
    database_path = Path(database).expanduser().resolve()
    if not database_path.is_file():
        raise FileNotFoundError(
            "Grayscale reconstruction self-supervised database does not exist: "
            f"{database_path}. Set self_supervised.database to an existing Dataset V1 "
            "classification or mask-refinement SQLite file."
        )
    target_shape = tuple(int(value) for value in config["data"]["input_shape"][:2])
    with sqlite3.connect(f"file:{database_path}?mode=ro", uri=True) as connection:
        try:
            row = connection.execute("SELECT dataset_type FROM dataset WHERE singleton = 1").fetchone()
        except sqlite3.OperationalError as exc:
            raise ValueError(
                f"Self-supervised database is not an Oracle Builder Dataset V1 file: {database_path}"
            ) from exc
        if row is None or row[0] not in {"classification", "mask_refinement"}:
            raise ValueError(
                f"Self-supervised database must be classification or mask_refinement Dataset V1: {database_path}"
            )
        kind = row[0]
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
        raise ValueError("Self-supervised database contains no image items")
    x = np.stack(values).astype("float32")
    settings = self_supervised_settings(config)
    return tf.data.Dataset.from_tensor_slices((x, x)).shuffle(len(x), seed=int(config["run"].get("seed", 123))).batch(int(settings.get("batch_size", config["data"].get("batch_size", 16)))).prefetch(tf.data.AUTOTUNE)


# Legacy import compatibility.
load_grayscale_reconstruction_dataset = load_grayscale_self_supervised_dataset


def grayscale_reconstruction_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a grayscale reconstruction configuration with a disposable linear head."""
    pre_config = copy.deepcopy(config)
    pre_config["run"]["task"] = "segmentation"
    pre_config["data"]["input_shape"] = [*config["data"]["input_shape"][:2], 1]
    pre_config["data"]["output_shape"] = [*config["data"]["input_shape"][:2], 1]
    pre_config["data"]["candidate_sdf"] = False
    pre_config["data"]["candidate_distance"] = "none"
    # This head is never transferred to segmentation. Keeping it linear prevents
    # the reconstruction loss from becoming gradient-free if sigmoid saturates.
    pre_config.setdefault("model", {})["final_activation"] = "linear"
    return pre_config


def foreground_weighted_reconstruction_mse(y_true, y_pred, foreground_weight: float = 4.0):
    """MSE that gives bright normalized image structure additional influence."""
    targets = tf.cast(y_true, y_pred.dtype)
    weight = 1.0 + (float(foreground_weight) - 1.0) * tf.clip_by_value(targets, 0.0, 1.0)
    return tf.reduce_mean(weight * tf.square(y_pred - targets), axis=[1, 2, 3])


def reconstruction_mse(y_true, y_pred):
    """Unweighted per-sample reconstruction MSE for pretraining diagnostics."""
    targets = tf.cast(y_true, y_pred.dtype)
    return tf.reduce_mean(tf.square(y_pred - targets), axis=[1, 2, 3])


def foreground_reconstruction_mse(y_true, y_pred):
    """MSE over bright structure, reported separately from background pixels."""
    targets = tf.cast(y_true, y_pred.dtype)
    foreground = tf.cast(targets > 0.1, y_pred.dtype)
    squared_error = tf.square(y_pred - targets)
    numerator = tf.reduce_sum(squared_error * foreground, axis=[1, 2, 3])
    denominator = tf.maximum(tf.reduce_sum(foreground, axis=[1, 2, 3]), 1.0)
    return numerator / denominator


def reconstruction_prediction_mean(_y_true, y_pred):
    return tf.reduce_mean(y_pred)


def reconstruction_prediction_std(_y_true, y_pred):
    return tf.math.reduce_std(y_pred)


class SelfSupervisedStatusCallback(keras.callbacks.Callback):
    """Emit structured status with one mutable progress line per epoch.

    Keras' built-in ``verbose=1`` progress bar is line-oriented when stdout is
    captured by a remote runner, so every batch update becomes a new log line.
    Keep Keras quiet and render the progress bar here: interactive terminals
    receive carriage-return updates, while captured/non-TTY logs receive only
    the epoch start and completion records. JSONL logging is unaffected.
    """

    def __init__(
        self,
        *,
        method: str,
        epochs: int,
        training_log: str | Path | None = None,
        run_id: str | None = None,
        verbose: int = 1,
        stream=None,
    ):
        super().__init__()
        self.method = method
        self.epochs = int(epochs)
        self.training_log = training_log
        self.run_id = run_id
        if isinstance(verbose, bool) or int(verbose) not in {0, 1, 2}:
            raise ValueError("self_supervised.verbose must be 0, 1, or 2")
        self.verbose = int(verbose)
        self.stream = stream if stream is not None else sys.stdout
        self._interactive = bool(
            self.verbose == 1
            and getattr(self.stream, "isatty", lambda: False)()
        )
        self._started_at: float | None = None
        self._epoch = 0
        self._steps = None
        self._last_progress_length = 0

    def _event(self, message: str, details: dict[str, Any]) -> None:
        if self.training_log is not None and self.run_id is not None:
            log_event(self.training_log, self.run_id, "INFO", message, details)

    def on_epoch_begin(self, epoch: int, logs=None):
        del logs
        self._started_at = time.perf_counter()
        self._epoch = int(epoch) + 1
        self._steps = self.params.get("steps")
        details = {
            "phase": "self_supervised",
            "method": self.method,
            "epoch": self._epoch,
            "epochs": self.epochs,
        }
        if self.verbose == 1:
            self._write(
                f"[self-supervised {self.method} {self._epoch}/{self.epochs}] started"
            )
        self._event("Self-supervised epoch started", details)

    @staticmethod
    def _metrics(logs) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for name, value in (logs or {}).items():
            try:
                metrics[str(name)] = float(value)
            except (TypeError, ValueError):
                continue
        return metrics

    def _write(self, message: str, *, final: bool = False) -> None:
        if self._interactive:
            padding = " " * max(0, self._last_progress_length - len(message))
            suffix = "\n" if final else ""
            self.stream.write(f"\r{message}{padding}{suffix}")
            self.stream.flush()
            self._last_progress_length = 0 if final else len(message)
        else:
            print(message, file=self.stream, flush=True)

    def on_train_batch_end(self, batch: int, logs=None):
        if not self._interactive:
            return
        current = int(batch) + 1
        total = self._steps if self._steps not in (None, -1) else "?"
        metrics = self._metrics(logs)
        rendered = " - ".join(
            f"{name}: {value:.4g}" for name, value in metrics.items()
        )
        suffix = f" - {rendered}" if rendered else ""
        self._write(
            f"[self-supervised {self.method} {self._epoch}/{self.epochs}] "
            f"{current}/{total}{suffix}"
        )

    def on_epoch_end(self, epoch: int, logs=None):
        metrics = self._metrics(logs)
        elapsed = time.perf_counter() - (self._started_at or time.perf_counter())
        details = {
            "phase": "self_supervised",
            "method": self.method,
            "epoch": int(epoch) + 1,
            "epochs": self.epochs,
            "elapsed_seconds": elapsed,
            "metrics": metrics,
        }
        rendered = ", ".join(f"{name}={value:.5g}" for name, value in metrics.items())
        suffix = f": {rendered}" if rendered else ""
        if self.verbose:
            self._write(
                f"[self-supervised {self.method} {epoch + 1}/{self.epochs}] "
                f"completed in {elapsed:.1f}s{suffix}",
                final=True,
            )
        self._event("Self-supervised epoch completed", details)

    def on_train_end(self, logs=None):
        """Terminate an in-progress interactive line on interruption/stop."""
        del logs
        if self._interactive and self._last_progress_length:
            self.stream.write("\n")
            self.stream.flush()
            self._last_progress_length = 0


# Legacy import compatibility for callers that used the old phase name.
PretrainingStatusCallback = SelfSupervisedStatusCallback


def run_grayscale_reconstruction_self_supervised(
    model,
    dataset,
    config,
    run_dir,
    strategy=None,
    *,
    training_log: str | Path | None = None,
    run_id: str | None = None,
):
    """Train the matching U-Net family as a 1-channel self-supervised reconstruction model."""
    strategy = strategy or tf.distribute.get_strategy()
    pre_config = grayscale_reconstruction_config(config)
    settings = self_supervised_settings(config)
    foreground_weight = float(settings.get("reconstruction_foreground_weight", 4.0))
    epochs = int(settings.get("epochs", 10))
    learning_rate = float(settings.get("learning_rate", 0.001))
    with strategy.scope():
        pre_model = get_model_builder(config["run"]["model"])(pre_config)
        pre_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss=lambda targets, prediction: foreground_weighted_reconstruction_mse(
                targets, prediction, foreground_weight
            ),
            metrics=[
                keras.metrics.MeanMetricWrapper(reconstruction_mse, name="reconstruction_mse"),
                keras.metrics.MeanMetricWrapper(
                    foreground_reconstruction_mse, name="foreground_mse"
                ),
                keras.metrics.MeanMetricWrapper(
                    reconstruction_prediction_mean, name="prediction_mean"
                ),
                keras.metrics.MeanMetricWrapper(
                    reconstruction_prediction_std, name="prediction_std"
                ),
            ],
        )
    history = pre_model.fit(
        dataset,
        epochs=epochs,
        verbose=0,
        callbacks=[
            RichTrainingStatusCallback(
                phase="SSL · grayscale reconstruction",
                epochs=epochs,
                training_log=training_log,
                run_id=run_id,
                display=_self_supervised_display(config, settings),
            )
        ],
    )
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
    layout = RunLayout(run_dir); layout.self_supervised_metrics.mkdir(parents=True, exist_ok=True); layout.self_supervised_model.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history.history).to_csv(layout.self_supervised_metrics / "metrics.csv", index_label="epoch")
    (layout.self_supervised_metrics / "metrics.json").write_text(json.dumps(history.history, indent=2, default=float) + "\n")
    write_history_jsonl(
        history.history,
        layout.self_supervised_metrics_jsonl,
        run_id=run_id,
        phase="self_supervised",
    )
    pre_model.save_weights(layout.self_supervised_model / "grayscale_reconstruction.weights.h5")
    return history


# Legacy import compatibility.
run_grayscale_reconstruction_pretraining = run_grayscale_reconstruction_self_supervised


def _view_config(config: dict[str, Any]) -> dict[str, Any]:
    self_supervised = self_supervised_settings(config)
    view_augmentation = dict(DEFAULT_VIEW_AUGMENTATION)
    view_augmentation.update(self_supervised.get("augmentation", {}))
    view_augmentation["enabled"] = bool(view_augmentation.get("enabled", True))
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


def vicreg_regularization(
    representations: tf.Tensor,
    *,
    target_std: float = 1.0,
    epsilon: float = 1e-4,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return VICReg variance/covariance penalties and mean feature stddev.

    The covariance term is normalized by feature dimension, so its scale remains
    bounded as projection dimensions change.  A one-example local batch has no
    off-diagonal covariance signal, but still receives the variance penalty.
    """
    representations = tf.convert_to_tensor(representations)
    centered = representations - tf.reduce_mean(representations, axis=0, keepdims=True)
    variance = tf.reduce_mean(tf.square(centered), axis=0)
    stabilized_stddev = tf.sqrt(variance + tf.cast(epsilon, representations.dtype))
    raw_stddev = tf.sqrt(tf.maximum(variance, tf.cast(0.0, representations.dtype)))
    variance_loss = tf.reduce_mean(
        tf.nn.relu(tf.cast(target_std, representations.dtype) - stabilized_stddev)
    )
    batch_size = tf.cast(tf.shape(representations)[0], representations.dtype)
    covariance = tf.matmul(centered, centered, transpose_a=True) / tf.maximum(
        batch_size - 1.0, 1.0
    )
    feature_dim = tf.cast(tf.shape(representations)[1], representations.dtype)
    off_diagonal = covariance - tf.linalg.diag(tf.linalg.diag_part(covariance))
    covariance_loss = tf.reduce_sum(tf.square(off_diagonal)) / tf.maximum(
        feature_dim, 1.0
    )
    return variance_loss, covariance_loss, tf.reduce_mean(raw_stddev)


def serving_feature_representations(representations: tf.Tensor) -> tf.Tensor:
    """Apply the classifier's serving-time unit-vector representation contract.

    SSL encoders intentionally expose ``embedding_projection`` (before the
    classifier's ``features`` normalization) so the projection head can retain
    magnitude information.  Collapse checks and encoder regularization must,
    however, operate in the same unit-vector space that clustering and
    inference export.  Match :class:`L2Normalization`'s zero-vector fallback
    here rather than letting ``l2_normalize`` turn an all-zero vector into a
    misleading zero vector.
    """
    representations = tf.convert_to_tensor(representations)
    norm = tf.norm(representations, axis=-1, keepdims=True)
    normalized = tf.math.divide_no_nan(representations, norm)
    fallback = tf.one_hot(
        tf.zeros(tf.shape(representations)[:-1], dtype=tf.int32),
        depth=tf.shape(representations)[-1],
        dtype=representations.dtype,
    )
    return tf.where(norm > 0, normalized, fallback)


def _mean_direction_norm(representations: tf.Tensor) -> tf.Tensor:
    """Return the norm of the batch mean for unit-vector health diagnostics."""
    return tf.norm(tf.reduce_mean(representations, axis=0))


def _all_gather_representations(representations: tf.Tensor) -> tf.Tensor:
    """Gather equal-sized per-replica representations for global SSL statistics."""
    replica_context = tf.distribute.get_replica_context()
    if replica_context is None:
        return representations
    return replica_context.all_gather(representations, axis=0)


def _cross_view_cosine_statistics(
    anchor_one: tf.Tensor,
    anchor_two: tf.Tensor,
    target_one: tf.Tensor,
    target_two: tf.Tensor,
    *,
    global_target_one: tf.Tensor | None = None,
    global_target_two: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return related and unrelated cross-view cosine similarity.

    Related pairs are the two views of the same sample. Unrelated similarity
    is the mean similarity to both views of every other sample in the global
    synchronized batch. The latter is computed from a global vector sum so it
    does not add another quadratic similarity matrix to the SSL step.
    """
    anchor_one = tf.math.l2_normalize(anchor_one, axis=-1)
    anchor_two = tf.math.l2_normalize(anchor_two, axis=-1)
    target_one = tf.math.l2_normalize(target_one, axis=-1)
    target_two = tf.math.l2_normalize(target_two, axis=-1)
    if global_target_one is None:
        global_target_one = _all_gather_representations(target_one)
    if global_target_two is None:
        global_target_two = _all_gather_representations(target_two)

    related = 0.5 * (
        tf.reduce_mean(tf.reduce_sum(anchor_one * target_two, axis=-1))
        + tf.reduce_mean(tf.reduce_sum(anchor_two * target_one, axis=-1))
    )

    global_target_sum = tf.reduce_sum(
        tf.concat([global_target_one, global_target_two], axis=0), axis=0
    )
    unrelated_count = tf.cast(
        tf.maximum(2 * tf.shape(global_target_one)[0] - 2, 1),
        anchor_one.dtype,
    )
    unrelated_one = tf.reduce_sum(
        anchor_one
        * (global_target_sum - target_one - target_two),
        axis=-1,
    ) / unrelated_count
    unrelated_two = tf.reduce_sum(
        anchor_two
        * (global_target_sum - target_one - target_two),
        axis=-1,
    ) / unrelated_count
    unrelated = 0.5 * (
        tf.reduce_mean(unrelated_one) + tf.reduce_mean(unrelated_two)
    )
    return related, unrelated


def _self_supervised_optimizer(settings: dict[str, Any]) -> keras.optimizers.Optimizer:
    """Build the SSL optimizer while retaining Adam as the historical default."""
    learning_rate = float(settings.get("learning_rate", 0.001))
    optimizer_name = str(
        settings.get("ssl_optimizer", settings.get("optimizer", "adam"))
    ).lower()
    optimizer_kwargs = {
        "learning_rate": learning_rate,
        "beta_1": float(settings.get("beta_1", 0.9)),
        "beta_2": float(settings.get("beta_2", 0.999)),
        "epsilon": float(settings.get("optimizer_epsilon", 1e-7)),
    }
    if optimizer_name == "adam":
        return keras.optimizers.Adam(**optimizer_kwargs)
    if optimizer_name == "adamw":
        return keras.optimizers.AdamW(
            weight_decay=float(settings.get("weight_decay", 1e-4)),
            **optimizer_kwargs,
        )
    raise ValueError("self_supervised.ssl_optimizer must be 'adam' or 'adamw'")


class StudentTeacherPretrainer(keras.Model):
    """BYOL-style student predictor trained against an EMA teacher."""

    def __init__(self, classifier: keras.Model, config: dict[str, Any]):
        super().__init__(name="student_teacher_pretrainer")
        settings = self_supervised_settings(config)
        self.view_config = _view_config(config)
        self.momentum = float(settings.get("teacher_momentum", 0.99))
        embedding_dim = int(classifier.get_layer("embedding_projection").output.shape[-1])
        projection_dim = int(settings.get("projection_dim", 128))
        hidden_dim = int(settings.get("projection_hidden_dim", max(embedding_dim, 256)))
        self.vicreg_variance_weight = float(settings.get("vicreg_variance_weight", 25.0))
        self.vicreg_covariance_weight = float(settings.get("vicreg_covariance_weight", 1.0))
        self.vicreg_target_std = float(settings.get("vicreg_target_std", 1.0))
        self.encoder_variance_weight = float(settings.get("encoder_variance_weight", 25.0))
        self.encoder_covariance_weight = float(settings.get("encoder_covariance_weight", 1.0))
        self.encoder_target_std = float(settings.get("encoder_target_std", 0.05))
        self.collapse_std_threshold = float(
            settings.get("collapse_std_threshold", 1e-3)
        )
        if self.vicreg_variance_weight < 0 or self.vicreg_covariance_weight < 0:
            raise ValueError("VICReg regularization weights must be non-negative")
        if self.vicreg_target_std <= 0:
            raise ValueError("vicreg_target_std must be greater than zero")
        if self.encoder_variance_weight < 0 or self.encoder_covariance_weight < 0:
            raise ValueError("encoder regularization weights must be non-negative")
        if self.encoder_target_std <= 0:
            raise ValueError("encoder_target_std must be greater than zero")
        if self.collapse_std_threshold <= 0:
            raise ValueError("collapse_std_threshold must be greater than zero")

        self.student_encoder = build_self_supervised_embedding_model(classifier)
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
        self.related_cosine_similarity_tracker = keras.metrics.Mean(
            name="related_cosine_similarity"
        )
        self.unrelated_cosine_similarity_tracker = keras.metrics.Mean(
            name="unrelated_cosine_similarity"
        )
        self.variance_loss_tracker = keras.metrics.Mean(name="variance_loss")
        self.covariance_loss_tracker = keras.metrics.Mean(name="covariance_loss")
        self.representation_std_tracker = keras.metrics.Mean(name="representation_std")
        # The historic VICReg metrics describe the projector.  Keep them for
        # compatibility, but also expose the encoder/product space explicitly.
        self.encoder_variance_loss_tracker = keras.metrics.Mean(name="encoder_variance_loss")
        self.encoder_covariance_loss_tracker = keras.metrics.Mean(name="encoder_covariance_loss")
        self.encoder_representation_std_tracker = keras.metrics.Mean(name="encoder_representation_std")
        self.encoder_mean_direction_norm_tracker = keras.metrics.Mean(name="encoder_mean_direction_norm")
        self.encoder_related_cosine_similarity_tracker = keras.metrics.Mean(name="encoder_related_cosine_similarity")
        self.encoder_unrelated_cosine_similarity_tracker = keras.metrics.Mean(name="encoder_unrelated_cosine_similarity")
        self.projector_representation_std_tracker = keras.metrics.Mean(name="projector_representation_std")
        self.projector_related_cosine_similarity_tracker = keras.metrics.Mean(name="projector_related_cosine_similarity")
        self.projector_unrelated_cosine_similarity_tracker = keras.metrics.Mean(name="projector_unrelated_cosine_similarity")
        self.collapse_indicator_tracker = keras.metrics.Mean(name="collapse_indicator")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.cosine_similarity_tracker,
            self.related_cosine_similarity_tracker,
            self.unrelated_cosine_similarity_tracker,
            self.variance_loss_tracker,
            self.covariance_loss_tracker,
            self.representation_std_tracker,
            self.encoder_variance_loss_tracker,
            self.encoder_covariance_loss_tracker,
            self.encoder_representation_std_tracker,
            self.encoder_mean_direction_norm_tracker,
            self.encoder_related_cosine_similarity_tracker,
            self.encoder_unrelated_cosine_similarity_tracker,
            self.projector_representation_std_tracker,
            self.projector_related_cosine_similarity_tracker,
            self.projector_unrelated_cosine_similarity_tracker,
            self.collapse_indicator_tracker,
        ]

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
            student_embedding_one = self.student_encoder(view_one, training=True)
            student_embedding_two = self.student_encoder(view_two, training=True)
            projected_one = self.student_projector(student_embedding_one, training=True)
            projected_two = self.student_projector(student_embedding_two, training=True)
            student_one = self.predictor(
                projected_one,
                training=True,
            )
            student_two = self.predictor(
                projected_two,
                training=True,
            )
            teacher_one = self.teacher_projector(
                self.teacher_encoder(view_one, training=False), training=False
            )
            teacher_two = self.teacher_projector(
                self.teacher_encoder(view_two, training=False), training=False
            )
            alignment_loss = 0.5 * (
                self._cosine_loss(student_one, teacher_two)
                + self._cosine_loss(student_two, teacher_one)
            )
            global_projected_one = _all_gather_representations(projected_one)
            global_projected_two = _all_gather_representations(projected_two)
            variance_one, covariance_one, std_one = vicreg_regularization(
                global_projected_one,
                target_std=self.vicreg_target_std,
            )
            variance_two, covariance_two, std_two = vicreg_regularization(
                global_projected_two,
                target_std=self.vicreg_target_std,
            )
            variance_loss = 0.5 * (variance_one + variance_two)
            covariance_loss = 0.5 * (covariance_one + covariance_two)
            representation_std = 0.5 * (std_one + std_two)
            encoder_one = serving_feature_representations(student_embedding_one)
            encoder_two = serving_feature_representations(student_embedding_two)
            global_encoder_one = _all_gather_representations(encoder_one)
            global_encoder_two = _all_gather_representations(encoder_two)
            encoder_variance_one, encoder_covariance_one, encoder_std_one = vicreg_regularization(
                global_encoder_one, target_std=self.encoder_target_std
            )
            encoder_variance_two, encoder_covariance_two, encoder_std_two = vicreg_regularization(
                global_encoder_two, target_std=self.encoder_target_std
            )
            encoder_variance_loss = 0.5 * (encoder_variance_one + encoder_variance_two)
            encoder_covariance_loss = 0.5 * (encoder_covariance_one + encoder_covariance_two)
            encoder_representation_std = 0.5 * (encoder_std_one + encoder_std_two)
            encoder_mean_direction_norm = 0.5 * (
                _mean_direction_norm(global_encoder_one)
                + _mean_direction_norm(global_encoder_two)
            )
            loss = (
                alignment_loss
                + self.vicreg_variance_weight * variance_loss
                + self.vicreg_covariance_weight * covariance_loss
                + self.encoder_variance_weight * encoder_variance_loss
                + self.encoder_covariance_weight * encoder_covariance_loss
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
        related_cosine_similarity, unrelated_cosine_similarity = (
            _cross_view_cosine_statistics(
                student_one,
                student_two,
                teacher_one,
                teacher_two,
            )
        )
        encoder_related_cosine_similarity, encoder_unrelated_cosine_similarity = (
            _cross_view_cosine_statistics(encoder_one, encoder_two, encoder_one, encoder_two)
        )
        cosine_similarity = related_cosine_similarity
        self.loss_tracker.update_state(loss)
        self.cosine_similarity_tracker.update_state(cosine_similarity)
        self.related_cosine_similarity_tracker.update_state(
            related_cosine_similarity
        )
        self.unrelated_cosine_similarity_tracker.update_state(
            unrelated_cosine_similarity
        )
        self.variance_loss_tracker.update_state(variance_loss)
        self.covariance_loss_tracker.update_state(covariance_loss)
        self.representation_std_tracker.update_state(encoder_representation_std)
        self.encoder_variance_loss_tracker.update_state(encoder_variance_loss)
        self.encoder_covariance_loss_tracker.update_state(encoder_covariance_loss)
        self.encoder_representation_std_tracker.update_state(encoder_representation_std)
        self.encoder_mean_direction_norm_tracker.update_state(encoder_mean_direction_norm)
        self.encoder_related_cosine_similarity_tracker.update_state(encoder_related_cosine_similarity)
        self.encoder_unrelated_cosine_similarity_tracker.update_state(encoder_unrelated_cosine_similarity)
        self.projector_representation_std_tracker.update_state(representation_std)
        self.projector_related_cosine_similarity_tracker.update_state(related_cosine_similarity)
        self.projector_unrelated_cosine_similarity_tracker.update_state(unrelated_cosine_similarity)
        self.collapse_indicator_tracker.update_state(
            tf.cast(
                encoder_representation_std < self.collapse_std_threshold,
                encoder_representation_std.dtype,
            )
        )
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
        settings = self_supervised_settings(config)
        self.view_config = _view_config(config)
        self.temperature = float(settings.get("temperature", 0.1))
        embedding_dim = int(classifier.get_layer("embedding_projection").output.shape[-1])
        projection_dim = int(settings.get("projection_dim", 128))
        hidden_dim = int(
            settings.get("projection_hidden_dim", max(embedding_dim, 256))
        )
        self.encoder = build_self_supervised_embedding_model(classifier)
        self.projector = _projection_head(
            embedding_dim,
            hidden_dim,
            projection_dim,
            "simclr_projector",
        )
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.simclr_loss_tracker = keras.metrics.Mean(name="simclr_loss")
        self.contrastive_accuracy = keras.metrics.Mean(
            name="contrastive_accuracy"
        )
        self.related_cosine_similarity_tracker = keras.metrics.Mean(
            name="related_cosine_similarity"
        )
        self.unrelated_cosine_similarity_tracker = keras.metrics.Mean(
            name="unrelated_cosine_similarity"
        )
        self.encoder_variance_weight = float(
            settings.get("encoder_variance_weight", 25.0)
        )
        self.encoder_covariance_weight = float(
            settings.get("encoder_covariance_weight", 1.0)
        )
        self.encoder_target_std = float(settings.get("encoder_target_std", 0.05))
        if self.encoder_variance_weight < 0 or self.encoder_covariance_weight < 0:
            raise ValueError("encoder regularization weights must be non-negative")
        if self.encoder_target_std <= 0:
            raise ValueError("encoder_target_std must be greater than zero")

        # Legacy metric names remain available, while the explicit names make
        # it impossible for projector success to hide a collapsed serving
        # encoder. ``representation_std`` now follows the exported encoder
        # representation, so existing collapse gates inspect the right space.
        self.encoder_variance_loss_tracker = keras.metrics.Mean(name="encoder_variance_loss")
        self.encoder_covariance_loss_tracker = keras.metrics.Mean(name="encoder_covariance_loss")
        self.encoder_representation_std_tracker = keras.metrics.Mean(name="encoder_representation_std")
        self.encoder_mean_direction_norm_tracker = keras.metrics.Mean(name="encoder_mean_direction_norm")
        self.encoder_related_cosine_similarity_tracker = keras.metrics.Mean(name="encoder_related_cosine_similarity")
        self.encoder_unrelated_cosine_similarity_tracker = keras.metrics.Mean(name="encoder_unrelated_cosine_similarity")
        self.projector_representation_std_tracker = keras.metrics.Mean(name="projector_representation_std")
        self.projector_related_cosine_similarity_tracker = keras.metrics.Mean(name="projector_related_cosine_similarity")
        self.projector_unrelated_cosine_similarity_tracker = keras.metrics.Mean(name="projector_unrelated_cosine_similarity")
        self.representation_std_tracker = keras.metrics.Mean(name="representation_std")
        self.collapse_indicator_tracker = keras.metrics.Mean(name="collapse_indicator")
        self.collapse_std_threshold = float(
            settings.get("collapse_std_threshold", 1e-3)
        )
        if self.collapse_std_threshold <= 0:
            raise ValueError("collapse_std_threshold must be greater than zero")

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.simclr_loss_tracker,
            self.contrastive_accuracy,
            self.related_cosine_similarity_tracker,
            self.unrelated_cosine_similarity_tracker,
            self.encoder_variance_loss_tracker,
            self.encoder_covariance_loss_tracker,
            self.encoder_representation_std_tracker,
            self.encoder_mean_direction_norm_tracker,
            self.encoder_related_cosine_similarity_tracker,
            self.encoder_unrelated_cosine_similarity_tracker,
            self.projector_representation_std_tracker,
            self.projector_related_cosine_similarity_tracker,
            self.projector_unrelated_cosine_similarity_tracker,
            self.representation_std_tracker,
            self.collapse_indicator_tracker,
        ]

    def call(self, inputs, training=False):
        """Expose encoder/projector inference so Keras can build before fit()."""
        embeddings = self.encoder(inputs, training=training)
        return self.projector(embeddings, training=training)

    def _augment(self, x):
        dummy = tf.zeros([tf.shape(x)[0]], dtype=tf.int32)
        augmented, _ = augment_batch(x, dummy, self.view_config)
        return augmented

    def _nt_xent(
        self,
        first,
        second,
        *,
        global_first=None,
        global_second=None,
    ):
        first = tf.math.l2_normalize(first, axis=-1)
        second = tf.math.l2_normalize(second, axis=-1)
        local_batch_size = tf.shape(first)[0]
        if global_first is None:
            global_first = _all_gather_representations(first)
        if global_second is None:
            global_second = _all_gather_representations(second)
        global_batch_size = tf.shape(global_first)[0]
        anchors = tf.concat([first, second], axis=0)
        representations = tf.concat([global_first, global_second], axis=0)
        logits = tf.matmul(
            anchors, representations, transpose_b=True
        ) / tf.cast(self.temperature, representations.dtype)
        replica_context = tf.distribute.get_replica_context()
        replica_id = (
            tf.cast(replica_context.replica_id_in_sync_group, tf.int32)
            if replica_context is not None
            else tf.constant(0, dtype=tf.int32)
        )
        offset = replica_id * local_batch_size
        local_indices = tf.range(local_batch_size, dtype=tf.int32) + offset
        positives = tf.concat(
            [
                global_batch_size + local_indices,
                local_indices,
            ],
            axis=0,
        )
        self_indices = tf.concat([local_indices, global_batch_size + local_indices], axis=0)
        logits = logits - tf.one_hot(
            self_indices,
            depth=2 * global_batch_size,
            dtype=logits.dtype,
        ) * tf.cast(1e9, logits.dtype)
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
            first_embedding = self.encoder(view_one, training=True)
            second_embedding = self.encoder(view_two, training=True)
            first = self.projector(
                first_embedding, training=True
            )
            second = self.projector(
                second_embedding, training=True
            )
            normalized_first = tf.math.l2_normalize(first, axis=-1)
            normalized_second = tf.math.l2_normalize(second, axis=-1)
            global_first = _all_gather_representations(normalized_first)
            global_second = _all_gather_representations(normalized_second)
            loss, accuracy = self._nt_xent(
                first,
                second,
                global_first=global_first,
                global_second=global_second,
            )
            related_cosine_similarity, unrelated_cosine_similarity = (
                _cross_view_cosine_statistics(
                    first,
                    second,
                    first,
                    second,
                    global_target_one=global_first,
                    global_target_two=global_second,
                )
            )
            # The projector is deliberately allowed to differ from the
            # product embedding space.  Its NT-Xent objective alone can look
            # healthy while the exported unit vectors collapse into one cone.
            # Regularize the serving representation directly to prevent that.
            serving_first_embedding = serving_feature_representations(first_embedding)
            serving_second_embedding = serving_feature_representations(second_embedding)
            global_serving_first = _all_gather_representations(serving_first_embedding)
            global_serving_second = _all_gather_representations(serving_second_embedding)
            encoder_variance_one, encoder_covariance_one, encoder_std_one = vicreg_regularization(
                global_serving_first, target_std=self.encoder_target_std
            )
            encoder_variance_two, encoder_covariance_two, encoder_std_two = vicreg_regularization(
                global_serving_second, target_std=self.encoder_target_std
            )
            encoder_variance_loss = 0.5 * (encoder_variance_one + encoder_variance_two)
            encoder_covariance_loss = 0.5 * (encoder_covariance_one + encoder_covariance_two)
            representation_std = 0.5 * (encoder_std_one + encoder_std_two)
            encoder_mean_direction_norm = 0.5 * (
                _mean_direction_norm(global_serving_first)
                + _mean_direction_norm(global_serving_second)
            )
            global_projected_embeddings = tf.concat(
                [
                    _all_gather_representations(first),
                    _all_gather_representations(second),
                ],
                axis=0,
            )
            projector_representation_std = tf.reduce_mean(
                tf.math.reduce_std(global_projected_embeddings, axis=0)
            )
            total_loss = (
                loss
                + self.encoder_variance_weight * encoder_variance_loss
                + self.encoder_covariance_weight * encoder_covariance_loss
            )
            optimization_loss = total_loss / tf.cast(
                tf.distribute.get_strategy().num_replicas_in_sync,
                total_loss.dtype,
            )
        variables = self.encoder.trainable_variables + self.projector.trainable_variables
        gradients = tape.gradient(optimization_loss, variables)
        self.optimizer.apply_gradients(
            (gradient, variable)
            for gradient, variable in zip(gradients, variables, strict=True)
            if gradient is not None
        )
        self.loss_tracker.update_state(total_loss)
        self.simclr_loss_tracker.update_state(loss)
        self.contrastive_accuracy.update_state(accuracy)
        self.related_cosine_similarity_tracker.update_state(
            related_cosine_similarity
        )
        self.unrelated_cosine_similarity_tracker.update_state(
            unrelated_cosine_similarity
        )
        encoder_related_cosine_similarity, encoder_unrelated_cosine_similarity = (
            _cross_view_cosine_statistics(
                serving_first_embedding,
                serving_second_embedding,
                serving_first_embedding,
                serving_second_embedding,
                global_target_one=global_serving_first,
                global_target_two=global_serving_second,
            )
        )
        self.encoder_variance_loss_tracker.update_state(encoder_variance_loss)
        self.encoder_covariance_loss_tracker.update_state(encoder_covariance_loss)
        self.encoder_representation_std_tracker.update_state(representation_std)
        self.encoder_mean_direction_norm_tracker.update_state(encoder_mean_direction_norm)
        self.encoder_related_cosine_similarity_tracker.update_state(encoder_related_cosine_similarity)
        self.encoder_unrelated_cosine_similarity_tracker.update_state(encoder_unrelated_cosine_similarity)
        self.projector_representation_std_tracker.update_state(projector_representation_std)
        self.projector_related_cosine_similarity_tracker.update_state(related_cosine_similarity)
        self.projector_unrelated_cosine_similarity_tracker.update_state(unrelated_cosine_similarity)
        self.representation_std_tracker.update_state(representation_std)
        self.collapse_indicator_tracker.update_state(
            tf.cast(
                representation_std < self.collapse_std_threshold,
                representation_std.dtype,
            )
        )
        return {metric.name: metric.result() for metric in self.metrics}


def run_self_supervised_training(
    classifier: keras.Model,
    dataset,
    config: dict[str, Any],
    run_dir: str | Path,
    strategy: tf.distribute.Strategy | None = None,
    *,
    training_log: str | Path | None = None,
    run_id: str | None = None,
):
    settings = self_supervised_settings(config)
    method = str(settings.get("method", "byol")).lower()
    if method not in {"byol", "student_teacher", "simclr"}:
        raise ValueError(
            "self_supervised.method must be byol, student_teacher, or simclr"
    )
    strategy = strategy or tf.distribute.get_strategy()
    if method == "simclr":
        batch_size = int(config.get("data", {}).get("batch_size", 16))
        replicas = int(strategy.num_replicas_in_sync)
        minimum_global_batch = int(
            settings.get("minimum_global_batch_size", 2)
        )
        if batch_size < minimum_global_batch:
            raise ValueError(
                "data.batch_size must be at least "
                "self_supervised.minimum_global_batch_size"
            )
        if batch_size // replicas < 2:
            raise ValueError(
                "SimCLR requires at least two samples per synchronized replica; "
                f"global batch size is {batch_size} across {replicas} replicas"
            )
    with strategy.scope():
        pretrainer = (
            SimCLRPretrainer(classifier, config)
            if method == "simclr"
            else StudentTeacherPretrainer(classifier, config)
        )
        pretrainer.compile(optimizer=_self_supervised_optimizer(settings))
        # Keras 3 may try a symbolic build before the first custom train_step.
        # Calling the explicit inference path here creates every wrapper variable
        # for all backbones, including EfficientNet.
        pretrainer(
            tf.zeros([1, *config["data"]["input_shape"]], dtype=tf.float32),
            training=False,
        )
    epochs = int(settings.get("epochs", 10))
    history = pretrainer.fit(
        dataset,
        epochs=epochs,
        verbose=0,
        callbacks=[
            RichTrainingStatusCallback(
                phase=f"SSL · {method.upper()}",
                epochs=epochs,
                training_log=training_log,
                run_id=run_id,
                display=_self_supervised_display(config, settings),
            )
        ],
    )
    from oracle_builder.artifacts.layout import RunLayout

    layout = RunLayout(run_dir)
    metrics_dir = layout.self_supervised_metrics
    model_dir = layout.self_supervised_model
    metrics_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history.history).to_csv(
        metrics_dir / "metrics.csv", index_label="epoch"
    )
    (metrics_dir / "metrics.json").write_text(
        json.dumps(history.history, indent=2, default=float) + "\n"
    )
    write_history_jsonl(
        history.history,
        layout.self_supervised_metrics_jsonl,
        run_id=run_id,
        phase="self_supervised",
    )
    classifier.save_weights(model_dir / "student_pretrained.weights.h5")
    return history


# Legacy import compatibility.
run_student_teacher_pretraining = run_self_supervised_training
_pretraining_optimizer = _self_supervised_optimizer
