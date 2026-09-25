"""Machine-readable Architecture V2 configuration contract.

This module is deliberately dependency-free.  It is the one place API clients
should ask what an editable Oracle Builder configuration looks like.  Runtime
defaults remain in :mod:`oracle_builder.config` while the migration is in
progress; the catalog derives its default values directly from that mapping so
the two cannot drift.  New user-facing fields belong here, including their UI
metadata and finite choices.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable

from oracle_builder.config import (
    STANDARD_DISTRIBUTION_STRATEGIES,
    STANDARD_TRAINING_LOSSES_BY_TASK,
    STANDARD_TRAINING_METRICS_BY_TASK,
    STANDARD_TRAINING_OPTIMIZERS,
    SUPPORTED_TRAINING_TASKS,
)


# Bump whenever clients need to understand a new catalog contract attribute.
# Version 3 removes the public V1 compatibility surface and adds explicit UI
# control semantics.  Clients must treat this as an authoring contract, not a
# dump of internal runtime defaults.
SCHEMA_VERSION = 4

EXPOSURES: tuple[dict[str, Any], ...] = (
    {
        "id": "standard",
        "label": "Standard",
        "description": "Frequently changed experimental protocol settings.",
        "default_visible": True,
    },
    {
        "id": "advanced",
        "label": "Advanced",
        "description": "Useful controls for regular, deliberate tuning.",
        "default_visible": True,
    },
    {
        "id": "expert",
        "label": "Expert",
        "description": "Low-level controls; reveal only with the expert toggle.",
        "default_visible": False,
    },
    {
        "id": "internal",
        "label": "Internal",
        "description": "Dataset-derived, runtime-owned, or compatibility-only values.",
        "default_visible": False,
    },
)

ARCHITECTURE_FAMILIES = (
    "simple_cnn", "resnet_like", "densenet_like", "resnet", "densenet",
    "efficientnet", "efficientnet_v2", "convnext", "mobilenet", "unet",
    "residual_unet", "unet_plus_plus",
)
ARCHITECTURE_VARIANTS = (
    "simple_cnn", "resnet_like", "densenet_like", "resnet18", "resnet34",
    "resnet50", "resnet101", "resnet152", "densenet121", "densenet169",
    "densenet201", *(f"efficientnet_b{index}" for index in range(8)),
    "efficientnet_v2_b0", "efficientnet_v2_b1", "efficientnet_v2_b2",
    "efficientnet_v2_b3", "efficientnet_v2_s", "efficientnet_v2_m",
    "efficientnet_v2_l", "convnext_tiny", "convnext_small",
    "mobilenet_v3_small", "mobilenet_v3_large", "unet", "residual_unet",
    "unet_plus_plus",
)
ARCHITECTURE_VARIANTS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "simple_cnn": ("simple_cnn",), "resnet_like": ("resnet_like",),
    "densenet_like": ("densenet_like",),
    "resnet": ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152"),
    "densenet": ("densenet121", "densenet169", "densenet201"),
    "efficientnet": tuple(f"efficientnet_b{index}" for index in range(8)),
    "efficientnet_v2": ("efficientnet_v2_b0", "efficientnet_v2_b1", "efficientnet_v2_b2", "efficientnet_v2_b3", "efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l"),
    "convnext": ("convnext_tiny", "convnext_small"),
    "mobilenet": ("mobilenet_v3_small", "mobilenet_v3_large"),
    "unet": ("unet",), "residual_unet": ("residual_unet",), "unet_plus_plus": ("unet_plus_plus",),
}


GROUPS: tuple[dict[str, str], ...] = (
    {"id": "run", "label": "Run"},
    {"id": "architecture", "label": "Architecture"},
    {"id": "input", "label": "Input"},
    {"id": "data", "label": "Data"},
    {"id": "preprocessing", "label": "Preprocessing"},
    {"id": "encoder", "label": "Encoder"},
    {"id": "stem", "label": "Stem"},
    {"id": "normalization", "label": "Normalization"},
    {"id": "pooling", "label": "Pooling"},
    {"id": "image_embedding", "label": "Image embedding"},
    {"id": "metadata", "label": "Metadata"},
    {"id": "fusion", "label": "Fusion"},
    {"id": "classifier", "label": "Classifier"},
    {"id": "classification", "label": "Classification options"},
    {"id": "self_supervised", "label": "Self-supervised pretraining"},
    {"id": "training", "label": "Training"},
    {"id": "monitoring", "label": "Monitoring"},
    {"id": "callbacks", "label": "Callbacks"},
    {"id": "recovery", "label": "Recovery"},
    {"id": "augmentation", "label": "Augmentation"},
    {"id": "distribution", "label": "Compute"},
    {"id": "inference", "label": "Inference"},
    {"id": "evaluation", "label": "Evaluation"},
    {"id": "tiling", "label": "Tiling"},
    {"id": "evidence", "label": "Evidence"},
    {"id": "output", "label": "Output"},
    {"id": "posthoc", "label": "Post-hoc predictor"},
)


@dataclass(frozen=True)
class FieldSpec:
    """A JSON-serializable definition for one dotted TOML leaf."""

    path: str
    value_type: str
    default: Any = None
    label: str | None = None
    help: str | None = None
    choices: tuple[Any, ...] | None = None
    minimum: float | int | None = None
    maximum: float | int | None = None
    advanced: bool = False
    editable: bool = True
    applies_to: tuple[str, ...] = ()
    visible_when: dict[str, Any] | None = None
    aliases: tuple[str, ...] = ()
    # ``control`` tells a GUI which interaction to use.  It deliberately
    # describes interaction rather than a visual component library so native
    # and web clients can implement it consistently.
    control: str | None = None
    choice_labels: tuple[str, ...] | None = None
    step: float | int | None = None
    item_type: str | None = None
    required_choices: tuple[Any, ...] = ()
    allowed_values_by: dict[str, tuple[Any, ...]] | None = None
    # When a finite choice set depends on another configuration value, name
    # that source explicitly.  This avoids clients guessing based on a field
    # name (for example, loss/metrics depend on ``run.task`` while encoder
    # variants depend on ``encoder.family``).
    allowed_values_path: str | None = None

    def as_dict(self, exposure: str) -> dict[str, Any]:
        result: dict[str, Any] = {
            "path": self.path,
            "type": self.value_type,
            "default": copy.deepcopy(self.default),
            "label": self.label or _label(self.path),
            "group": self.path.split(".", 1)[0],
            # Retained for clients written before exposure tiers. Expert
            # controls are necessarily advanced, while standard controls are
            # not. New clients should use ``exposure``.
            "advanced": self.advanced or exposure in {"advanced", "expert"},
            "exposure": exposure,
            "editable": self.editable,
            "control": self.control or _control_for(self),
        }
        if self.help:
            result["help"] = self.help
        if self.choices is not None:
            result["choices"] = list(self.choices)
            labels = self.choice_labels or tuple(_choice_label(choice) for choice in self.choices)
            result["choice_options"] = [
                {"value": copy.deepcopy(choice), "label": label}
                for choice, label in zip(self.choices, labels, strict=True)
            ]
        if self.minimum is not None:
            result["minimum"] = self.minimum
        if self.maximum is not None:
            result["maximum"] = self.maximum
        if self.step is not None:
            result["step"] = self.step
        if self.item_type is not None:
            result["item_type"] = self.item_type
        if self.required_choices:
            result["required_choices"] = list(self.required_choices)
        if self.allowed_values_by:
            result["allowed_values_by"] = {
                key: list(values) for key, values in self.allowed_values_by.items()
            }
        if self.allowed_values_path:
            result["allowed_values_path"] = self.allowed_values_path
        if self.applies_to:
            result["applies_to"] = list(self.applies_to)
        if self.visible_when:
            result["visible_when"] = copy.deepcopy(self.visible_when)
        if self.aliases:
            result["aliases"] = list(self.aliases)
        return result


def _label(path: str) -> str:
    return path.rsplit(".", 1)[-1].replace("_", " ").capitalize()


def _choice_label(choice: Any) -> str:
    if choice is True:
        return "On"
    if choice is False:
        return "Off"
    if choice == "auto":
        return "Auto"
    return str(choice).replace("_", " ").replace("-", " ").title()


def _control_for(field: FieldSpec) -> str:
    if field.value_type == "boolean":
        return "toggle"
    if field.value_type in {"boolean_or_auto", "integer_or_auto"}:
        return "segmented" if field.value_type == "boolean_or_auto" else "auto_number"
    if field.choices is not None:
        return "segmented" if len(field.choices) in {2, 3} else "select"
    if field.value_type in {"integer", "number"}:
        return "number"
    if field.value_type == "shape":
        return "shape"
    if field.value_type == "list":
        return "list"
    return "text"


def _type_of(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "list"
    if value is None:
        return "null"
    return "string"


def _leaves(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        if not value:
            return
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else key
            yield from _leaves(item, child)
        return
    yield prefix, value


# Field semantics which cannot be inferred from a Python default.  Keep choices
# as machine values (rather than labels) so the same contract validates TOML,
# API drafts, and GUI select controls.
_OVERRIDES: dict[str, dict[str, Any]] = {
    "architecture.version": {"choices": (2,), "editable": False, "help": "Architecture V2 is the supported authoring contract."},
    "data.split_strategy": {"choices": ("auto", "random", "source_partitions")},
    "data.candidate_distance": {"choices": ("none", "euclidean_sdf", "geodesic"), "advanced": True, "applies_to": ("segmentation",)},
    "data.streaming.reader_workers": {"minimum": 1, "advanced": True},
    "data.streaming.prefetch_batches": {"minimum": 1, "advanced": True},
    "data.streaming.sqlite_cache_kib": {"minimum": 1, "advanced": True},
    "data.materialization.mode": {"choices": ("off", "run", "shared"), "advanced": True, "applies_to": ("classification",)},
    "data.materialization.format": {"choices": ("npy_shards",), "advanced": True, "applies_to": ("classification",)},
    "data.materialization.dtype": {"choices": ("float16", "float32"), "advanced": True, "applies_to": ("classification",)},
    "data.materialization.shard_samples": {"minimum": 1, "advanced": True, "applies_to": ("classification",)},
    "preprocessing.resize_mode": {"choices": ("fit_pad", "fit_pad_max_2x", "fit_pad_max_3x", "center_pad", "center_roi_pad", "fill_crop", "center_crop", "stretch", "none", "fit")},
    "preprocessing.normalization": {"choices": ("dtype", "minmax", "percentile", "none")},
    "preprocessing.invert": {"choices": (True, False, "auto"), "value_type": "boolean_or_auto", "choice_labels": ("Yes", "No", "Auto")},
    "preprocessing.pad_mode": {"choices": ("constant", "edge", "reflect", "symmetric"), "advanced": True},
    "preprocessing.interpolation": {"choices": ("nearest", "bilinear", "bicubic", "lanczos")},
    "preprocessing.channel_mode": {"choices": ("auto", "grayscale", "rgb", "rgba")},
    "normalization.type": {"choices": ("batch", "group", "layer", "layernorm", "layer_norm", "none")},
    "pooling.type": {"choices": ("avg", "max", "avg_max", "gem")},
    "fusion.type": {"choices": ("concat", "projected")},
    "classifier.type": {"choices": ("linear", "mlp", "cosine", "prototype")},
    "classifier.prototypes_per_class": {"minimum": 1, "visible_when": {"classifier.type": "prototype"}},
    "classifier.prototype_metric": {"choices": ("cosine", "euclidean"), "visible_when": {"classifier.type": "prototype"}},
    "metadata.augmentation.gaussian_variance": {"minimum": 0, "applies_to": ("classification",), "advanced": True},
    "metadata.augmentation.probability": {"minimum": 0, "maximum": 1, "advanced": True},
    "metadata.augmentation.apply_to": {"choices": ("continuous",), "advanced": True},
    # These are the ordinary experiment controls.  Their maintained choices
    # originate in config.py, so guided clients cannot silently drift from the
    # configuration accepted by the training runtime.
    "training.epochs": {"minimum": 1, "maximum": 100000, "step": 1},
    "training.optimizer": {
        "choices": STANDARD_TRAINING_OPTIMIZERS,
        "help": "Optimizer used for supervised training.",
    },
    "training.learning_rate": {
        "minimum": 1e-8,
        "maximum": 10.0,
        "step": 0.0001,
        "help": "Initial supervised-training learning rate.",
    },
    "training.loss": {
        "choices": tuple(sorted({loss for values in STANDARD_TRAINING_LOSSES_BY_TASK.values() for loss in values})),
        "allowed_values_by": STANDARD_TRAINING_LOSSES_BY_TASK,
        "allowed_values_path": "run.task",
        "help": "Task-compatible supervised loss.",
    },
    "training.metrics": {
        "choices": tuple(sorted({metric for values in STANDARD_TRAINING_METRICS_BY_TASK.values() for metric in values})),
        "allowed_values_by": STANDARD_TRAINING_METRICS_BY_TASK,
        "allowed_values_path": "run.task",
        "control": "multi_select",
        "item_type": "metric",
        "help": "Metrics reported for each training and validation epoch.",
    },
    "training.display": {"choices": ("rich", "text", "off")},
    "training.weight_decay": {"minimum": 0, "advanced": True},
    "training.class_weights.mode": {"choices": ("explicit", "inverse_frequency", "effective_number"), "applies_to": ("classification",)},
    "monitoring.primary_metric": {"help": "Metric used for the dashboard health summary. Use auto to prefer validation macro F1, then validation accuracy or loss."},
    "monitoring.target_metric": {"help": "Exact emitted metric name to watch, for example val_macro_f1 or val_loss."},
    "monitoring.target_value": {"help": "Target is satisfied when higher-is-better metrics meet it, or lower-is-better metrics are at or below it."},
    "monitoring.max_validation_loss_increase": {"minimum": 0, "help": "Alert when validation loss rises this far above its best completed value."},
    "monitoring.max_generalization_gap": {"minimum": 0, "help": "Alert when validation is this far worse than its paired training metric."},
    "distribution.strategy": {
        "choices": STANDARD_DISTRIBUTION_STRATEGIES,
        "help": "How the selected endpoint distributes this run across compute devices.",
    },
    "distribution.cross_device_ops": {"choices": ("auto", "nccl", "hierarchical_copy", "hierarchical-copy"), "advanced": True},
    "distribution.gpu_selection": {"choices": ("unused_first",), "advanced": True},
    "evidence.knn_k": {"minimum": 1},
    "inference.batch_size": {"choices": ("auto",), "value_type": "integer_or_auto", "help": "Use Auto or a positive integer.", "minimum": 1},
    "inference.minimum_batch_size": {"minimum": 1},
    "inference.memory_budget_mb": {"minimum": 64},
    "tiling.blend_mode": {"choices": ("hann", "uniform"), "applies_to": ("segmentation",)},
    "self_supervised.method": {"choices": ("byol", "student_teacher", "simclr", "grayscale_reconstruction"), "advanced": True},
    "self_supervised.ssl_optimizer": {"choices": ("adam", "adamw"), "advanced": True},
}


# Fields that are required, V2 component settings, or family-specific options
# and therefore have no leaf in DEFAULT_CONFIG.  A GUI must be able to render
# them even before a focused recipe is selected.
_REFERENCE_ONLY_DEFAULTS: dict[str, Any] = {
    "classifier.prototypes_per_class": 1,
    "classifier.prototype_metric": "cosine",
    "classification.stratification.enabled": False,
    "classification.stratification.dimensions": [64, 128, 256],
    "classification.stratification.basis": "max_original_dimension",
    "classification.stratification.batch_size_policy": "constant_input_tensor",
    "classification.stratification.weight_sharing": "shared",
    "classification.stratification.normalization": "batch",
    "classification.stratification.group_norm_groups": 8,
    "classification.stratification.assignment_policy": "smallest_fitting",
    "classification.stratification.supra_epochs": 5,
    "classification.stratification.schedule": "interleaved_steps",
    "classification.stratification.steps_per_stratum": 1,
    "classification.stratification.conditioning.enabled": False,
    "classification.stratification.conditioning.embedding_dim": 16,
    "classification.stratification.training_routing.enabled": False,
    "classification.stratification.training_routing.seed": 123,
    "classification.stratification.training_routing.adjacent_lower_probability": 0.0,
    "classification.stratification.cycle_scheduler.aggregation": "sample_weighted",
    "classification.stratification.cycle_scheduler.guardrail_metric": "macro_f1",
    "classification.stratification.cycle_scheduler.reduce_lr_patience": 3,
    "classification.stratification.cycle_scheduler.reduce_lr_factor": 0.5,
    "classification.stratification.cycle_scheduler.max_stratum_drop": 0.05,
    "self_supervised.beta_1": 0.9,
    "self_supervised.beta_2": 0.999,
    "self_supervised.optimizer_epsilon": 1e-7,
    "self_supervised.database": None,
    "self_supervised.augmentation.enabled": True,
    "self_supervised.augmentation.rotation": 0.08,
    "self_supervised.augmentation.zoom": 0.15,
    "self_supervised.augmentation.translation": 0.10,
    "self_supervised.augmentation.skew": 0.05,
    "self_supervised.augmentation.flip_horizontal": True,
    "self_supervised.augmentation.flip_vertical": False,
    "self_supervised.augmentation.brightness": 0.20,
    "self_supervised.augmentation.contrast": 0.20,
    "self_supervised.augmentation.gaussian_noise": 0.03,
    "self_supervised.augmentation.invert": False,
    "self_supervised.augmentation.fill_value": 0.0,
    "augmentation.signed_distance_fill_value": -1.0,
    "augmentation.photometric_channels": [0],
    "augmentation.mask_input_channels": [1],
    "augmentation.signed_distance_input_channels": [2],
}


_EXTRA_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("run.task", "enum", "classification", choices=SUPPORTED_TRAINING_TASKS, help="Training task."),
    FieldSpec("data.input_shape", "shape", [128, 128], help="[height, width] for classification/embedding; segmentation includes channels."),
    FieldSpec("data.output_shape", "shape", [128, 128, 1], applies_to=("segmentation",), help="Required segmentation target shape."),
    FieldSpec("data.num_classes", "integer", None, editable=False, applies_to=("classification", "embedding"), help="Read from the selected Oracle SQLite dataset."),
    FieldSpec(
        "input.channels", "list", ["intensity"],
        choices=("intensity", "gradient_magnitude", "gx", "gy", "gaussian_blur", "laplacian", "local_contrast", "foreground_mask", "foreground_distance"),
        choice_labels=("Intensity", "Sobel gradient", "Sobel X", "Sobel Y", "Gaussian blur", "Laplacian", "Local contrast", "Foreground mask", "Foreground distance"),
        required_choices=("intensity",), control="multi_select", item_type="channel",
        help="Choose the image channels constructed for both training and serving. Intensity is always required.",
    ),
    FieldSpec("input.geometry.type", "enum", "preserve_aspect_pad", choices=("preserve_aspect_pad", "center_roi_pad", "resize_crop", "center_crop", "direct_resize"), help="V2 geometry declaration; it resolves the preprocessing resize mode."),
    FieldSpec("encoder.family", "enum", "simple_cnn", choices=ARCHITECTURE_FAMILIES),
    FieldSpec("encoder.variant", "enum", "simple_cnn", choices=ARCHITECTURE_VARIANTS, help="Concrete supported topology for the selected encoder family.", allowed_values_by=ARCHITECTURE_VARIANTS_BY_FAMILY, allowed_values_path="encoder.family"),
    FieldSpec("stem.type", "enum", "native", choices=("native",), help="Native family stem; its parameters are explicit below."),
    FieldSpec("stem.kernel_size", "integer", 3, minimum=1),
    FieldSpec("stem.stride", "integer", 1, minimum=1),
    FieldSpec("stem.width", "integer", None, minimum=1, advanced=True),
    FieldSpec("stem.pool", "enum", "none", choices=("none", "max")),
    FieldSpec("normalization.groups", "integer_or_auto", "auto", advanced=True, visible_when={"normalization.type": "group"}),
    FieldSpec("pooling.p", "number", 3.0, minimum=0, advanced=True, visible_when={"pooling.type": "gem"}),
    FieldSpec("pooling.trainable_p", "boolean", False, advanced=True, visible_when={"pooling.type": "gem"}),
    FieldSpec("image_embedding.projection.output_dim", "integer", 256, minimum=1),
    FieldSpec("image_embedding.projection.l2_normalize", "boolean", False),
    FieldSpec("image_embedding.projection.normalize", "boolean", True, help="Normalizes the exported fused features representation."),
    FieldSpec("image_embedding.projection.hidden_units", "list", [], advanced=True, visible_when={"image_embedding.projection.type": "mlp"}),
    FieldSpec("image_embedding.projection.activation", "keras_identifier", "gelu", advanced=True),
    FieldSpec("image_embedding.projection.dropout", "number", 0.0, minimum=0, maximum=1),
    FieldSpec("metadata.encoder.output_dim", "integer", None, minimum=1, advanced=True, visible_when={"metadata.encoder.type": "mlp"}),
    FieldSpec("metadata.encoder.type", "enum", "direct", choices=("direct", "mlp"), help="Directly concatenate metadata or learn a metadata embedding."),
    FieldSpec("metadata.encoder.hidden_units", "list", [], advanced=True, visible_when={"metadata.encoder.type": "mlp"}),
    FieldSpec("metadata.encoder.activation", "keras_identifier", "relu", advanced=True),
    FieldSpec("metadata.encoder.dropout", "number", 0.0, minimum=0, maximum=1, advanced=True),
    FieldSpec("fusion.output_dim", "integer", None, minimum=1, advanced=True, visible_when={"fusion.type": "projected"}),
    FieldSpec("fusion.activation", "keras_identifier", "relu", advanced=True, visible_when={"fusion.type": "projected"}),
    FieldSpec("classifier.hidden_units", "list", [], advanced=True, visible_when={"classifier.type": "mlp"}),
    FieldSpec("classifier.activation", "keras_identifier", "relu", advanced=True, visible_when={"classifier.type": "mlp"}),
    FieldSpec("classifier.dropout", "number", 0.0, minimum=0, maximum=1, advanced=True, visible_when={"classifier.type": "mlp"}),
    FieldSpec("classifier.cosine_scale", "number", 16.0, minimum=0, advanced=True, visible_when={"classifier.type": ("cosine", "prototype")}),
    FieldSpec(
        "preprocessing.derive_after_augmentation", "boolean", False, advanced=True,
        visible_when={"input.channels": ("gradient_magnitude", "gx", "gy", "gaussian_blur", "laplacian", "local_contrast", "foreground_mask", "foreground_distance")},
        help="Build selected derived channels after random image augmentation.",
    ),
    FieldSpec(
        "preprocessing.derived_channels.local_contrast_sigma", "number", 3.0, minimum=0, advanced=True,
        visible_when={"input.channels": ("gaussian_blur", "laplacian", "local_contrast")},
        help="Filter scale used by the selected blur, Laplacian, or local-contrast channel.",
    ),
    FieldSpec("output.prediction_commit_batches", "integer", 20, minimum=1, advanced=True),
    FieldSpec("posthoc.enabled", "boolean", False, advanced=True),
    FieldSpec("posthoc.representation", "enum", "fused_embedding", choices=("fused_embedding",), advanced=True),
    FieldSpec("posthoc.estimator", "enum", "logistic_regression", choices=("logistic_regression",), advanced=True),
    FieldSpec("metadata.fields[].transform", "enum", "identity", choices=("identity", "log1p", "sqrt"), help="Per-field transform applied before fitting metadata scaling."),
    *(
        FieldSpec(
            path,
            _OVERRIDES.get(path, {}).get("value_type", _type_of(default)),
            default,
            advanced=True,
            **{key: value for key, value in _OVERRIDES.get(path, {}).items() if key != "value_type"},
        )
        for path, default in _REFERENCE_ONLY_DEFAULTS.items()
    ),
)


def v2_defaults() -> dict[str, Any]:
    """Return public V2 authoring defaults without runtime compatibility keys."""
    # Local import avoids a module cycle while config.py owns runtime merging.
    from oracle_builder.config import DEFAULT_CONFIG, deep_merge

    defaults = deep_merge(DEFAULT_CONFIG, {})
    # The runtime keeps these keys to execute historical artifacts.  They are
    # intentionally absent here: a new definition must only express V2.
    defaults["self_supervised"] = deep_merge(defaults.pop("pretraining"), {})
    defaults.pop("model", None)
    defaults.get("run", {}).pop("model", None)
    return defaults


# Compatibility name for callers that previously consumed the private helper.
_runtime_defaults = v2_defaults


_STANDARD_PATHS = {
    "run.task", "run.seed", "run.notes",
    "data.input_shape", "data.output_shape", "data.batch_size",
    "data.validation_split", "data.test_split", "data.shuffle_buffer",
    "input.channels", "input.geometry.type",
    "preprocessing.resize_mode", "preprocessing.rescale", "preprocessing.invert",
    "preprocessing.channel_mode", "preprocessing.interpolation",
    "encoder.family", "encoder.variant", "stem.type", "stem.kernel_size",
    "stem.stride", "stem.pool", "normalization.type", "pooling.type",
    "image_embedding.projection.type", "image_embedding.projection.output_dim",
    "metadata.fields", "metadata.scaling", "fusion.type", "classifier.type",
    "training.epochs", "training.optimizer",
    "training.learning_rate", "training.loss", "training.metrics",
    "training.class_weights.mode", "callbacks.early_stopping",
    "callbacks.early_stopping_patience", "callbacks.reduce_lr_on_plateau",
    "augmentation.enabled", "augmentation.rotation", "augmentation.zoom",
    "augmentation.translation", "augmentation.flip_horizontal",
    "augmentation.flip_vertical", "distribution.strategy",
}

_INTERNAL_PATHS = {
    "architecture.version", "data.num_classes",
    # Batch sizing is selected and sealed against a specific compute allocation
    # at Queue validation, not edited while defining reusable model science.
    "data.batch_size",
    # In V2, geometry is the single authoring choice; the runtime derives its
    # resize-mode compatibility value from ``input.geometry.type``.
    "preprocessing.resize_mode",
    # ``input.channels`` is the V2 authoring surface.  These legacy boolean
    # aliases must not create a second, contradictory way to choose channels.
    "preprocessing.derived_channels.gradient_magnitude",
    "preprocessing.derived_channels.local_contrast",
}
_EXPERT_PREFIXES = (
    "classification.stratification.", "data.geodesic_distance.",
    "data.streaming.", "inference.", "evaluation.", "tiling.",
    "self_supervised.",
)
_EXPERT_PATHS = {
    "data.materialization.root", "data.materialization.format",
    "data.materialization.dtype", "data.materialization.shard_samples",
    "data.materialization.build_if_missing", "data.materialization.wait_seconds",
    "data.materialization.max_cache_gib", "data.candidate_sdf",
    "data.candidate_sdf_clip_distance", "data.candidate_distance",
    "data.candidate_distance_clip", "preprocessing.upscale_limit",
    "preprocessing.pad_anchor", "preprocessing.pad_mode", "preprocessing.crop_anchor",
    "preprocessing.percentile_low", "preprocessing.percentile_high",
    "preprocessing.derive_after_augmentation", "preprocessing.derived_channels.local_contrast_sigma",
    "stem.width", "normalization.groups", "pooling.p", "pooling.trainable_p",
    "distribution.cross_device_ops", "distribution.gpu_selection",
    "distribution.require_unused_gpu", "distribution.allow_busy_fallback",
    "distribution.gpu_light_share_memory_mb", "distribution.gpu_light_share_utilization_percent",
    "distribution.gpu_lease_directory", "output.prediction_commit_batches",
}


def field_exposure(path: str) -> str:
    """Return the UI exposure tier for a canonical dotted configuration path."""
    if path in _INTERNAL_PATHS:
        return "internal"
    if path in _EXPERT_PATHS or path.startswith(_EXPERT_PREFIXES):
        return "expert"
    if path in _STANDARD_PATHS:
        return "standard"
    return "advanced"


def field_specs() -> list[FieldSpec]:
    """Return every editable/defaulted field in deterministic path order."""
    specs: dict[str, FieldSpec] = {}
    for path, default in _leaves(v2_defaults()):
        metadata = _OVERRIDES.get(path, {})
        value_type = metadata.get("value_type", _type_of(default))
        specs[path] = FieldSpec(
            path,
            value_type,
            default,
            **{key: value for key, value in metadata.items() if key != "value_type"},
        )
    for field in _EXTRA_FIELDS:
        specs[field.path] = field
    return [specs[path] for path in sorted(specs)]


def configuration_schema() -> dict[str, Any]:
    """Return the complete JSON-safe contract consumed by guided clients."""
    fields = [field.as_dict(field_exposure(field.path)) for field in field_specs()]
    fingerprint_input = json.dumps(fields, sort_keys=True, separators=(",", ":"), default=str)
    return {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": hashlib.sha256(fingerprint_input.encode("utf-8")).hexdigest(),
        "defaults": v2_defaults(),
        # Keep the stable string list for existing API clients; group metadata
        # gives new clients display labels without a second dictionary.
        "groups": [group["id"] for group in GROUPS],
        "group_definitions": [dict(group) for group in GROUPS],
        "exposures": [dict(exposure) for exposure in EXPOSURES],
        "fields": fields,
    }


def classification_preset_defaults() -> dict[str, Any]:
    """Return the maintained, dataset-independent V2 classification baseline.

    Focused TOMLs intentionally do not repeat every advanced knob.  This is
    their shared authored surface; omitted settings inherit the runtime V2
    defaults exposed by :func:`configuration_schema` and documented in the
    exhaustive reference.
    """
    return {
        "run": {"task": "classification", "seed": 123},
        "architecture": {"version": 2},
        "input": {"channels": ["intensity"], "geometry": {"type": "preserve_aspect_pad"}},
        "data": {
            "input_shape": [128, 128], "batch_size": 16, "shuffle_buffer": 512,
            "validation_split": 0.20, "test_split": 0.10,
            "streaming": {"enabled": True, "reader_workers": 4, "prefetch_batches": 2, "deterministic": True, "sqlite_cache_kib": 65536},
        },
        "preprocessing": {
            "resize_mode": "fit_pad", "normalization": "dtype", "rescale": True,
            "invert": "auto", "pad_value": 0.0, "interpolation": "bilinear", "channel_mode": "grayscale",
        },
        "normalization": {"type": "batch"},
        "pooling": {"type": "avg"},
        "image_embedding": {"projection": {"type": "linear", "output_dim": 256, "l2_normalize": False}},
        "metadata": {
            "fields": [], "scaling": "standard", "encoder": {"type": "direct"},
            "augmentation": {"gaussian_variance": 0.0, "probability": 1.0, "apply_to": "continuous", "fields": []},
        },
        "fusion": {"type": "concat"},
        "classifier": {"type": "linear"},
        "model": {"embedding_dim": 256, "normalize_embeddings": True, "dropout": 0.0},
        "training": {"epochs": 50, "optimizer": "adam", "learning_rate": 0.001, "loss": "weighted_sparse_categorical_crossentropy", "metrics": ["accuracy", "macro_f1"], "class_weights": {"mode": "effective_number", "beta": 0.999, "normalize": True}},
        "callbacks": {"early_stopping": True, "early_stopping_patience": 18, "reduce_lr_on_plateau": True, "checkpoint_monitor": "val_loss"},
        "augmentation": {"enabled": True, "repeats_per_epoch": 1, "invert": False, "rotation": 0.5, "zoom": 0.20, "translation": [0.15, 0.15], "skew": 0.20, "flip_horizontal": True, "flip_vertical": True, "brightness": 0.20, "contrast": 0.20, "gaussian_noise": 0.05, "fill_value": 0.0},
        "distribution": {"strategy": "single", "devices": [], "cross_device_ops": "auto", "fallback_to_single": True, "memory_growth": True, "gpu_selection": "unused_first", "require_unused_gpu": False, "allow_busy_fallback": False, "gpu_light_share_memory_mb": 1024, "gpu_light_share_utilization_percent": 15},
        "evidence": {"enabled": True, "knn_k": 5},
        "output": {"save_predictions": True, "save_figures": True, "export_savedmodel": True},
    }


def classification_stratification_preset() -> dict[str, Any]:
    """Conservative shared-resolution policy used by family baseline TOMLs."""
    return {
        "classification": {
            "stratification": {
                "enabled": False,
                "dimensions": [32, 64, 128],
                "basis": "max_original_dimension",
                "batch_size_policy": "constant_input_tensor",
                "weight_sharing": "shared",
                "normalization": "group",
                "group_norm_groups": 8,
                "assignment_policy": "smallest_fitting",
                "supra_epochs": 1,
                "schedule": "interleaved_steps",
                "steps_per_stratum": 1,
                "conditioning": {"enabled": True, "embedding_dim": 16},
                "training_routing": {"enabled": False, "seed": 123, "adjacent_lower_probability": 0.0},
                "cycle_scheduler": {"aggregation": "equal_strata", "guardrail_metric": "macro_f1", "reduce_lr_patience": 3, "reduce_lr_factor": 0.5, "max_stratum_drop": 0.03},
            }
        }
    }
