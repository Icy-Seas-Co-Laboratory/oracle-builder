#!/usr/bin/env python3
"""Generate the documented, dataset-independent classification examples."""

from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent


FAMILIES = {
    "simple_cnn": {
        "summary": "Small three-stage CNN; fastest baseline and useful for pipeline checks.",
        "dropout_help": "Dropout applied before the final classifier.",
        "model": """\
# Width of the three convolution stages is base_filters, 2x, and 4x.
# Increase this for more capacity; memory and compute grow quickly.
base_filters = 32""",
    },
    "resnet_like": {
        "summary": "Compact fixed-depth residual CNN; a lightweight residual baseline.",
        "dropout_help": "Dropout applied before the final classifier.",
        "model": """\
# Width of the three residual stages is base_filters, 2x, and 4x.
# This model has a fixed shallow topology. Use the resnet family when depth
# selection or an ImageNet-style stem is important.
base_filters = 32""",
        "normalize_embeddings": "false",
    },
    "densenet_like": {
        "summary": "Compact fixed-depth densely connected CNN; a lightweight dense baseline.",
        "dropout_help": "Dropout applied before the final classifier.",
        "model": """\
# Controls both the initial width and the growth rate of this compact network.
# This model has a fixed shallow topology. Use the densenet family for standard
# DenseNet-121/169/201 layouts and finer architectural control.
base_filters = 24""",
    },
    "resnet": {
        "summary": "Configurable standard ResNet family with basic and bottleneck variants.",
        "dropout_help": "Dropout applied before the final classifier.",
        "model": """\
# Valid variants: resnet18, resnet34, resnet50, resnet101, resnet152.
# 18/34 use basic blocks; 50/101/152 use higher-capacity bottleneck blocks.
# ResNet-18 is the best first run when speed or dataset size is uncertain.
variant = "resnet18"
base_filters = 64

# The stem downsamples by stem_stride and, when enabled, another 2x in stem_pool.
# For small objects or low-resolution inputs, try kernel 3, stride 1, pool false.
stem_kernel_size = 3
stem_stride = 1
stem_pool = false

# Optional expert override; four positive stage depths are required. Leaving
# this commented uses the selected variant's canonical layout.
# block_counts = [2, 2, 2, 2]""",
        "normalize_embeddings": "false",
    },
    "densenet": {
        "summary": "Configurable standard DenseNet family with feature reuse across layers.",
        "dropout_help": (
            "Dropout applied inside every dense layer and before the final classifier."
        ),
        "model": """\
# Valid variants: densenet121, densenet169, densenet201.
# DenseNet-121 is the usual first choice; larger variants cost more memory.
variant = "densenet121"
growth_rate = 32
initial_filters = 64
bottleneck_multiplier = 4

# Transition layers retain this fraction of channels. Lower values save memory.
compression = 0.50
stem_kernel_size = 7
stem_stride = 2
stem_pool = true

# Optional expert override; four positive block sizes are required.
# block_config = [6, 12, 24, 16]""",
    },
    "efficientnet": {
        "summary": "Compound-scaled EfficientNet family for strong accuracy/compute tradeoffs.",
        "dropout_help": (
            "Classifier dropout; omit this line to use the selected variant's canonical value."
        ),
        "model": """\
# Valid variants: efficientnet_b0 through efficientnet_b7.
# B0 is the most economical default. Larger variants need progressively more
# memory and normally benefit from larger images (B0..B7 canonical resolutions
# are 224, 240, 260, 300, 380, 456, 528, and 600 respectively).
variant = "efficientnet_b0"
stem_kernel_size = 3
stem_stride = 2
se_ratio = 0.25

# Optional expert overrides. Omit them to use the selected variant's canonical
# width, depth, stem width, top width, and classifier dropout.
# width_coefficient = 1.0
# depth_coefficient = 1.0
# stem_filters = 32
# top_filters = 1280""",
    },
}

for _details in FAMILIES.values():
    _details.setdefault("normalize_embeddings", "true")


def render(family: str, details: dict[str, str]) -> str:
    # Keep all non-architecture settings byte-for-byte common across families.
    text = f"""\
# Oracle Builder V1 classification example: {family}
# {details["summary"]}
#
# This file is intentionally dataset-independent. Oracle Builder reads the
# class count, class order, labels, split assignments, and dataset identity from
# the input SQLite dataset. Do not add data.num_classes by hand.

[run]
task = "classification"
model = "{family}"
seed = 123
notes = "Documented {family} baseline"

[data]
# Classification defaults assume one-channel images. Increase height/width when
# fine detail matters, recognizing that memory use grows roughly with image area.
input_shape = [128, 128]
batch_size = 8
shuffle_buffer = 4096

# Split membership belongs to each model run, not the dataset. These fractions
# create deterministic assignments that are saved in the run artifact.
validation_split = 0.20
test_split = 0.10

[data.streaming]
# Keep encoded images in SQLite and decode only the active examples/batches.
enabled = true
reader_workers = 4
prefetch_batches = 2
deterministic = true
sqlite_cache_kib = 65536

[preprocessing]
# fit_pad preserves the full image; fill_crop fills the frame; stretch may
# distort geometry. Preprocessing is exported as part of the model contract.
resize_mode = "fit_pad"
normalization = "dtype"
rescale = true
invert = false
pad_value = 0.0
interpolation = "bilinear"
channel_mode = "grayscale"
percentile_low = 1.0
percentile_high = 99.0

[distribution]
# auto uses MirroredStrategy when multiple GPUs are visible, otherwise one
# device. Global data.batch_size must divide sensibly across all replicas.
strategy = "auto"
devices = []
cross_device_ops = "auto"
fallback_to_single = true
memory_growth = true

[model]
{details["model"]}

# Every classifier exposes a fixed-size penultimate embedding named "features",
# plus logits and softmax probabilities. 256 is a portable default for KNN.
embedding_dim = 256
normalize_embeddings = {details["normalize_embeddings"]}
# {details["dropout_help"]}
dropout = 0.20

[classification.stratification]
# Recommended shared-model recipe for size-correlated ROIs. It is opt-in: first
# verify support for every configured stratum, then set enabled = true.
# The scheduler interleaves optimizer steps across all strata, rather than
# letting one resolution overwrite the others in a contiguous training block.
enabled = false
dimensions = [32, 64, 128]
basis = "max_original_dimension"
assignment_policy = "smallest_fitting"
batch_size_policy = "constant_input_tensor"
weight_sharing = "shared"
schedule = "interleaved_steps"
steps_per_stratum = 1
# A global interleaved epoch completes before validation/checkpoint selection.
supra_epochs = 1
# GroupNorm remains stable when high-resolution microbatches are small. The
# canonical ResNet builder supports it; other family examples retain batch
# normalization until their builders gain an equivalent implementation.
normalization = "{'group' if family == 'resnet' else 'batch'}"
group_norm_groups = 8

[classification.stratification.conditioning]
# Give the shared model the selected resolution as a learned, soft cue. This
# does not mask classes that happen to be absent from one training stratum.
enabled = true
embedding_dim = 16

[classification.stratification.training_routing]
# Establish a canonical-routing baseline before introducing cross-stratum
# resizing as an augmentation.
enabled = false
adjacent_lower_probability = 0.0
seed = 123

[classification.stratification.cycle_scheduler]
# Equal stratum weighting and a per-stratum guardrail prevent a populous or
# easy bin from hiding catastrophic forgetting in another bin.
aggregation = "equal_strata"
guardrail_metric = "macro_f1"
max_stratum_drop = 0.03
reduce_lr_patience = 3
reduce_lr_factor = 0.5

[self_supervised]
# Self-supervised pretraining initializes this same architecture before labels
# are used. BYOL is less batch-size-sensitive. SimCLR uses in-batch negatives
# and generally benefits from larger batches; temperature applies to SimCLR.
enabled = false
method = "byol" # "byol", "student_teacher" (BYOL alias), or "simclr"
epochs = 30
learning_rate = 0.001
teacher_momentum = 0.99
temperature = 0.10
projection_dim = 128
projection_hidden_dim = 256
use_training_augmentation = true

[training]
epochs = 50
optimizer = "adam"
learning_rate = 0.0003
# AdamW-style decoupled regularization is supported by the classification
# trainer; this small value is a conservative first shared-model setting.
weight_decay = 0.0001

# Weighted cross entropy is the classification default. For already balanced
# datasets, ordinary sparse_categorical_crossentropy is the simpler alternative.
loss = "weighted_sparse_categorical_crossentropy"
metrics = ["accuracy", "macro_f1"]

[training.class_weights]
# Weights are calculated from the training split and saved in the run artifact.
# effective_number is a stable default; inverse_frequency is more aggressive.
mode = "effective_number" # effective_number, inverse_frequency, or explicit
beta = 0.999               # only used by effective_number; closer to 1 is gentler
normalize = true           # preserves an average training-example weight of 1
# For explicit mode, provide one weight per dataset class in class_index order:
# values = [1.0, 2.5, 0.75]

[evidence]
# Saves normalized embeddings, class prototypes, and KNN evidence alongside
# logits/probabilities. Disable only when smaller prediction artifacts matter.
enabled = true
knn_k = 5

[callbacks]
early_stopping = true
early_stopping_patience = 8
reduce_lr_on_plateau = true
checkpoint_monitor = "val_loss"

[recovery]
# Keep one full-state snapshot for restart after an interruption. This is not
# the optional archive of per-epoch checkpoints above.
enabled = true
save_every_epochs = 1

[augmentation]
# Conservative policy for a shared 32/64/128-pixel model. Apply stronger
# transforms only after visually validating them for the 32-pixel inputs.
enabled = true
repeats_per_epoch = 1
invert = false # Assume black background and light foreground.
rotation = 0.25
zoom = 0.10
translation = [0.075, 0.075]
skew = 0.10
flip_horizontal = true
flip_vertical = true
brightness = 0.10
contrast = 0.10
gaussian_noise = 0.02
fill_value = 0.0

[inference]
# auto selects a bounded batch size from input shape and memory budget. It may
# back off after an out-of-memory error without changing prediction semantics.
batch_size = "auto"
minimum_batch_size = 1
maximum_batch_size = 64
memory_budget_mb = 512
progress = true

[evaluation.benchmark]
# Runs a short synthetic forward-pass benchmark after dataset evaluation.
enabled = true
warmup_batches = 2
measured_batches = 10

[output]
# The final portable model is always saved; per-epoch checkpoints are opt-in.
save_checkpoints = false
save_predictions = true
save_figures = true
export_savedmodel = true
prediction_commit_batches = 20
"""
    return dedent(text)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate documented defaults for every classification family."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/classification_defaults"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for family, details in FAMILIES.items():
        (args.output / f"{family}.toml").write_text(
            render(family, details),
            encoding="utf-8",
        )
    print(f"Wrote {len(FAMILIES)} documented family configs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
