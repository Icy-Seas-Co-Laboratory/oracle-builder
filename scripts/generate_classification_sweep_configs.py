#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


MODELS = [
    "simple_cnn",
    "resnet_like",
    "densenet_like",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "densenet121",
    "densenet169",
    "densenet201",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
]


def render(model: str, pretrained: bool, batch_size: int) -> str:
    mode = "pretrained" if pretrained else "supervised"
    return f"""# Matched classification sweep: {model}, {mode}.
# All sweep configs use identical data, augmentation, and evaluation choices.

[run]
task = "classification"
model = "{model}"
seed = 123
notes = "Classification architecture sweep: {model}, {mode}"

[data]
# Grayscale is the default; use three channels only for genuinely color data.
input_shape = [224, 224, 1]
batch_size = {batch_size}
shuffle_buffer = 4096
validation_split = 0.20
test_split = 0.10

[data.streaming]
enabled = true
reader_workers = 4
prefetch_batches = 2
deterministic = true
sqlite_cache_kib = 65536

[preprocessing]
resize_mode = "fit_pad"
normalization = "dtype"
rescale = true
invert = false
pad_value = 0.0
interpolation = "bilinear"
channel_mode = "grayscale"

[distribution]
strategy = "auto"
devices = []
cross_device_ops = "auto"
fallback_to_single = true
memory_growth = true

[model]
# Standard families expose variants through run.model in sweep configs:
# ResNet-18/34/50/101/152, DenseNet-121/169/201, and EfficientNet-B0..B7.
embedding_dim = 256
normalize_embeddings = true
dropout = 0.20

[evidence]
enabled = true
knn_k = 5

[pretraining]
# BYOL is the robust default; SimCLR is an alternative for larger batches.
enabled = {str(pretrained).lower()}
method = "byol"
epochs = 30
learning_rate = 0.001
teacher_momentum = 0.99
projection_dim = 128
projection_hidden_dim = 256
use_training_augmentation = true

[training]
epochs = 50
optimizer = "adam"
learning_rate = 0.0003
loss = "weighted_sparse_categorical_crossentropy"
metrics = ["accuracy"]

[training.class_weights]
# effective_number is gentler than inverse-frequency weighting.
mode = "effective_number"
beta = 0.999
normalize = true

[callbacks]
early_stopping = true
early_stopping_patience = 8
reduce_lr_on_plateau = true
checkpoint_monitor = "val_loss"

[augmentation]
# Shared policy for fair comparisons. rotation=0.5 permits any orientation.
enabled = true
repeats_per_epoch = 1
invert = false # Assume black background and light foreground.
rotation = 0.5
zoom = 0.20
translation = [0.15, 0.15]
skew = 0.20
flip_horizontal = true
flip_vertical = true
brightness = 0.20
contrast = 0.20
gaussian_noise = 0.05
fill_value = 0.0

[output]
save_checkpoints = false
save_predictions = true
save_figures = true
export_savedmodel = true
prediction_commit_batches = 20
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate matched classification sweep configs.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        for pretrained in (False, True):
            mode = "pretrained" if pretrained else "supervised"
            path = args.output / f"{model}_{mode}.toml"
            path.write_text(render(model, pretrained, args.batch_size))
    print(f"Wrote {len(MODELS) * 2} configs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
