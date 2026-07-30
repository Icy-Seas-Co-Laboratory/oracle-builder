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
    return f"""[run]
task = "classification"
model = "{model}"
seed = 123
notes = "Classification architecture sweep: {model}, {mode}"

[data]
input_shape = [224, 224, 3]
batch_size = {batch_size}
shuffle_buffer = 4096
validation_split = 0.20
test_split = 0.10

[preprocessing]
resize_mode = "fit_pad"
normalization = "dtype"
rescale = true
invert = false
pad_value = 0.0
interpolation = "bilinear"
channel_mode = "rgb"

[model]
embedding_dim = 256
normalize_embeddings = true
dropout = 0.20

[evidence]
enabled = true
knn_k = 5

[pretraining]
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
loss = "sparse_categorical_crossentropy"
metrics = ["accuracy"]

[callbacks]
early_stopping = true
early_stopping_patience = 8
reduce_lr_on_plateau = true
checkpoint_monitor = "val_loss"

[augmentation]
enabled = true
repeats_per_epoch = 1
rotation = 0.08
zoom = 0.25
translation = 0.15
skew = 0.10
flip_horizontal = true
flip_vertical = true
brightness = 0.25
contrast = 0.25
gaussian_noise = 0.05
fill_value = 0.0

[output]
save_checkpoints = true
save_predictions = true
save_figures = true
export_savedmodel = true
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
