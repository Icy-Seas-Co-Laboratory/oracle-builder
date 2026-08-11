# Getting started

## Install and verify

```bash
uv sync
uv run python scripts/check_tensorflow_devices.py
```

For NVIDIA Linux/WSL or Apple Metal, synchronize the corresponding extra before
verification, for example `uv sync --extra gpu-macos`. The available extras are
`gpu-linux`, `gpu-wsl2`, and `gpu-macos`.

## Run a small classification example

```bash
uv run python -m oracle_builder.data.sqlite_dataset \
  --classification datasets/example_classification.sqlite

uv run oracle-dataset checkpoint \
  datasets/example_classification.sqlite \
  --output datasets/example_classification.training.sqlite

uv run python model_training.py \
  --config configs/example_classification.toml \
  --input datasets/example_classification.training.sqlite \
  --output example-classification-run \
  --overwrite
```

The checkpoint is a frozen dataset revision; training intentionally refuses a
working dataset. The resulting run is a sealed artifact in
`runs/example-classification-run` unless another runs directory is selected.

Next: [classification workflow](classification-workflow.md),
[mask-refinement workflow](mask-refinement-workflow.md), or
[external model products](model-products.md).
