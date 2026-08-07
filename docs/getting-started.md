# Getting started

## Install and verify

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
python3 scripts/check_tensorflow_devices.py
```

For NVIDIA Linux/WSL or Apple Metal, install the corresponding
`requirements-gpu-*.txt` file before verification.

## Run a small classification example

```bash
python3 -m oracle_builder.data.sqlite_dataset \
  --classification datasets/example_classification.sqlite

oracle-dataset checkpoint \
  datasets/example_classification.sqlite \
  --output datasets/example_classification.training.sqlite

python3 model_training.py \
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
