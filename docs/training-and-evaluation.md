# Training, evaluation, and recovery

This guide covers choices shared by classification and mask-refinement runs.

## Training command

```bash
python3 model_training.py \
  --config CONFIG.toml \
  --input DATASET.sqlite \
  --output RUN_NAME
```

Use `--preflight` to validate without creating a run, `--dry-run` to inspect the
resolved configuration, and `--resume runs/RUN_NAME` to continue an interrupted
or failed run from its validated recovery snapshot.

## Configuration essentials

Every config has `[run]`, `[data]`, and `[training]`. Use the checked-in
examples as the source of valid model-specific settings.

```toml
[augmentation]
enabled = true
repeats_per_epoch = 1
invert = false
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
```

Augmentation applies only to the training split. Geometric transforms are shared
between images and masks; photometric transforms never change masks.

Classification defaults use `weighted_sparse_categorical_crossentropy`. Class
weights are calculated from the run’s training assignment only. Student–teacher
BYOL or SimCLR pretraining uses `[pretraining]`; it is available for every
classification family, not just ResNet.

## Streaming and multiple GPUs

Classification SQLite loading is streaming by default: images are decoded and
prepared lazily, batches are prefetched in bounded memory, and only lightweight
row metadata is indexed.

```toml
[distribution]
strategy = "auto" # auto, single, mirrored, cpu
fallback_to_single = true
memory_growth = true
```

When more than one GPU is visible, `auto` selects `MirroredStrategy`. Batch size
is global and must divide evenly across replicas.

## Evaluate and infer

```bash
python3 model_evaluate.py \
  --run runs/RUN_NAME --input DATASET.sqlite --split test \
  --output evaluations/RUN_NAME-test

python3 model_inference.py \
  --run runs/RUN_NAME --input DATASET.sqlite --split all \
  --prediction-set RUN_NAME --output predictions/RUN_NAME.sqlite
```

Evaluation reports classification accuracy, balanced accuracy, F1, calibration,
per-class metrics, and scalable confusion-matrix products. Segmentation reports
DICE, specificity, pixel accuracy, and threshold optimization. Inference batch
size defaults to `auto`: it is selected and verified on the inference host.

## Recovery and artifacts

One rolling full-Keras recovery snapshot is written after each epoch by default,
including optimizer state. Resume validates the dataset fingerprint, split
manifest, model/training contract, and snapshot checksum.

Completed, failed, and interrupted runs are sealed artifacts. Read the
[run-artifact reference](run-artifact-v1.md) for layout, integrity, packing, and
portable/runtime boundaries.
