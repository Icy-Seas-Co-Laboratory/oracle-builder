# Mask-refinement workflow

This workflow builds a `mask_refinement` SQLite dataset from local images or
Pelagia ROIs, edits validated masks in napari, then trains a U-Net-family model.

## 1. Install the editor and create/edit a dataset

```bash
python3 -m pip install -r requirements-gui.txt

python3 mask_builder.py \
  --image images/to_mask \
  --database datasets/unet_training.sqlite
```

For a Pelagia ROI, provide an API URL and a detection/ROI ID:

```bash
python3 mask_builder.py \
  --api-base-url http://localhost:8000 \
  --detection-id DETECTION_UUID \
  --database datasets/unet_training.sqlite
```

The editor’s `validated mask` layer is the editable result. `Save and next`,
`Skip`, and the thumbnail navigator support queue review and revisiting saved
ROIs. Candidate masks and annotation history are retained for audit.

Legacy ROI databases are automatically backed up and migrated to Dataset V1
when opened. Details are in the [dataset schema reference](dataset-schema-v1.md).

For an active review workspace, save a complete immutable milestone without
locking the editable source:

```bash
oracle-dataset snapshot workspaces/roi-review.sqlite \
  --output snapshots/roi-review-milestone.sqlite \
  --note "Completed first-pass review"
```

## 2. Validate, configure, and freeze

```bash
python3 mask_builder.py \
  --database datasets/unet_training.sqlite \
  --validate-unet-dataset \
  --unet-input-shape 256,256,2 \
  --unet-output-shape 256,256,1

python3 mask_builder.py \
  --database datasets/unet_training.sqlite \
  --write-unet-config configs/unet_training.toml \
  --unet-input-shape 256,256,2 \
  --unet-output-shape 256,256,1

oracle-dataset checkpoint datasets/unet_training.sqlite \
  --output datasets/unet_training.v1.sqlite
```

If this is an active annotation workspace rather than a direct training
database, create a release instead. It retains human curation but excludes
derived model evidence:

```bash
oracle-dataset release-training workspaces/roi-review.sqlite \
  datasets/unet_training.v1.sqlite --name "unet-training-v1"
```

Two-channel inputs conventionally contain ROI intensity then candidate mask.
Set `data.candidate_sdf = true` to include a signed-distance candidate channel.

## 3. Train

```bash
python3 model_training.py \
  --config configs/unet_training.toml \
  --input datasets/unet_training.v1.sqlite \
  --output unet-v1
```

Supported architectures are `unet`, `residual_unet`/`resunet`, and
`unet_plus_plus`/`unetpp`. The loss combines BCE and soft DICE; final validation
selects the probability threshold that optimizes DICE. Choose either
`validated_mask` or `candidate_delta` target mode.

## Large ROIs and tiling

When an ROI exceeds model input dimensions, enable tiling. Tiles overlap by the
configured fraction and are reassembled with uniform or Hann weighting.

```toml
[tiling]
enabled = true
overlap_fraction = 0.5
blend_mode = "hann"
tile_large_rois_only = true
```

## 4. Inspect results

```bash
python3 scripts/visualize_unet_training_rois.py \
  --database datasets/unet_training.v1.sqlite \
  --run runs/unet-v1 \
  --split test \
  --side-by-side \
  --predictions predictions/unet-v1.sqlite \
  --prediction-set unet-v1 \
  --output figures/unet-v1-comparison.png
```

The side-by-side view shows candidate/original, prediction, and validated-mask
overlays for the same ROI. Prediction outputs retain every ROI plus the run-owned
train/validation/test tag.
