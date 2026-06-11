# oracle-builder

oracle-builder is a small internal machine-learning experimentation tool for TensorFlow/Keras CNN work. It trains, evaluates, saves, reloads, and inspects runs using ordinary local files: TOML, JSON, CSV, SQLite, and Keras/TensorFlow model artifacts.

It is intentionally not a full ML platform. It does not require MLflow, Weights & Biases, DVC, Hydra, Airflow, Docker, or a database server.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Dataset Schema

Input datasets are SQLite files with a `samples` table:

```sql
CREATE TABLE samples (
    uuid TEXT PRIMARY KEY,
    split TEXT,
    input_blob BLOB,
    input_blob_encoding TEXT,
    input_blob_dimensions TEXT,
    input_aux_blob BLOB,
    input_aux_blob_encoding TEXT,
    input_aux_blob_dimensions TEXT,
    output_blob BLOB,
    output_blob_encoding TEXT,
    output_blob_dimensions TEXT,
    label_text TEXT,
    sample_weight REAL,
    metadata_json TEXT
);
```

Supported encodings currently include `utf-8`, `json`, `int`, `float`, `png`, and `npy`. `zstd` is stubbed with a clear error until the optional dependency is added. Older databases without the optional `input_aux_*` columns are migrated by `mask_builder.py` when opened.

Create example datasets:

```bash
python3 -m oracle_builder.data.sqlite_dataset --classification datasets/example_classification.sqlite
python3 -m oracle_builder.data.sqlite_dataset --segmentation datasets/example_segmentation.sqlite
```

## Training

```bash
python3 ./model_training.py \
  -c ./configs/example_classification.toml \
  -i ./datasets/example_classification.sqlite \
  -o example-classification-run
```

Useful flags:

```bash
--runs-dir ./runs
--overwrite
--dry-run
--debug
```

`--resume` is reserved but not implemented yet.

## Evaluation

```bash
python3 ./model_evaluate.py \
  --run ./runs/example-classification-run \
  --input ./datasets/example_classification.sqlite \
  --split test
```

## Inference

```bash
python3 ./model_inference.py \
  --run ./runs/example-classification-run \
  --input ./datasets/example_classification.sqlite \
  --output ./runs/example-classification-run/predictions/predictions.sqlite
```

## Mask Builder

oracle-builder includes a small local mask builder for U-Net training data. It can load an image from a SQLite dataset or from a local image file, generate an initial mask from thresholding, allow manual correction in napari, validate the mask, and save it back into the standardized SQLite schema.

Install GUI dependencies separately from the base training stack:

```bash
python3 -m pip install -r requirements-gui.txt
```

Launch from a SQLite sample:

```bash
python3 mask_builder.py \
  --database ./datasets/dataset3.sqlite \
  --uuid sample-001
```

Launch from a local image and save it into a dataset:

```bash
python3 mask_builder.py \
  --image ./images/sample_001.png \
  --database ./datasets/dataset3.sqlite \
  --uuid sample_001
```

Browse samples with missing masks:

```bash
python3 mask_builder.py \
  --database ./datasets/dataset3.sqlite \
  --missing-masks-only
```

Load a detection ROI and optional mask from the Pelagia REST API, then save edits into SQLite:

```bash
python3 mask_builder.py \
  --api-base-url http://localhost:8000 \
  --detection-id detection-001 \
  --database ./datasets/dataset3.sqlite
```

List available Pelagia detection/ROI ids:

```bash
python3 mask_builder.py \
  --api-base-url http://localhost:8000 \
  --list-api-rois
```

Any Pelagia `/detections` filter can be supplied as a CLI option, for example:

```bash
python3 mask_builder.py \
  --list-api-rois \
  --asset-id ASSET_ID \
  --min-area 500 \
  --max-bbox-w 120 \
  --limit 50
```

Open a random Pelagia detection/ROI:

```bash
python3 mask_builder.py \
  --api-base-url http://localhost:8000 \
  --random-api-roi \
  --database ./datasets/dataset3.sqlite
```

Open the first matching Pelagia detection/ROI as a queue. In this mode, `Save and next` fetches the next ROI from the API:

```bash
python3 mask_builder.py \
  --api-base-url http://localhost:8000 \
  --api-browse-rois \
  --api-asset-id ASSET_ID \
  --database ./datasets/dataset3.sqlite
```

Useful filters include `--run-id`, `--asset-id`, `--collection`, `--frame-id`, `--start-frame`, `--end-frame`, `--roi-index`, bbox filters such as `--min-bbox-x` and `--max-bbox-w`, area/perimeter filters such as `--min-area 500`, ROI/mask payload filters such as `--roi-encoding`, `--roi-format`, `--api-mask-encoding`, and paging/sorting filters such as `--limit`, `--offset`, `--sort-by`, and `--sort-dir`.

By default the API loader uses Pelagia detection endpoints:

- `GET /detections/{detection_id}` for metadata
- `GET /detections/{detection_id}/roi?format=png` for the ROI image
- `GET /detections/{detection_id}/mask?format=png` for an optional initial mask

The ROI/mask endpoints may return PNG/JPEG bytes or Pelagia’s JSON matrix response with `dtype`, `shape`, and `data`. If your API uses a different route, set `--api-endpoint-template` to use the generic JSON loader instead:

```bash
python3 mask_builder.py \
  --api-base-url http://localhost:8000 \
  --api-roi-id roi-001 \
  --api-endpoint-template "/api/rois/{roi_id}" \
  --database ./datasets/dataset3.sqlite
```

For Pelagia-loaded images, the detection id is preserved as the SQLite `samples.uuid` value and also stored in `metadata_json` as `pelagia_detection_id` with the full Pelagia detection metadata under `metadata_json.pelagia`.

You can also pass a folder to `--image` or use an image folder with legacy `--input`. The mask builder opens the first supported image and `Save and next` advances through the folder:

```bash
python3 mask_builder.py \
  --image ./images/to_mask \
  --database ./datasets/dataset3.sqlite
```

The viewer uses napari labels layers for manual editing. API-provided masks are loaded into a `candidate mask` layer, hidden by default, and copied into the editable `validated mask` layer. Paint foreground in `validated mask` with label `1`, erase with label `0`, adjust brush size with napari’s normal label controls, and use the side panel for thresholding, image inversion before thresholding, simple cleanup, black/white viewer background toggling, validation, saving, `Save and next`, and `Skip`. `Skip` advances to the next queued ROI or image without writing a mask.

Mask storage is intentionally redundant:

- `samples.input_blob` stores the model input. For API samples with a candidate mask this is an `npy` tensor with shape `[height, width, 2]`: channel 0 is the grayscale ROI and channel 1 is the candidate mask.
- `samples.input_aux_blob` stores the candidate mask separately for viewer round-tripping.
- `samples.output_blob` stores the current accepted validated mask with shape `[height, width, 1]`.
- `mask_annotations` stores append-only annotation history.

Each save inserts a new `mask_annotations` row. Accepted saves update `samples.output_blob`, `samples.output_blob_encoding`, and `samples.output_blob_dimensions`. Existing metadata is preserved and merged with mask-builder provenance.
Pelagia ROI crops can have different raw sizes; during segmentation training, `model_training.py` resizes inputs and validated masks to the configured `data.input_shape` and `data.output_shape`.

Mask validation checks for empty masks, NaN/Inf values, binary labels, image/mask dimension mismatch, foreground fraction, connected components, and whether foreground touches the border. Warnings are shown in the GUI. In this first version, invalid masks are blocked from saving.

To check that accepted masks are ready for U-Net training:

```bash
python3 mask_builder.py \
  --database ./datasets/dataset3.sqlite \
  --validate-unet-dataset
```

To generate a U-Net training config from the accepted masks:

```bash
python3 mask_builder.py \
  --database ./datasets/dataset3.sqlite \
  --write-unet-config ./configs/dataset3_unet.toml \
  --unet-batch-size 8 \
  --unet-epochs 20
```

Run the same compatibility check from the training CLI before starting a run:

```bash
python3 model_training.py \
  --config ./configs/dataset3_unet.toml \
  --input ./datasets/dataset3.sqlite \
  --output dataset3-unet-run \
  --preflight
```

Then train normally by removing `--preflight`.

To make a contact sheet of the training ROIs with the API candidate mask in transparent blue and the validated mask in transparent green:

```bash
python3 scripts/visualize_unet_training_rois.py \
  --database ./datasets/dataset3.sqlite \
  --config ./configs/dataset3_unet.toml \
  --split train \
  --output ./runs/dataset3-unet-run/training_roi_overview.png
```

Known limitations: this is not a full annotation platform, batch browsing is intentionally simple, PNG masks are limited to 2D binary masks, and there is no polygon editing, cloud sync, multi-user review, or AI-assisted segmentation.

## Run Outputs

Each run is self-contained under `runs/<run_name>/`. Common outputs include:

- `run_config.toml`: original user config
- `resolved_config.json`: config with defaults and resolved paths
- `run_metadata.json`: run id, status, and summary
- `environment.json`: Python/TensorFlow/Keras/NumPy/platform/GPU/git metadata
- `requirements_freeze.txt`: package snapshot
- `training_log.sqlite`: run, epoch metrics, and event tables
- `metrics.csv` and `metrics.json`: training history
- `model/`: model artifacts and load test report
- `evaluation/`: task-specific evaluation files
- `predictions/predictions.sqlite`: prediction records
- `figures/`: plots where available

## Model Portability Strategy

oracle-builder saves models in multiple ways:

- `model/final.keras` for full Keras reloads
- `model/weights.weights.h5` for rebuild-and-load workflows
- `model/export_savedmodel/` for inference-oriented TensorFlow export

This redundancy is intentional. Keras serialization, TensorFlow export behavior, and custom model code can change over time, so every completed run immediately performs a reload/prediction check and writes `model/load_test_report.json`.

## Models

The initial registry supports:

- `simple_cnn`
- `unet`
- `resnet_like`
- `densenet_like`

Each model module exposes `build_model(config: dict)`. Add future models in `models/` and register them in `oracle_builder/registry.py`.

## R and Python Analysis

Python helpers:

```bash
python3 analysis/python/inspect_run.py runs/example-classification-run
python3 analysis/python/compare_runs.py runs/run-a runs/run-b
```

R helpers use `DBI`, `RSQLite`, `jsonlite`, and `ggplot2`:

```bash
Rscript analysis/R/inspect_run.R runs/example-classification-run
Rscript analysis/R/compare_runs.R runs/run-a runs/run-b
```

## Known Limitations

- Resume support is not implemented.
- Segmentation evaluation assumes binary masks and thresholded predictions.
- SavedModel loading is inference-only.
- The SQLite loader currently materializes arrays in memory before building `tf.data.Dataset`.
- Custom layers/losses should be explicitly Keras-serializable before relying on `final.keras`.
- The mask builder requires optional GUI dependencies and does not automate napari GUI testing.
