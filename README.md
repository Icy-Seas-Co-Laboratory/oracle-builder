# oracle-builder

oracle-builder is a small file-based TensorFlow/Keras experimentation tool. It trains, evaluates, saves, reloads, and inspects CNN runs using local files: TOML configs, SQLite datasets, JSON/CSV summaries, and Keras/TensorFlow model artifacts.

It is intentionally not a full ML platform. It does not require MLflow, Weights & Biases, DVC, Hydra, Airflow, Docker, or a database server.

## Quick Start

Install the base training stack:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

For GPU acceleration, prefer the platform-specific install profile after creating the virtual environment:

```bash
# Linux with NVIDIA CUDA
python3 -m pip install -r requirements-gpu-linux.txt

# Windows with NVIDIA CUDA through WSL2
python3 -m pip install -r requirements-gpu-wsl2.txt

# macOS with Apple Metal
python3 -m pip install -r requirements-gpu-macos.txt
```

Native Windows GPU support is not available for modern TensorFlow releases; use WSL2 for NVIDIA GPU training. TensorFlow's official pip guide recommends `tensorflow[and-cuda]` for Linux and WSL2 CUDA installs. Apple provides `tensorflow-metal` for Mac GPU acceleration. The macOS profile pins TensorFlow to the 2.18 line because the current `tensorflow-metal` compatibility table documents support through TensorFlow 2.18.

Verify device visibility:

```bash
python3 scripts/check_tensorflow_devices.py
```

Create and train a synthetic classification example:

```bash
python3 -m oracle_builder.data.sqlite_dataset \
  --classification datasets/example_classification.sqlite

python3 model_training.py \
  --config configs/example_classification.toml \
  --input datasets/example_classification.sqlite \
  --output example-classification-run \
  --overwrite
```

Evaluate and write predictions:

```bash
python3 model_evaluate.py \
  --run runs/example-classification-run \
  --input datasets/example_classification.sqlite \
  --split test

python3 model_inference.py \
  --run runs/example-classification-run \
  --input datasets/example_classification.sqlite \
  --output runs/example-classification-run/predictions/predictions.sqlite
```

## U-Net Mask Workflow

For GUI mask editing, install the optional GUI stack:

```bash
python3 -m pip install -r requirements-gui.txt
```

The usual U-Net workflow is:

1. Load ROIs from Pelagia or local images.
2. Review/edit the mask in napari.
3. Save accepted masks into a SQLite dataset.
4. Validate the dataset.
5. Train U-Net from the validated masks.
6. Inspect a contact sheet of the training ROIs and masks.

### Build A Dataset From Pelagia

List matching Pelagia detection/ROI ids:

```bash
python3 mask_builder.py \
  --api-base-url http://localhost:8000 \
  --list-api-rois \
  --min-area 500 \
  --limit 50
```

Open a random matching ROI and save edits to a SQLite dataset:

```bash
python3 mask_builder.py \
  --api-base-url http://localhost:8000 \
  --min-area 500 \
  --limit 100 \
  --database datasets/unet_training.sqlite \
  --random-api-roi
```

Browse a queue of matching ROIs. In this mode, `Save and next` saves the current validated mask and fetches the next ROI; `Skip` advances without saving:

```bash
python3 mask_builder.py \
  --api-base-url http://localhost:8000 \
  --api-browse-rois \
  --min-area 500 \
  --limit 100 \
  --database datasets/unet_training.sqlite
```

Open a specific Pelagia detection/ROI:

```bash
python3 mask_builder.py \
  --api-base-url http://localhost:8000 \
  --detection-id DETECTION_UUID \
  --database datasets/unet_training.sqlite
```

Useful Pelagia filters include `--run-id`, `--asset-id`, `--collection`, `--frame-id`, `--start-frame`, `--end-frame`, `--roi-index`, bbox filters such as `--min-bbox-x` and `--max-bbox-w`, area/perimeter filters such as `--min-area 500`, payload filters such as `--roi-encoding`, `--roi-format`, `--api-mask-encoding`, `--mask-format`, and paging/sorting filters such as `--limit`, `--offset`, `--sort-by`, and `--sort-dir`.

By default the Pelagia loader uses:

- `GET /detections/{detection_id}` for metadata
- `GET /detections/{detection_id}/roi?format=png` for the ROI image
- `GET /detections/{detection_id}/mask?format=png` for the candidate mask

If an API uses a different route, set `--api-endpoint-template` and `--api-roi-id`:

```bash
python3 mask_builder.py \
  --api-base-url http://localhost:8000 \
  --api-roi-id ROI_ID \
  --api-endpoint-template "/api/rois/{roi_id}" \
  --database datasets/unet_training.sqlite
```

### Build A Dataset From Local Images

Open one image and save masks into a dataset:

```bash
python3 mask_builder.py \
  --image images/sample_001.png \
  --database datasets/unet_training.sqlite \
  --uuid sample_001
```

Open a folder as a queue. `Save and next` advances through supported image files:

```bash
python3 mask_builder.py \
  --image images/to_mask \
  --database datasets/unet_training.sqlite
```

### Napari Mask Editing

The viewer has three important layers:

- `image`: the ROI image.
- `candidate mask`: the mask provided by the API, shown as a labels layer and hidden by default.
- `validated mask`: the editable manual result, initialized from the candidate mask.

Paint foreground in `validated mask` with label `1`; erase with label `0`. The side panel includes thresholding, image inversion before thresholding, cleanup actions, black/white background switching, validation, `Save`, `Save and next`, and `Skip`.

Invalid masks are blocked from saving. Validation checks for empty masks, NaN/Inf values, binary labels, image/mask dimension mismatch, foreground fraction, connected components, and foreground touching the border.

### Validate And Train U-Net

Check that accepted masks are trainable:

```bash
python3 mask_builder.py \
  --database datasets/unet_training.sqlite \
  --validate-unet-dataset \
  --unet-input-shape 256,256,2 \
  --unet-output-shape 256,256,1
```

Generate a config from the accepted masks:

```bash
python3 mask_builder.py \
  --database datasets/unet_training.sqlite \
  --write-unet-config configs/unet_training.toml \
  --unet-input-shape 256,256,2 \
  --unet-output-shape 256,256,1 \
  --unet-batch-size 8 \
  --unet-epochs 20
```

Run a training preflight using the config. This validates the SQLite dataset and exits without creating a run directory:

```bash
python3 model_training.py \
  --config configs/unet_training.toml \
  --input datasets/unet_training.sqlite \
  --output unet-test \
  --preflight
```

Train:

```bash
python3 model_training.py \
  --config configs/unet_training.toml \
  --input datasets/unet_training.sqlite \
  --output unet-test \
  --overwrite
```

The example U-Net config at `configs/example_segmentation_unet.toml` expects two-channel inputs:

```toml
[data]
input_shape = [256, 256, 2]
output_shape = [256, 256, 1]
```

Pelagia ROI crops can have different raw sizes. During segmentation training, oracle-builder resizes inputs to `data.input_shape` and validated masks to `data.output_shape`. The ROI channel uses bilinear resizing; candidate and validated mask channels use nearest-neighbor resizing.

### Visualize Training ROIs

Create a contact sheet of the selected split. Candidate/API masks are transparent blue; validated/refined masks are transparent green:

```bash
python3 scripts/visualize_unet_training_rois.py \
  --database datasets/unet_training.sqlite \
  --config configs/unet_training.toml \
  --split train \
  --output runs/unet-test/training_roi_overview.png
```

Useful options:

```bash
--split train
--thumbnail-size 180
--columns 0
--limit 100
--candidate-alpha 0.35
--refined-alpha 0.45
--no-labels
```

## SQLite Dataset Format

Datasets are SQLite files with a `samples` table. Core columns are:

```sql
CREATE TABLE samples (
    uuid TEXT PRIMARY KEY,
    split TEXT,
    input_blob BLOB,
    input_blob_encoding TEXT,
    input_blob_dimensions TEXT,
    output_blob BLOB,
    output_blob_encoding TEXT,
    output_blob_dimensions TEXT,
    label_text TEXT,
    sample_weight REAL,
    metadata_json TEXT
);
```

The mask builder also migrates datasets to include optional candidate-mask columns:

```sql
input_aux_blob BLOB,
input_aux_blob_encoding TEXT,
input_aux_blob_dimensions TEXT
```

Supported blob encodings include `utf-8`, `json`, `int`, `float`, `png`, `jpg`, `jpeg`, `tif`, `tiff`, and `npy`. `zstd` currently raises a clear error until optional support is added.

For mask-builder U-Net datasets:

- `samples.input_blob` stores the model input. For API samples with a candidate mask, this is an `npy` tensor shaped `[height, width, 2]`: channel 0 is grayscale ROI, channel 1 is candidate mask.
- `samples.input_aux_blob` stores the candidate mask separately for viewer round-tripping.
- `samples.output_blob` stores the accepted validated mask shaped `[height, width, 1]`.
- `mask_annotations` stores append-only annotation history.

Pelagia detection ids are preserved as `samples.uuid` values unless `--uuid` is explicitly provided. Pelagia metadata is stored in `metadata_json`, including `pelagia_detection_id` and the full detection metadata under `metadata_json.pelagia`.

Splits can be stored explicitly in `samples.split`. Rows without a split are assigned deterministically from the config seed and `validation_split`/`test_split`.

## Config Files

Configs are TOML files. Required sections are `[run]`, `[data]`, and `[training]`.

For classification:

```toml
[run]
task = "classification"
model = "simple_cnn"

[data]
input_shape = [128, 128, 3]
num_classes = 3

[training]
loss = "sparse_categorical_crossentropy"
```

For segmentation:

```toml
[run]
task = "segmentation"
model = "unet"

[data]
input_shape = [256, 256, 2]
output_shape = [256, 256, 1]

[training]
loss = "binary_crossentropy"
metrics = ["accuracy", "dice", "iou"]
```

Default values are merged in from `oracle_builder/config.py`. Common settings include `batch_size`, `shuffle_buffer`, `validation_split`, `test_split`, `epochs`, `optimizer`, `learning_rate`, callback settings, and output toggles.

## Input Augmentation

Input augmentation is configured in TOML under `[augmentation]` and is applied only to the training split. Validation and test splits are never augmented.

```toml
[augmentation]
enabled = true
invert = true
rotation = 0.05
zoom = 0.25
translation = [0.25, 0.25]
skew = 0.05
flip_horizontal = true
flip_vertical = true
brightness = 0.25
contrast = 0.25
gaussian_noise = 0.10
fill_value = 0.0
mask_fill_value = 0.0
```

Geometric augmentation includes rotation, zoom, translation, skew/shear, and horizontal/vertical flips. For segmentation, the same geometric transform is applied to both the model input and the validated output mask. Image-like input channels use bilinear interpolation; masks use nearest-neighbor interpolation and are re-binarized.

Photometric augmentation includes inversion, brightness, contrast, and Gaussian noise. These affect only image-like input channels, not output masks.

For two-channel U-Net inputs shaped `[height, width, 2]`, oracle-builder assumes channel 0 is the ROI image and channel 1 is the candidate/API mask. By default:

```toml
photometric_channels = [0]
mask_input_channels = [1]
```

For ordinary RGB classification inputs, photometric augmentation defaults to all channels. Override either list if your channel layout is different.

Augmentation values are fractions:

- `rotation = 0.05` samples rotations from +/- 5% of a full turn.
- `zoom = 0.25` samples scale from 0.75x to 1.25x.
- `translation = [0.25, 0.25]` samples height and width shifts up to +/- 25%.
- `skew = 0.05` samples horizontal shear from +/- 0.05.

## Training CLI

```bash
python3 model_training.py \
  --config CONFIG.toml \
  --input DATASET.sqlite \
  --output RUN_NAME
```

Useful flags:

```bash
--runs-dir runs
--overwrite
--dry-run
--preflight
--debug
```

`--resume` is reserved but not implemented.

## Evaluation And Inference

Evaluate a saved run:

```bash
python3 model_evaluate.py \
  --run runs/RUN_NAME \
  --input DATASET.sqlite \
  --split test
```

Write predictions from a saved run:

```bash
python3 model_inference.py \
  --run runs/RUN_NAME \
  --input DATASET.sqlite \
  --split test \
  --output runs/RUN_NAME/predictions/predictions.sqlite
```

## Run Outputs

Each training run is self-contained under `runs/<run_name>/`. Common outputs include:

- `run_config.toml`: original user config.
- `resolved_config.json`: config after defaults and path resolution.
- `run_metadata.json`: run id, status, and evaluation summary or error.
- `environment.json`: Python, TensorFlow, Keras, NumPy, platform, GPU, and git metadata.
- `requirements_freeze.txt`: package snapshot.
- `training_log.sqlite`: run, epoch metrics, and event tables.
- `metrics.csv` and `metrics.json`: training history.
- `model/`: model artifacts and load test report.
- `evaluation/`: task-specific evaluation files.
- `predictions/predictions.sqlite`: prediction records.
- `figures/`: plots where available.

## Model Portability

oracle-builder saves models in several ways:

- `model/final.keras` for full Keras reloads.
- `model/weights.weights.h5` for rebuild-and-load workflows.
- `model/export_savedmodel/` for inference-oriented TensorFlow export.

Every completed run immediately performs a reload/prediction check and writes `model/load_test_report.json`.

## Models

The registry currently supports:

- `simple_cnn`
- `unet`
- `resnet_like`
- `densenet_like`

Each model module exposes `build_model(config: dict)`. Add future models in `models/` and register them in `oracle_builder/registry.py`.

## Analysis Helpers

Python helpers:

```bash
python3 analysis/python/inspect_run.py runs/RUN_NAME
python3 analysis/python/compare_runs.py runs/run-a runs/run-b
```

R helpers use `DBI`, `RSQLite`, `jsonlite`, and `ggplot2`:

```bash
Rscript analysis/R/inspect_run.R runs/RUN_NAME
Rscript analysis/R/compare_runs.R runs/run-a runs/run-b
```

## Troubleshooting

`ValueError: Dataset must contain or create a train split`

This usually means no accepted masks exist for training, or all train rows failed to load. Run:

```bash
python3 model_training.py \
  --config CONFIG.toml \
  --input DATASET.sqlite \
  --output preflight-check \
  --preflight
```

For mask-builder datasets, also run:

```bash
python3 mask_builder.py \
  --database DATASET.sqlite \
  --validate-unet-dataset
```

`sqlite3.DatabaseError: file is not a database`

Check that `--database` points to a SQLite file, not an image. For image imports, use:

```bash
python3 mask_builder.py \
  --image path/to/image.png \
  --database datasets/masks.sqlite
```

`ModuleNotFoundError` for `napari`, `qtpy`, `skimage`, or `scipy`

Install GUI dependencies:

```bash
python3 -m pip install -r requirements-gui.txt
```

`ModuleNotFoundError` for `sklearn`

Install base dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Matplotlib cache warnings on locked-down systems can be avoided by pointing `MPLCONFIGDIR` at a writable folder:

```bash
mkdir -p .cache/matplotlib
MPLCONFIGDIR=.cache/matplotlib python3 model_evaluate.py --help
```

TensorFlow does not list a GPU

Run:

```bash
python3 scripts/check_tensorflow_devices.py
```

On Linux or Windows WSL2, install `requirements-gpu-linux.txt` or `requirements-gpu-wsl2.txt` and make sure the NVIDIA driver is visible with `nvidia-smi`. On macOS, install `requirements-gpu-macos.txt`; `tensorflow-metal` uses Apple's TensorFlow PluggableDevice support.

`tensorflow-metal` fails to load `_pywrap_tensorflow_internal.so`

This usually means `tensorflow-metal` is installed next to a TensorFlow version it does not support. Repair the macOS GPU environment with:

```bash
python3 -m pip uninstall -y tensorflow tensorflow-metal keras
python3 -m pip install -r requirements-gpu-macos.txt
python3 scripts/check_tensorflow_devices.py
```

At the time of writing, this installs TensorFlow `>=2.18,<2.19` with `tensorflow-metal==1.2.0`.

## Known Limitations

- Resume support is not implemented.
- Segmentation evaluation assumes binary masks and thresholded predictions.
- SavedModel loading is inference-only.
- The SQLite loader currently materializes arrays in memory before building `tf.data.Dataset`.
- The mask builder requires optional GUI dependencies and does not automate napari GUI testing.
- The U-Net loader resizes variable-size ROIs to the configured shape; it does not pad or preserve aspect ratio.
