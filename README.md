# oracle-builder

oracle-builder is a small file-based TensorFlow/Keras experimentation tool. It trains, evaluates, saves, reloads, and inspects CNN runs using local files: TOML configs, SQLite datasets, JSON/CSV summaries, and Keras/TensorFlow model artifacts.

It is intentionally not a full ML platform. It does not require MLflow, Weights & Biases, DVC, Hydra, Airflow, Docker, or a database server.

## Dataset architecture V1

Datasets use a versioned, use-case-specific schema: `classification` or
`mask_refinement`. One SQLite file represents one dataset with a stable UUID,
an independently addressable revision UUID, standardized metadata,
content-addressed assets, annotation history, lifecycle state,
and a semantic fingerprint.

Active databases are `working`. Save a frozen training checkpoint with:

```bash
oracle-dataset checkpoint path/to/training.sqlite
```

The source remains editable; the timestamped copy is frozen. Use
`oracle-dataset thaw <database> --reason "..."` to make a frozen database
editable again through an explicit, audited transition. Actual training requires
a frozen dataset, while validation and preflight can inspect working datasets.

Legacy ROI databases with `samples` and the older `mask_annotations` schema are
automatically migrated when opened by mask-refinement tools. Oracle Builder
first writes a timestamped `*.pre-v1-*.sqlite` backup, builds and validates V1
in a temporary database, and then atomically swaps it into place. Historical
annotations and compatible additive prediction tables are preserved. Legacy
dataset-level split assignments are recorded in the migration receipt and
replaced by run-owned split manifests. This
applies to the mask editor, ROI analysis and visualization, segmentation
training, saved-run evaluation, and inference. You can run the conversion
explicitly with:

```bash
python3 -m oracle_builder.datasets.cli migrate-roi \
  datasets/unet_training.sqlite
```

The migrated file is a `working` dataset; checkpoint it before model training.
U-Net validation reports the resulting dataset/revision UUIDs, schema version,
lifecycle, semantic fingerprint, schema validation, and migration receipt.

See [the V1 dataset contract](docs/dataset-schema-v1.md) for the schema,
deterministic folder interchange format, lifecycle commands, sidecar metadata,
and PostgreSQL mapping.

## Quick Start

Install the base training stack:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
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

### Multi-GPU Training

Training automatically uses `tf.distribute.MirroredStrategy` when TensorFlow
reports more than one logical GPU:

```toml
[distribution]
strategy = "auto"
devices = []
cross_device_ops = "auto"
fallback_to_single = true
memory_growth = true
```

`data.batch_size` is the global batch size and must be divisible by the number
of synchronized replicas. For example, `batch_size = 32` on four GPUs gives a
per-replica batch size of eight. Model construction, optimizer creation,
supervised training, and student–teacher pretraining all occur in the same
strategy scope.

To require mirrored execution and fail when two GPUs are not available:

```toml
[distribution]
strategy = "mirrored"
fallback_to_single = false
```

Specific logical GPUs and cross-device reduction may be selected explicitly:

```toml
[distribution]
strategy = "mirrored"
devices = ["/GPU:0", "/GPU:1"]
cross_device_ops = "nccl"
```

Supported strategies are `auto`, `single`, `mirrored`, and `cpu`.
`cross_device_ops` can be `auto`, `nccl`, or `hierarchical_copy`. Every run
writes `provenance/distribution.json` with the resolved strategy, devices, replica count,
and global/per-replica batch sizes.

Create and train a synthetic classification example:

```bash
python3 -m oracle_builder.data.sqlite_dataset \
  --classification datasets/example_classification.sqlite

python3 -m oracle_builder.datasets.cli checkpoint \
  datasets/example_classification.sqlite \
  --output datasets/example_classification.training.sqlite

python3 model_training.py \
  --config configs/example_classification.toml \
  --input datasets/example_classification.training.sqlite \
  --output example-classification-run \
  --overwrite
```

### Import A Classification Folder Library

Classification image libraries can be imported directly from folders:

```text
example training library/
├── cod/
│   ├── cod-001.jpg
│   └── nested/cod-002.png
├── salmon/
│   └── salmon-001.jpg
├── metadata.toml
└── about.yml
```

```bash
python3 scripts/import_classification_folders.py \
  --input "$HOME/Desktop/example training library" \
  --output datasets/example_training.sqlite \
  --validation-fraction 0.20 \
  --test-fraction 0.10 \
  --seed 123
```

Immediate child folders are class names, and images are discovered recursively
within them. JPEG, PNG, and TIFF are supported. Original encoded image bytes are
stored by default so different models can later use different preprocessing.
Class indices are stable and written to both `classification_labels` and a
`.labels.json` sidecar.

Top-level `.json`, `.toml`, `.yaml`, and `.yml` files are parsed as dataset
metadata. Their parsed JSON, original text, format, checksum, and source filename
are stored in `metadata_documents`. Each import is recorded in `import_events`
with its options and summary.

Attach or replace a metadata document on an existing editable dataset with:

```bash
python3 -m oracle_builder.datasets.cli metadata-add \
  datasets/example_training.sqlite \
  metadata.toml \
  --actor "$USER"
```

The source filename is the default logical name. Use `--name about` to update a
stable logical document from a differently named file. The operation preserves
the source text, validates and stores parsed JSON, records its SHA-256 checksum,
adds a provenance event, and changes the dataset fingerprint. Frozen datasets
must be explicitly thawed before their metadata can be changed.

Inspect a proposed import without creating or modifying the database:

```bash
python3 scripts/import_classification_folders.py \
  --input "$HOME/Desktop/example training library" \
  --output datasets/example_training.sqlite \
  --dry-run
```

Every import writes `.import_report.json`, `.import_report.csv`, and
`.labels.json` beside the output. Reports include proposed per-class split hints,
warnings, corrupt files, duplicates, and import status.

Important importer controls include:

| Option | Behavior |
|---|---|
| `--split-mode stratified-hash` | Include proposed split counts in the import report only; default. |
| `--split-mode existing-folders` | Read class folders nested under source `train/`, `validation/`, and `test/` directories without storing those assignments. |
| `--split-mode none` | Omit proposed split grouping from the import report. |
| `--label-map PATH` | Use an explicit class-name-to-index JSON mapping. |
| `--allow-new-classes` | Permit new classes when appending to an existing database. |
| `--duplicate-policy error\|skip\|allow` | Handle identical image content. |
| `--existing-policy error\|skip\|update` | Handle deterministic UUID collisions. |
| `--on-error error\|skip` | Abort or report-and-skip invalid images. |
| `--require-rgb` | Reject images whose decoded source mode is not RGB. |
| `--no-allow-grayscale` | Reject grayscale inputs. |
| `--minimum-images-per-class N` | Reject undersized classes. |

Existing-split layout is also supported:

```text
library/
├── train/cod/
├── train/salmon/
├── validation/cod/
├── validation/salmon/
├── test/cod/
└── test/salmon/
```

For specialized portable datasets, preprocessing may be materialized during
import:

```bash
python3 scripts/import_classification_folders.py \
  --input images/by-class \
  --output datasets/materialized.sqlite \
  --storage-mode materialized \
  --input-shape 224 224 3 \
  --resize-mode fit_pad \
  --channel-mode rgb \
  --normalization dtype \
  --invert
```

Original storage is recommended. Materialized storage permanently fixes image
dimensions and intensity processing in the database.

### Classification Preprocessing

Original images are deterministically prepared when a classification dataset is
loaded:

```toml
[preprocessing]
resize_mode = "fit_pad"
normalization = "dtype"
rescale = true
invert = false
pad_value = 0.0
interpolation = "bilinear"
channel_mode = "auto"
percentile_low = 1.0
percentile_high = 99.0
```

`data.input_shape` supplies the target height, width, and channel count.
`fit_pad` preserves aspect ratio and center-pads, `fill_crop` preserves aspect
ratio and center-crops, `stretch` directly resizes, and `none` requires an exact
match. `fit` resizes within the bounds but is intended for inspection because
variable output sizes cannot form ordinary training batches.

Channel mode can be `auto`, `grayscale`, `rgb`, or `rgba`. Normalization can be
`dtype`, `minmax`, `percentile`, or `none`. Inversion occurs after normalization
as `1 - x`.

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

Authenticated Pelagia deployments require a bearer token on read endpoints. You can pass an existing token:

```bash
export PELAGIA_API_TOKEN=YOUR_TOKEN
```

Or let `mask_builder.py` log in before loading data:

```bash
export PELAGIA_USERNAME=ada
export PELAGIA_PASSWORD=secret
export PELAGIA_PROJECT_KEY=default
```

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
  --api-username ada \
  --api-password secret \
  --api-project-key default \
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

Useful Pelagia filters include `--run-id`, `--asset-id`, `--collection`, `--frame-id`, `--start-frame`, `--end-frame`, `--roi-index`, bbox filters such as `--min-bbox-x`, `--min-width`, and `--max-bbox-w`, area/perimeter filters such as `--min-area 500`, payload filters such as `--roi-encoding`, `--roi-format`, `--api-mask-encoding`, `--mask-format`, and paging/sorting filters such as `--limit`, `--offset`, `--sort-by`, and `--sort-dir`.

By default the Pelagia loader uses:

- `GET /detections/{detection_id}` for metadata
- `GET /detections/{detection_id}/roi?format=png` for the ROI image
- `GET /detections/{detection_id}/mask?format=png` for the candidate mask

These calls send `Authorization: Bearer <token>` when `--api-token`, `PELAGIA_API_TOKEN`, or login credentials are provided. A Pelagia `viewer` project role is enough for these read-only dataset-building operations.

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

### Mask Builder CLI Reference

Run `python3 mask_builder.py --help` for the argparse source of truth. Current arguments:

| Argument | Description |
| --- | --- |
| `--database PATH` | SQLite dataset to read from or write to. |
| `--input PATH` | Legacy alias for `--database`; image paths are treated as image imports. |
| `--image PATH` | Local image file or folder to mask. |
| `--output PATH` | Legacy alias for `--database`. |
| `--uuid UUID` | Sample UUID; defaults to the image filename stem for image imports. |
| `--missing-masks-only` | When reading SQLite samples, show only samples without saved masks. |
| `--read-only` | Open without saving to a dataset. |
| `--mask-encoding {png,npy}` | Encoding used when saving validated masks. |
| `--debug` | Print extra launch and selection details. |

Pelagia connection and auth:

| Argument | Description |
| --- | --- |
| `--api-base-url URL` | Pelagia base URL; defaults to `http://localhost:8000`. |
| `--api-token TOKEN` | Bearer token; defaults to `PELAGIA_API_TOKEN`. |
| `--api-username NAME` | Username for `POST /auth/login`; defaults to `PELAGIA_USERNAME`. |
| `--api-password PASSWORD` | Password for `POST /auth/login`; defaults to `PELAGIA_PASSWORD`. |
| `--api-project-key KEY` | Project key for login; defaults to `PELAGIA_PROJECT_KEY` or `default`. |
| `--api-roi-id ID`, `--roi-id ID`, `--detection-id ID` | Pelagia detection/ROI id to open. |
| `--api-endpoint-template TEMPLATE` | Generic ROI endpoint template using `{roi_id}` instead of the Pelagia detection routes. |
| `--list-api-rois` | List matching Pelagia detections and exit. |
| `--api-browse-rois` | Open the first matching Pelagia detection as a queue. |
| `--random-api-roi` | Pick a random matching Pelagia detection. |

Pelagia detection filters:

| Argument | Description |
| --- | --- |
| `--api-run-id ID`, `--run-id ID` | Filter by run id. |
| `--api-asset-id ID`, `--asset-id ID` | Filter by asset id. |
| `--api-collection NAME`, `--collection NAME` | Filter by collection. |
| `--api-frame-id ID`, `--frame-id ID` | Filter by frame id. |
| `--api-start-frame N`, `--start-frame N` | Minimum frame index. |
| `--api-end-frame N`, `--end-frame N` | Maximum frame index. |
| `--api-roi-index N`, `--roi-index N` | Filter by ROI index. |
| `--api-min-bbox-x N`, `--min-bbox-x N` | Minimum bounding-box x. |
| `--api-max-bbox-x N`, `--max-bbox-x N` | Maximum bounding-box x. |
| `--api-min-bbox-y N`, `--min-bbox-y N` | Minimum bounding-box y. |
| `--api-max-bbox-y N`, `--max-bbox-y N` | Maximum bounding-box y. |
| `--api-min-bbox-w N`, `--min-bbox-w N`, `--min-width N` | Minimum bounding-box width. |
| `--api-max-bbox-w N`, `--max-bbox-w N` | Maximum bounding-box width. |
| `--api-min-bbox-h N`, `--min-bbox-h N`, `--min-height N` | Minimum bounding-box height. |
| `--api-max-bbox-h N`, `--max-bbox-h N` | Maximum bounding-box height. |
| `--api-min-area N`, `--min-area N` | Minimum ROI area. |
| `--api-max-area N`, `--max-area N` | Maximum ROI area. |
| `--api-min-perimeter N`, `--min-perimeter N` | Minimum ROI perimeter. |
| `--api-max-perimeter N`, `--max-perimeter N` | Maximum ROI perimeter. |
| `--api-roi-encoding VALUE`, `--roi-encoding VALUE` | Filter by ROI payload encoding. |
| `--api-roi-format VALUE`, `--roi-format VALUE` | Request/filter ROI response format. |
| `--api-mask-encoding VALUE` | Filter by mask payload encoding. |
| `--api-mask-format VALUE`, `--mask-format VALUE` | Request/filter mask response format. |
| `--api-limit N`, `--limit N` | Maximum detections to list or sample. |
| `--api-offset N`, `--offset N` | Detection list offset. |
| `--api-sort-by FIELD`, `--sort-by FIELD` | Detection list sort field; defaults to `asset_frame`, or `random` for `--random-api-roi`. |
| `--api-sort-dir {asc,desc}`, `--sort-dir {asc,desc}` | Detection list sort direction. |

U-Net dataset utilities:

| Argument | Description |
| --- | --- |
| `--validate-unet-dataset` | Validate that `--database` is ready for U-Net training and exit. |
| `--write-unet-config PATH` | Write a U-Net TOML config inferred from the dataset and exit. |
| `--unet-batch-size N` | Batch size for generated U-Net configs. |
| `--unet-epochs N` | Epoch count for generated U-Net configs. |
| `--unet-model NAME` | Generated architecture: `unet`, `residual_unet`, or `unet_plus_plus`. |
| `--unet-segmentation-target NAME` | Generated target mode: `validated_mask` or `candidate_delta`. |
| `--unet-candidate-sdf` | Generate a three-channel model config with candidate signed distance as channel 2. |
| `--unet-candidate-sdf-clip-distance N` | Candidate SDF distance mapped to magnitude `1`; defaults to `32` pixels. |
| `--unet-tiling` | Enable large-ROI tiling in the generated config. |
| `--unet-tiling-overlap N` | Adjacent tile overlap fraction; defaults to `0.5`. |
| `--unet-tiling-blend NAME` | Reassembly weighting: `uniform` or `hann`. |
| `--unet-input-shape SHAPE` | Target input shape, for example `256,256,2`. |
| `--unet-output-shape SHAPE` | Target output shape, for example `256,256,1`. |

### Napari Mask Editing

The viewer has three important layers:

- `image`: the ROI image.
- `candidate mask`: the mask provided by the API, shown as a labels layer and hidden by default.
- `validated mask`: the editable manual result, initialized from the candidate mask.

Paint foreground in `validated mask` with label `1`; erase with label `0`. The side panel includes thresholding, image inversion before thresholding, cleanup actions, black/white background switching, validation, `Save`, `Save and next`, and `Skip`. A horizontally scrollable ROI navigator at the bottom shows the active database, API, or folder queue; click any thumbnail to move backward or forward without saving the current edits. The selected ROI is highlighted. Thresholding and cleanup actions update the selected labels layer, so local image sources without an API mask can select `candidate mask` and use `Apply threshold` to generate an initial candidate. Candidate and validated masks render with distinct foreground colors. On save, the image and current candidate mask are stored as the two-channel input blob, and the validated mask is stored as the output blob.

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

Pelagia ROI crops can have different raw sizes. During segmentation training, oracle-builder preserves their aspect ratio while rescaling and center-padding inputs to `data.input_shape` and validated masks to `data.output_shape`. The ROI channel uses bilinear resizing; candidate and validated mask channels use nearest-neighbor resizing. Padding uses zeros.

### Visualize Training ROIs

Create a contact sheet of the selected split. Candidate/API masks are transparent blue; validated/refined masks are transparent green:

```bash
python3 scripts/visualize_unet_training_rois.py \
  --database datasets/unet_training.sqlite \
  --run runs/unet-test \
  --split train \
  --output runs/unet-test/training_roi_overview.png
```

Compare each ROI across candidate/original, model-output, and validated-mask columns:

```bash
python3 scripts/visualize_unet_training_rois.py \
  --database datasets/unet_training.sqlite \
  --run runs/unet-test \
  --split test \
  --side-by-side \
  --predictions runs/unet-test/predictions/predictions.sqlite \
  --prediction-set unet-test \
  --output runs/unet-test/mask_comparison.png
```

The predictions database is a complete SQLite backup of the input dataset with additive `prediction_sets` and `predictions` tables. Post-training prediction includes every ROI and records its `train`, `validation`, or `test` tag. ROIs without ground truth still receive predictions, with null truth and metrics fields.

Visualization arguments:

| Argument | Description |
| --- | --- |
| `--database PATH` | Required oracle-builder SQLite dataset. |
| `--output PATH` | Required output PNG path. |
| `--predictions PATH` | Augmented dataset SQLite database; required with `--side-by-side`. |
| `--prediction-set NAME` | Prediction set to show; optional when the database contains exactly one set. |
| `--side-by-side` | Show original/candidate, model-output, and validated overlays as columns. |
| `--run PATH` | Preferred model-run artifact; uses its authoritative split manifest and resolved tiling config. |
| `--config PATH` | Legacy fallback TOML config for deriving splits when no run artifact is available. |
| `--split NAME` | Split to visualize; defaults to `train`. |
| `--thumbnail-size N` | Maximum tile image size in pixels. |
| `--columns N` | Grid columns; `0` chooses approximately square layout. |
| `--limit N` | Maximum number of ROIs to render. |
| `--candidate-alpha N` | Blue candidate-mask overlay opacity. |
| `--refined-alpha N` | Green validated-mask overlay opacity. |
| `--prediction-alpha N` | Orange model-output overlay opacity. |
| `--prediction-threshold N` | Override the threshold saved with the prediction set. |
| `--seed N` | Override split assignment seed. |
| `--validation-split N` | Override validation split fraction. |
| `--test-split N` | Override test split fraction. |
| `--no-labels` | Hide UUID labels under tiles. |
| `--show-tile-grid` | Draw tile boundaries from the supplied config over each source ROI. |

## SQLite Dataset Format

See also the [project architecture](docs/architecture.md) and the
[V1 dataset contract](docs/dataset-schema-v1.md).

Oracle Builder V1 treats one SQLite file as one logical dataset revision. Every
file has a stable lineage `dataset_id`, a distinct `revision_id`, an explicit
`dataset_type`, schema identity and version, lifecycle state, semantic
fingerprint, source metadata, and audit events.

The shared tables are:

- `ob_schema`: schema name and semantic version.
- `dataset`: the singleton dataset identity and metadata record.
- `assets`: content-addressed embedded bytes or external URIs.
- `dataset_items`: stable item IDs, weight, source key, and metadata. Split
  membership is not stored in the dataset schema.
- `metadata_documents`: parsed and original metadata sidecars.
- `import_events` and `dataset_events`: provenance and lifecycle history.

Use-case tables are deliberately typed:

- Classification uses `classification_labels`, `classification_items`, and
  append-only `classification_annotations`.
- Mask refinement uses `mask_refinement_items` and append-only
  `mask_annotations`. The ROI image, original candidate mask, and every
  validated-mask revision are distinct assets.

This avoids a generic input/output blob whose meaning changes with the model.
Models consume a typed dataset through the repository/loader layer; model
architecture is not part of the dataset schema.

### Working datasets, checkpoints, and training

New and imported datasets begin in the `working` lifecycle. Create a consistent
timestamped training checkpoint with:

```bash
python3 -m oracle_builder.datasets.cli checkpoint datasets/library.sqlite
```

The source remains editable. The copied file is marked `frozen`, and SQLite
triggers reject changes to dataset content or semantic metadata. Training
requires a frozen V1 checkpoint and records its `dataset_id`, `revision_id`,
schema version, and semantic SHA-256 fingerprint in `config/resolved.json`.

An explicit thaw makes a database editable again without changing its
`dataset_id`; it branches a new working `revision_id` from the frozen revision:

```bash
python3 -m oracle_builder.datasets.cli thaw \
  datasets/library.checkpoint-20260730T201503.125000Z.sqlite \
  --actor "$USER" \
  --reason "continue annotation review"
```

Use `info`, `validate`, or `freeze` for inspection and lifecycle management:

```bash
python3 -m oracle_builder.datasets.cli info datasets/library.sqlite
python3 -m oracle_builder.datasets.cli validate datasets/library.sqlite
python3 -m oracle_builder.datasets.cli freeze datasets/library.sqlite
```

### Deterministic folder transfer

Export either dataset type to an ordinary folder bundle:

```bash
python3 -m oracle_builder.datasets.cli export \
  datasets/library.sqlite \
  exports/library-v1
```

The bundle contains `manifest.json`, `metadata.toml`, source metadata
documents, images/masks, and `checksums.sha256`. Classification images are
placed under class folders. The manifest retains stable IDs, complete
annotation history, asset metadata, and the dataset fingerprint.

Reconstruct the SQLite file and verify every checksum and semantic fingerprint:

```bash
python3 -m oracle_builder.datasets.cli import \
  exports/library-v1 \
  datasets/library-restored.sqlite
```

The complete contract and planned SQLite-to-PostgreSQL type mapping are in
[Dataset schema V1](docs/dataset-schema-v1.md).

Prediction output databases retain these source tables and add:

```sql
CREATE TABLE prediction_sets (
    prediction_set TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    run_id TEXT,
    run_name TEXT,
    artifact_id TEXT,
    dataset_id TEXT,
    dataset_fingerprint_sha256 TEXT,
    config_json TEXT NOT NULL
);

CREATE TABLE predictions (
    prediction_set TEXT NOT NULL,
    uuid TEXT NOT NULL,
    split TEXT,
    y_true_blob BLOB,
    y_true_encoding TEXT,
    y_pred_blob BLOB,
    y_pred_encoding TEXT,
    y_prob_json TEXT,
    metrics_json TEXT,
    metadata_json TEXT,
    target_mode TEXT NOT NULL DEFAULT 'validated_mask',
    reconstructed_pred_blob BLOB,
    reconstructed_pred_encoding TEXT,
    features_blob BLOB,
    features_encoding TEXT,
    features_dim INTEGER,
    prediction_packet_json TEXT,
    PRIMARY KEY (prediction_set, uuid)
);
```

Multiple runs can append distinct prediction sets to the same output database. Reusing a set name replaces that set's prediction for the same ROI.

Classification prediction rows store their fixed-size feature vectors in
`features_blob` using NumPy encoding. The `prediction_set` associates each vector
with the exact run configuration and model that produced its feature space.
`prediction_packet_json` stores softmax, prototype, and KNN evidence together.

For candidate-delta runs, `y_pred_blob` stores raw delta probabilities and `reconstructed_pred_blob` stores reconstructed validated-mask probabilities. The side-by-side visualizer automatically shows candidate, predicted changes, reconstructed mask, and validated mask columns.

Pelagia detection IDs are preserved as `dataset_items.item_id` values unless
`--uuid` is explicitly provided. Pelagia metadata is stored in
`dataset_items.metadata_json`, including `pelagia_detection_id` and the full
detection metadata under `pelagia`.

Train/validation/test assignments are stored in
`protocol/splits.json` inside each model-run artifact. The manifest is keyed by
dataset item ID and bound to the exact dataset revision and fingerprint.

## Config Files

Configs are TOML files. Required sections are `[run]`, `[data]`, and `[training]`.
For classification, `data.num_classes` may be omitted when using SQLite input;
oracle-builder infers it from `classification_labels`.

For classification:

```toml
[run]
task = "classification"
model = "simple_cnn"

[data]
input_shape = [128, 128, 1]

[training]
loss = "weighted_sparse_categorical_crossentropy"

[training.class_weights]
mode = "effective_number"
```

For classification, `data.num_classes` is optional. When omitted, oracle-builder
infers it from the selected SQLite database's `classification_labels` table.
This keeps model configs reusable across training libraries. Explicit
`num_classes` remains supported. Classification defaults assume grayscale input,
effective-number class weighting, and no per-epoch checkpoints; each can be
overridden explicitly.

For segmentation:

```toml
[run]
task = "segmentation"
model = "unet"

[data]
input_shape = [256, 256, 2]
output_shape = [256, 256, 1]

[training]
loss = "bce_soft_dice"
bce_weight = 1.0
soft_dice_weight = 1.0
metrics = ["accuracy", "dice", "iou"]
segmentation_target = "validated_mask"
```

All U-Net architectures can instead learn only the corrections between the candidate and validated masks:

```toml
[training]
segmentation_target = "candidate_delta"
loss = "bce_soft_dice"
```

Delta mode uses the binary target `candidate XOR validated`. It requires candidate-mask inputs with the candidate in channel 1; the model input has two channels normally or three when SDF input is enabled. At inference, the thresholded model output is XORed with that candidate to reconstruct the predicted validated mask. Validation threshold selection maximizes reconstructed validated-mask Dice, while reports also include raw delta Dice, candidate baseline Dice, Dice improvement, correction fraction, and addition/removal pixel counts. Spatial edge weights, when enabled, are calculated from the delta target. In the visualizer, predicted additions are cyan and predicted removals are red.

Candidate-mask datasets can optionally provide a signed Euclidean distance field as a third model input:

```toml
[data]
input_shape = [256, 256, 3]
candidate_sdf = true
candidate_sdf_clip_distance = 32.0
```

The database remains a two-channel ROI-plus-candidate dataset. During loading, oracle-builder resizes and pads the candidate, calculates `distance_inside - distance_outside`, divides by `candidate_sdf_clip_distance`, clips to `[-1, 1]`, and appends it as channel 2. Positive values are inside the candidate and negative values are outside. Empty and full masks produce constant `-1` and `1` fields. Only channel 0 receives photometric augmentation; channel 1 remains a nearest-neighbor binary mask, while the SDF receives the same geometry with bilinear interpolation.

Generate this configuration directly with:

```bash
python3 mask_builder.py \
  --database datasets/unet_training.sqlite \
  --write-unet-config configs/delta_sdf_unet.toml \
  --unet-segmentation-target candidate_delta \
  --unet-candidate-sdf \
  --unet-candidate-sdf-clip-distance 32
```

### Large ROI Tiling

Large segmentation ROIs can be processed at their native resolution instead of being downscaled:

```toml
[tiling]
enabled = true
overlap_fraction = 0.5
blend_mode = "hann"
tile_large_rois_only = true
normalize_training_coverage = true
```

Tile height and width come from `data.input_shape`. With a 256×256 model input, a 512×512 ROI produces four tiles at zero overlap or nine tiles at 50% overlap. Non-divisible dimensions receive a final tile anchored to the far edge, ensuring complete coverage. If only one source dimension exceeds the tile, that dimension is tiled and the smaller dimension is center-padded.

ROI split assignment happens before expansion, so every tile from one source remains in the same train, validation, or test split. ROI image, candidate mask, validated or delta target, SDF, and spatial edge weights share identical tile coordinates. SDF and boundary weights are calculated over the full ROI before cropping, avoiding artificial tile-edge boundaries.

When `normalize_training_coverage` is enabled, per-pixel training weights are divided by the number of overlapping tiles covering that source pixel. This prevents overlap from increasing the effective loss weight of central regions.

Inference predicts every tile and reassembles one full-resolution probability map per original ROI:

```text
full_probability = sum(tile_probability * blend_weight) / sum(blend_weight)
```

`uniform` performs ordinary averaging. `hann` reduces contributions near tile edges and uses a small positive edge floor so source-border pixels remain covered. Candidate-delta reconstruction and validation threshold optimization occur after full-resolution reassembly. Prediction databases retain one row per original ROI and record its original shape, tile count, overlap, and blend mode.

U-Net segmentation can emphasize errors near target boundaries using per-pixel weights
`w = 1 + lambda * exp(-d^2 / (2 * sigma^2))`, where `d` is the Euclidean distance to the active training-target boundary:

```toml
[training]
loss = "bce_soft_dice"
spatial_edge_weighting = true
edge_weight_lambda = 4.0
edge_weight_sigma = 5.0
```

The `bce_soft_dice` objective is binary cross-entropy plus `1 - soft_dice`, calculated per image. `bce_weight` and `soft_dice_weight` default to `1.0`; `soft_dice_smooth` defaults to `1e-6`. `edge_weight_lambda` controls the additional spatial weight at the boundary (the maximum weight is `1 + lambda`). `edge_weight_sigma`, measured in pixels at `data.output_shape`, controls how far that emphasis extends. Empty masks receive neutral spatial weights of `1`. The same spatial transform is applied to weights during geometric augmentation.

When `dice` appears in `training.metrics`, training history includes `dice` and `val_dice`, and the run writes `figures/dice_curve.png`. After segmentation training, oracle-builder predicts the validation split at thresholds from `0.05` through `0.95`, selects the threshold with the highest aggregate Dice, and uses it for final evaluation and saved prediction metrics. Ties are resolved toward `0.5`. Results are written to:

- `evaluation/validation_threshold_analysis.json`
- `evaluation/validation_threshold_curve.csv`
- `config/resolved.json` under `evaluation.segmentation_threshold`
- `artifact.json` under `summary.validation_threshold_analysis`

Default values are merged in from `oracle_builder/config.py`. Common settings include `batch_size`, `shuffle_buffer`, `validation_split`, `test_split`, `epochs`, `optimizer`, `learning_rate`, callback settings, and output toggles.

## Streaming Classification Data

Classification training streams image data from SQLite by default:

```toml
[data.streaming]
enabled = true
reader_workers = 4
prefetch_batches = 2
deterministic = true
sqlite_cache_kib = 65536

[output]
prediction_commit_batches = 20
```

Only row IDs, labels, run-owned split assignments, encodings, dimensions, and lightweight
metadata are indexed in memory. Each mapping worker owns a read-only SQLite
connection and lazily fetches and preprocesses the current image. Memory is
bounded by the shuffle buffer, reader workers, active batch, and configured
prefetch batches rather than total database size.

The same source feeds supervised training, unlabeled student–teacher
pretraining, evaluation, feature extraction, prototype construction, and
prediction writing. Dataset rows are never mutated to create splits.
Classification prediction rows are committed incrementally.
Evidence embeddings are written to memory-mapped `.npy` files; prototype sums
are accumulated online.

Set `data.streaming.enabled = false` only for legacy/debug parity. Segmentation
currently retains its specialized eager tile/reassembly pathway.

## Input Augmentation

Input augmentation is configured in TOML under `[augmentation]` and is applied only to the training split. Validation and test splits are never augmented.

```toml
[augmentation]
enabled = true
repeats_per_epoch = 4
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

`repeats_per_epoch` controls how many augmented passes over the training split are used per epoch. For example, `repeats_per_epoch = 4` means every stored training ROI is reused four times in each epoch, with augmentation sampled independently for each pass. Validation and test data are still used once and are not augmented.

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

Arguments:

| Argument | Description |
| --- | --- |
| `-c PATH`, `--config PATH` | Required TOML config. |
| `-i PATH`, `--input PATH` | Required SQLite dataset. |
| `-o NAME`, `--output NAME` | Required run name under `--runs-dir`. |
| `--runs-dir PATH` | Parent folder for runs; defaults to `./runs`. |
| `--overwrite` | Replace an existing run directory with the same name. |
| `--resume RUN_DIRECTORY` | Resume a failed or interrupted run from its validated rolling recovery snapshot. The original dataset path is used unless `--input` is supplied. |
| `--dry-run` | Print resolved config and run path without training. |
| `--preflight` | Validate segmentation SQLite compatibility and exit. |
| `--debug` | Enable debug mode in the resolved config. |

Oracle Builder keeps one rolling full-model recovery snapshot by default, while
ordinary per-epoch checkpoints remain disabled. Recovery includes optimizer
state and is compatible with both classification and segmentation training:

```toml
[recovery]
enabled = true
save_every_epochs = 1
```

After an interruption or a post-training failure, resume without re-specifying
the config, output name, or split policy:

```bash
python3 model_training.py --resume runs/RUN_NAME
```

Resume verifies the run artifact, frozen dataset identity/fingerprint, split
manifest, recovery checksum, and resolved training contract before continuing.
If the database moved, provide `--input /new/path/dataset.sqlite`; it must still
match exactly. Recovery begins after the last completed supervised epoch.

## Evaluation And Inference

Inference uses a separate batch policy from training. By default,
`inference.batch_size = "auto"` chooses a bounded candidate from the input shape
and architecture, verifies it with a forward pass, and halves it if TensorFlow
reports resource exhaustion. Set a positive integer for a fixed, still-verified
batch size. Long post-training stages report their current operation and batch
progress.

Classification evaluation always preserves raw and normalized confusion
matrices as CSV/JSON. For large label spaces the primary figure uses sparse
ticks without per-cell text, and `top_confusions.csv`/`.png` rank the most
frequent off-diagonal class pairs.

### Weighted classification and SimCLR

Classification supports ordinary sparse cross entropy and class-weighted sparse
cross entropy. Automatic weights are calculated only from the frozen training
split and the resolved class counts and values are stored in the run config:

```toml
[training]
loss = "weighted_sparse_categorical_crossentropy"

[training.class_weights]
mode = "effective_number" # explicit, inverse_frequency, effective_number
beta = 0.999
normalize = true
```

Self-supervised pretraining supports the existing BYOL student-teacher method
and SimCLR with NT-Xent:

```toml
[pretraining]
enabled = true
method = "simclr"
temperature = 0.1
projection_dim = 128
```

See `configs/example_classification_weighted_simclr.toml` for a complete
weighted SimCLR configuration. Commented, dataset-agnostic starting points for
every supported classifier family are indexed in
`configs/classification_defaults/README.md`; their preprocessing and
augmentation policies are deliberately identical for fair architecture
comparisons.

Final classification evaluation reports accuracy, balanced accuracy,
macro/weighted precision, recall and F1, Cohen's kappa, multiclass Matthews
correlation, top-k accuracy, log loss, multiclass Brier score, expected
calibration error, per-class metrics and ranked confusions. Segmentation
evaluation additionally reports specificity, pixel accuracy and DICE
distribution statistics.

When enabled, `evaluation/performance.json` contains a bounded synthetic
inference benchmark:

```toml
[evaluation.benchmark]
enabled = true
warmup_batches = 2
measured_batches = 10
```

It records model parameter counts, batch and approximate sample latency,
throughput, TensorFlow devices, the resolved inference batch, evaluation
pipeline duration and peak device memory when available.

Evaluate a saved run:

```bash
python3 model_evaluate.py \
  --run runs/RUN_NAME \
  --input DATASET.sqlite \
  --split test \
  --output evaluations/RUN_NAME-test
```

Evaluation arguments:

| Argument | Description |
| --- | --- |
| `--run PATH` | Required V1 run artifact directory. |
| `--input PATH` | Required SQLite dataset. |
| `--split NAME` | Required split to evaluate. |
| `--output PATH` | Required new directory outside the sealed run artifact. |

Write predictions from a saved run:

```bash
python3 model_inference.py \
  --run runs/RUN_NAME \
  --input DATASET.sqlite \
  --split all \
  --prediction-set RUN_NAME \
  --output predictions/RUN_NAME.sqlite
```

Inference arguments:

| Argument | Description |
| --- | --- |
| `--run PATH` | Required V1 run artifact directory. |
| `--input PATH` | Required SQLite dataset. |
| `--output PATH` | Required augmented SQLite output path. A new file starts as a complete backup of `--input`. |
| `--split NAME` | One of `all`, `train`, `validation`, or `test`; defaults to `all`. |
| `--prediction-set NAME` | Set name used as part of each prediction key; defaults to the run directory name. |

## Run Outputs

Each training run is a V1 model-run artifact rooted at `runs/<run_name>/`.
`artifact.json` is authoritative and records stable artifact/run IDs, dataset
identity and fingerprint, model input/output contract, status, lifecycle, paths,
inventory, and artifact fingerprint.

Portable and machine-specific concerns are separated:

- `config/source.toml` and `config/resolved.json`: portable configuration.
- `provenance/`: runtime paths, environment, package snapshot, and distribution.
- `logs/training.sqlite`: run, epoch metrics, and execution events.
- `metrics/`: training and pretraining histories.
- `model/`: model formats, model manifest, checkpoints, and load-test report.
- `evaluation/`, `predictions/`, and `figures/`: derived run products.
- `README.md` and `MODEL_CARD.md`: reader-facing identity and limitations.
- `checksums.sha256`: complete sealed-file integrity inventory.

Completed and failed runs are sealed automatically. Inspect, verify, and create a
deterministic preservation package with:

```bash
oracle-run info runs/RUN_NAME
oracle-run validate runs/RUN_NAME
oracle-run pack runs/RUN_NAME archive/RUN_NAME.oracle-run.zip
oracle-run migrate-legacy runs/OLD_RUN preserved/OLD_RUN-v1
```

See [Model-run artifact V1](docs/run-artifact-v1.md) for the complete layout,
lifecycle, portability, preservation, and Pelagia/PostgreSQL mapping.

## Model Portability

oracle-builder saves models in several ways:

- `model/final.keras` for full Keras reloads.
- `model/weights.weights.h5` for rebuild-and-load workflows.
- `model/export_savedmodel/` for inference-oriented TensorFlow export.

Every completed run immediately performs a reload/prediction check and writes `model/load_test_report.json`.

## Models

The registry currently supports:

- `simple_cnn`
- `resnet` (`resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`)
- `densenet` (`densenet121`, `densenet169`, `densenet201`)
- `efficientnet` (`efficientnet_b0` through `efficientnet_b7`)
- `unet`
- `residual_unet` (alias: `resunet`)
- `unet_plus_plus` (alias: `unetpp`)
- `resnet_like`
- `densenet_like`

Each model module exposes `build_model(config: dict)`. Add future models in `models/` and register them in `oracle_builder/registry.py`.

Documented, runnable, dataset-independent defaults for every classification
family are in [`configs/classification_defaults/`](configs/classification_defaults/):

- `simple_cnn.toml`
- `resnet_like.toml`
- `densenet_like.toml`
- `resnet.toml`
- `densenet.toml`
- `efficientnet.toml`

Each file explains the family-specific architecture choices as well as shared
preprocessing, streaming, multi-GPU, embedding, pretraining, weighted-loss,
augmentation, inference, evaluation, and benchmark settings. The class count,
class order and labels are resolved from the input V1 SQLite dataset; split
assignments are resolved into the run artifact
dataset rather than duplicated in a config. Regenerate the checked-in examples
after changing the shared policy with:

```bash
python3 scripts/generate_classification_default_configs.py
```

Classification architectures may be selected either by family and variant:

```toml
[run]
task = "classification"
model = "resnet"

[model]
variant = "resnet50"
stem_kernel_size = 7
stem_stride = 2
embedding_dim = 256
normalize_embeddings = true
```

or directly with `model = "resnet50"`. The same applies to the named DenseNet and
EfficientNet variants. All three families support `stem_kernel_size`, `stem_stride`,
and `dropout`. ResNet also supports `base_filters`, `block_counts`, and `stem_pool`;
DenseNet supports `growth_rate`, `initial_filters`, `bottleneck_multiplier`,
`compression`, `block_config`, and `stem_pool`; EfficientNet supports
`width_coefficient`, `depth_coefficient`, `stem_filters`, `top_filters`, and
`se_ratio`. The builders initialize weights from scratch and use the configured
input shape and number of classes. Full examples are in
`configs/example_classification_{resnet,densenet,efficientnet}.toml`.

Every classification model has the same embedding contract. Its named `features`
layer produces a fixed-size vector immediately before dropout and the final
classifier. `embedding_dim` defaults to `256`, and `normalize_embeddings` defaults
to `true`, which gives every nonzero vector unit L2 length for similarity search
and clustering. Training remains a single-output classification problem. In
Python, retrieve both outputs without changing the trained model:

```python
from oracle_builder.classification.features import build_feature_model

feature_model = build_feature_model(model)
outputs = feature_model.predict(images)
probabilities = outputs["probabilities"]
features = outputs["features"]
```

Classification SavedModel exports provide `serving_default`, `classify`, and
`embed` signatures. The default signature returns both `probabilities` and
`features`; the other signatures return only their named output.

### Class Prototypes and KNN Evidence

Classification runs build an identity-evidence index from unaugmented,
L2-normalized training embeddings after supervised fitting:

```toml
[evidence]
enabled = true
knn_k = 5
```

Each class prototype is the L2-normalized mean of that class's normalized
training embeddings. Prototype scores are cosine similarities between the query
and every class prototype. New runs save a disk-backed
`model/classification_evidence/` directory containing memory-mappable `.npy`
arrays for normalized reference embeddings, labels, UUIDs, and prototypes, plus
`metadata.json`. Legacy `.npz` indices remain readable.

Exact KNN lookup uses an inner-product matrix operation and partial top-k
selection. Because both query and reference embeddings are normalized, this has
the same ranking as Euclidean distance and can later be replaced by an
approximate inner-product index without changing packet semantics. When a
training ROI is queried, its own UUID is excluded.

Every classification `prediction_packet_json` contains:

- Softmax probabilities, predicted class, and confidence.
- Similarity to each class prototype, nearest prototype, and prototype margin.
- Nearest-neighbor cosine similarity.
- Agreement fraction for the strongest top-k label.
- Per-label top-k counts.
- Similarity-weighted label support, using `(cosine_similarity + 1) / 2` as the
  nonnegative neighbor weight and normalizing support across represented labels.
- Margin between the strongest and second-strongest weighted label supports.
- Neighbor UUIDs, labels, and cosine similarities for auditability.

### Student–Teacher Self-Supervised Pretraining

Classification runs can optionally begin with BYOL-style student–teacher
pretraining:

```toml
[pretraining]
enabled = true
method = "byol"
epochs = 50
learning_rate = 0.001
teacher_momentum = 0.99
projection_dim = 128
projection_hidden_dim = 256
use_training_augmentation = true

[pretraining.augmentation]
rotation = 0.08
zoom = 0.15
translation = 0.10
flip_horizontal = true
brightness = 0.20
contrast = 0.20
gaussian_noise = 0.03
```

When `use_training_augmentation = true`, pretraining starts from the run's
ordinary `[augmentation]` profile. An optional `[pretraining.augmentation]`
section may override individual values. This allows model sweeps to define one
canonical augmentation policy without duplicating it.

For each ROI, two independently augmented views are passed through a student and
teacher encoder. The student predictor learns the representation produced by the
opposite-view teacher. The teacher is not optimized by gradients; its weights are
an exponential moving average of the student controlled by `teacher_momentum`.
No labels or negative image pairs are used.

Only training-split ROIs are used for pretraining, preventing validation or test
leakage. Unlabeled training ROIs are included in the self-supervised stage and
excluded from supervised fine-tuning. Once pretraining completes, the student
backbone and fixed-size `features` layer continue directly into ordinary
supervised training.

Pretraining writes `metrics/pretraining/metrics.csv`,
`metrics/pretraining/metrics.json`, and
`model/pretraining/student_pretrained.weights.h5`. The projection and prediction heads
exist only during pretraining and are not part of the final classifier. See
`configs/example_classification_student_teacher.toml` for a complete run.

The residual U-Net replaces each ordinary convolutional block with a learned residual block and projection shortcut. U-Net++ uses nested dense skip connections between encoder and decoder nodes. Both accept the standard U-Net model options: `base_filters`, `depth`, `dropout`, `activation`, and `final_activation`. U-Net++ additionally accepts `deep_supervision = true`; its full-resolution supervision heads are averaged into one segmentation output so it remains compatible with the existing dataset, loss, evaluation, and prediction pipeline. Example configurations are provided in `configs/example_segmentation_residual_unet.toml` and `configs/example_segmentation_unet_plus_plus.toml`.

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
- Segmentation still materializes decoded arrays while its tiling and
  reassembly pathway is migrated to the streaming source abstraction.
- The mask builder requires optional GUI dependencies and does not automate napari GUI testing.
