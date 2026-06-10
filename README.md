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
    output_blob BLOB,
    output_blob_encoding TEXT,
    output_blob_dimensions TEXT,
    label_text TEXT,
    sample_weight REAL,
    metadata_json TEXT
);
```

Supported encodings currently include `utf-8`, `json`, `int`, `float`, `png`, and `npy`. `zstd` is stubbed with a clear error until the optional dependency is added.

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

