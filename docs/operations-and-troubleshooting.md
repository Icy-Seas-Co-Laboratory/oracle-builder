# Operations and troubleshooting

## Inspect, validate, and package a run

```bash
oracle-run info runs/RUN_NAME
oracle-run validate runs/RUN_NAME
oracle-run pack runs/RUN_NAME archive/RUN_NAME.oracle-run.zip
oracle-run unpack archive/RUN_NAME.oracle-run.zip restored/RUN_NAME
```

Use `oracle-run reopen ... --reason "..."` only when deliberately modifying a
sealed artifact, then seal it again. Run documentation and model cards live
inside the artifact; derived evaluation/prediction outputs should live outside.

## Dataset lifecycle

```bash
oracle-dataset freeze working.sqlite
oracle-dataset checkpoint working.sqlite
oracle-dataset thaw release.sqlite --reason "correct annotation"
oracle-dataset snapshot workspaces/review.sqlite --note "Milestone review"
oracle-dataset restore-workspace snapshots/review.snapshot.sqlite workspaces/review-restored.sqlite
oracle-dataset export release.sqlite release-folder
oracle-dataset import release-folder restored.sqlite
```

See [Dataset schema V1](dataset-schema-v1.md) for the precise lifecycle,
deterministic exchange format, metadata, and PostgreSQL mapping.

## Common problems

### `ModuleNotFoundError: oracle_builder`

Run commands from the repository root after activating its environment, then
install the project:

```bash
source .venv/bin/activate
python3 -m pip install -e .
```

### Dataset is not trainable

Training needs a frozen Dataset V1 file and usable accepted annotations/labels.
For segmentation, run:

```bash
python3 mask_builder.py --database DATASET.sqlite --validate-unet-dataset
python3 model_training.py --config CONFIG.toml --input DATASET.sqlite --output check --preflight
```

### TensorFlow cannot see a GPU

```bash
python3 scripts/check_tensorflow_devices.py
```

Install the matching `requirements-gpu-linux.txt`, `requirements-gpu-wsl2.txt`,
or `requirements-gpu-macos.txt`. On NVIDIA systems also verify `nvidia-smi`.

### Napari or mask-editor dependencies are missing

```bash
python3 -m pip install -r requirements-gui.txt
```

### Matplotlib cannot write its cache

```bash
mkdir -p .cache/matplotlib
MPLCONFIGDIR=.cache/matplotlib python3 model_evaluate.py --help
```

## Analysis helpers

```bash
python3 analysis/python/inspect_run.py runs/RUN_NAME
python3 analysis/python/compare_runs.py runs/run-a runs/run-b
Rscript analysis/R/inspect_run.R runs/RUN_NAME
Rscript analysis/R/compare_runs.R runs/run-a runs/run-b
```
