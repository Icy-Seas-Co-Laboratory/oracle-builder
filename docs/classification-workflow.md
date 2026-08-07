# Classification workflow

Use this guide when your source library is a folder per label, optionally with
top-level TOML, JSON, YAML, or YML metadata sidecars.

## 1. Import the library

```text
training-library/
├── cod/cod-001.jpg
├── salmon/salmon-001.jpg
└── metadata.toml
```

```bash
python3 scripts/import_classification_folders.py \
  --input "$HOME/Desktop/training-library" \
  --output datasets/training.sqlite
```

Run with `--dry-run` first to inspect image errors, duplicates, and class
counts. The importer never creates a random training split.

## Image polarity and Pelagia-compatible model inputs

Oracle Builder preserves original image bytes, but its canonical model input is
light foreground on a black background. The importer records source polarity
at dataset level; it does not alter the originals. By default it uses a
conservative border-versus-center estimate across representative images:

```bash
oracle-import-classification \
  --input "$HOME/Desktop/training-library" \
  --output datasets/training.sqlite \
  --source-polarity auto
```

Use an explicit value when the acquisition convention is known:

```bash
--source-polarity dark_on_light   # brightfield/shadowgraph source; model inverts
--source-polarity light_on_dark   # already Pelagia-canonical; model does not invert
```

`mixed` and `unknown` are retained as provenance but do not silently invert.
Classification example configurations use `preprocessing.invert = "auto"`,
which resolves against the frozen dataset polarity and is saved as a concrete
boolean in the model artifact. Set `true` or `false` to override it for a run.

An existing conventional source layout is also accepted without an option:

```text
training-library/
├── train/cod/cod-001.jpg
├── validation/cod/cod-002.jpg
└── test/cod/cod-003.jpg
```

Those directory names are retained as `source_partition` provenance for each
item. They do not become dataset-level split state. A training run with the
default `data.split_strategy = "auto"` uses a complete source-partition layout;
otherwise it creates its own deterministic assignments. Set
`split_strategy = "random"` to ignore source partitions, or
`"source_partitions"` to require them.

Original encoded image bytes are the default and recommended storage form.
`--storage-mode materialized` is for a deliberately fixed, preprocessed image
representation.

## 2. Add metadata and freeze a training revision

```bash
oracle-dataset metadata-add datasets/training.sqlite metadata.toml --actor "$USER"
oracle-dataset checkpoint datasets/training.sqlite --output datasets/training.v1.sqlite
```

The source remains editable. The checkpoint is a separate frozen revision that
can be used by one or many reproducible runs. See the
[dataset schema reference](dataset-schema-v1.md).

## 3. Train a classifier

Choose a dataset-independent config from `configs/classification_defaults/`,
or begin with an example such as `configs/example_classification_resnet.toml`.

```bash
python3 model_training.py \
  --config configs/example_classification_resnet.toml \
  --input datasets/training.v1.sqlite \
  --output plankton-resnet50
```

The class count, label order, and run-specific split assignments are resolved
from the frozen database. Classification model families include simple CNN,
ResNet, DenseNet, and EfficientNet variants. Every native classifier exposes
probabilities, logits, and a fixed-size L2-normalized `features` embedding.

To enable weighted loss and student–teacher/SimCLR pretraining, start with
`configs/example_classification_weighted_simclr.toml` or see
[training and evaluation](training-and-evaluation.md).

## 4. Evaluate and write predictions

```bash
python3 model_evaluate.py \
  --run runs/plankton-resnet50 \
  --input datasets/training.v1.sqlite \
  --split test \
  --output evaluations/plankton-resnet50-test

python3 model_inference.py \
  --run runs/plankton-resnet50 \
  --input datasets/training.v1.sqlite \
  --split all \
  --prediction-set plankton-resnet50 \
  --output predictions/plankton-resnet50.sqlite
```

The prediction database begins as a complete copy of the source dataset and
adds prediction sets/results. It never changes the source database.

## Key choices

- Use `fit_pad` preprocessing when preserving object aspect ratio matters.
- Keep augmentations identical across architecture comparisons.
- Use a frozen checkpoint for every training run.
- Treat the saved run’s split manifest—not the source-folder layout—as the
  source of truth for evaluation.
- Use [model products](model-products.md) to ingest an externally trained
  classifier instead of retraining it.
