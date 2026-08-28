# Oracle Builder model-run artifact V1

New artifacts also carry the profiled [Oracle Model Artifact Standard V2](model-artifact-standard-v2.md).
V1 readers remain supported for compatibility.

## Purpose

A model run is a durable scientific artifact, not merely a directory of files
produced by a training process. V1 separates portable model identity and
contracts from machine-specific execution provenance and mutable derived work.

Every new run has:

- a stable artifact UUID and execution run UUID;
- the exact dataset UUID, schema version, and semantic fingerprint;
- a model-independent input/output contract;
- portable source and resolved configuration;
- an item-level, dataset-fingerprint-bound train/validation/test split manifest;
- isolated machine and software provenance;
- training logs and metrics in inspectable formats;
- one or more independently loadable model representations;
- evaluation and prediction products;
- a generated model card;
- a complete checksum inventory and semantic artifact fingerprint when sealed.

## Directory contract

```text
run-name/
├── artifact.json
├── checksums.sha256
├── README.md
├── MODEL_CARD.md
├── config/
│   ├── source.toml
│   └── resolved.json
├── protocol/
│   └── splits.json
├── provenance/
│   ├── runtime.json
│   ├── environment.json
│   ├── requirements.txt
│   └── distribution.json
├── logs/
│   └── training.sqlite
├── metrics/
│   ├── history.csv
│   ├── history.json
│   └── pretraining/
├── model/
│   ├── model_manifest.json
│   ├── final.keras
│   ├── weights.weights.h5
│   ├── export_savedmodel/
│   ├── pretraining/
│   ├── checkpoints/
│   ├── recovery/
│   │   ├── latest.keras
│   │   └── state.json
│   ├── model_summary.txt
│   └── load_test_report.json
├── evaluation/
├── predictions/
└── figures/
```

Empty optional directories may be absent from a packed artifact.

## Portable versus runtime configuration

`config/resolved.json` is sufficient to reconstruct preprocessing, architecture,
training semantics, and model outputs on another machine. Absolute input, run,
and source configuration paths are removed from it.

Those execution-only paths are retained in `provenance/runtime.json`. This
preserves audit information without making a path on one computer part of the
portable model contract.

The dataset is referenced rather than duplicated. `artifact.json` records the
dataset ID, dataset type, schema version, dataset version, lifecycle at training,
and semantic SHA-256 fingerprint. A receiving system can therefore resolve the
same dataset in SQLite cold storage, PostgreSQL warm storage, or an object
catalog without depending on its original path.

## Split protocol

Dataset items do not intrinsically belong to training, validation, or test.
Each run owns `protocol/splits.json`, which records the split policy, seed,
fractions, counts, every item ID assignment, dataset/revision IDs, and semantic
dataset fingerprint. This makes evaluation reproducible while allowing the same
frozen dataset to support multiple experimental protocols without mutation.

The resolved configuration stores split policy parameters but not the
potentially large assignment map. Loaders attach the manifest in memory and
reject missing item IDs. Named-split evaluation is permitted only against the
exact dataset revision and fingerprint recorded by the manifest.

## Model representations

`model/model_manifest.json` describes the input shape, preprocessing, output
semantics, task, architecture, run identity, and saved formats.
New artifacts additionally write `model/contract.json`, which is the
authoritative self-describing inference contract.

- `final.keras` is the preferred full Keras representation.
- `weights.weights.h5` supports deterministic architecture rebuilds.
- `export_savedmodel/` is the inference-oriented TensorFlow representation.
- Classification SavedModels expose probability and identity-embedding
  signatures. A reusable encoder may instead expose a representation signature.
- Embedding runs use `task = "embedding"` and expose only the normalized (or
  explicitly non-normalized) representation through `embed` and
  `serving_default`.
- Cluster assignments and cluster evidence are downstream analysis provenance;
  they are not a first-class output of a newly created model product.

The independent load test report documents which formats were successfully
reloaded and exercised before sealing.

## Lifecycle and integrity

Runs begin with lifecycle `working` and status `running`. Training sets status to
`complete`, `failed`, or `interrupted`, writes final execution events, and seals
the artifact. Failed and interrupted runs are also sealed so forensic information
and a valid rolling recovery snapshot are preserved.

Recovery uses one replaceable full Keras model snapshot, including optimizer
state, rather than an ever-growing checkpoint directory. `state.json` records
its checksum, completed supervised epoch, phase, artifact/run IDs, and a hash of
the dataset, split, model, preprocessing, and training contract. Resume validates
all of these before reopening the artifact and continuing:

```bash
oracle-train --resume runs/example
```

The default recovery cadence is one snapshot per completed epoch. It can be
adjusted with `[recovery] save_every_epochs`; setting `recovery.enabled = false`
disables restart support. Mid-pretraining recovery is intentionally not supported
in V1; a completed pretraining phase is preserved before supervised training.

Sealing:

1. inventories every file except `artifact.json` and `checksums.sha256`;
2. records file path, role, byte size, and SHA-256;
3. writes the ordinary `checksums.sha256` interchange file;
4. calculates a semantic artifact fingerprint;
5. changes lifecycle to `sealed`.

Loading a V1 model validates the sealed artifact first. Evaluation and inference
outputs created after sealing must be written outside the run directory.

Training records write canonical lifecycle events and metric rows as JSONL in
`logs/events.jsonl` and `metrics/metrics.jsonl`. SQLite and CSV files are
compatibility views for existing tooling.

An explicit reopen removes the active checksum seal and records a lifecycle
event. Resealing produces a new fingerprint:

```bash
oracle-run reopen runs/example --reason "attach reviewed documentation"
oracle-run seal runs/example
```

## Inspection and preservation packages

```bash
oracle-run info runs/example
oracle-run validate runs/example

oracle-run pack \
  runs/example \
  archive/example.oracle-run.zip

oracle-run unpack \
  archive/example.oracle-run.zip \
  restored/example
```

Packages use stable member ordering, timestamps, and permissions so identical
sealed runs produce identical ZIP bytes. Unpack refuses an existing destination,
protects against path traversal, and validates the restored artifact.

The package SHA-256 identifies the ZIP bytes. The artifact fingerprint identifies
the logical run and is independent of the package filename.

Existing pre-V1 runs can be copied into the new layout:

```bash
oracle-run migrate-legacy runs/old-run preserved/old-run-v1
```

Migration never edits the source. Recognized files are placed in their V1
namespaces, a model manifest is synthesized when needed, unknown root products
are retained under `attachments/`, and complete or failed runs are sealed.

## Pelagia and PostgreSQL integration

`artifact.json` is the intended catalog-ingestion boundary. A warm-storage
implementation can store:

- `artifact_id` and `run_id` as PostgreSQL UUIDs;
- the manifest as validated `jsonb`;
- dataset and future model-artifact relationships as UUID foreign keys;
- inventory rows as relational records when search is useful;
- large model files in content-addressed object storage using their SHA-256;
- package location and package checksum as preservation/export records.

Pelagia should consume the model input/output contract and stable IDs rather
than Oracle Builder directory conventions. Directory paths are a cold-storage
representation; the manifest is the portable contract.

## Preservation boundaries

- Training datasets and model runs are different artifacts joined by stable
  identity and fingerprint.
- Predictions are derived records, not annotations or model identity.
- Evaluation performed against an additional dataset should be a separate
  derived artifact referencing both model artifact and dataset.
- Checkpoints are execution state. Final model formats and their load tests are
  the preservation minimum.
- Sealing detects change; it is not encryption, access control, or a substitute
  for redundant storage and backup.
