# Oracle Builder dataset schema V1

## Contract

Schema V1 separates a dataset's scientific meaning from model architecture. One
SQLite file contains exactly one dataset and one use-case schema:

- `classification` stores source images, a class vocabulary, and append-only
  class annotations.
- `mask_refinement` stores source images, optional candidate masks, and
  append-only refined-mask annotations.

`ob_schema` identifies the contract as `oracle_builder_dataset` version `1.5.0`.
`dataset.dataset_id` is the stable lineage UUID. `dataset.revision_id` identifies
one workspace/checkpoint revision, while `parent_revision_id` links a checkpoint
to the workspace revision it captured. All three survive export, import, and
future movement into warm storage.

Schema 1.1 removes `dataset_items.split` entirely. Schema 1.2 adds generic
annotation-workspace tables for ROI labels, reviews, inference provenance, and
derived evidence. Schema 1.3 adds shared taxonomy concepts, Schema 1.4 promotes
classification review and descriptor curation, and Schema 1.5 adds canonical
item geometry. Opening an older V1 database migrates it transactionally.
Opening a 1.0 database
transactionally rebuilds `dataset_items`, discards obsolete stored assignments
and split-hint metadata, updates `ob_schema`, reinstalls lifecycle triggers, and
records a `schema.migrated` event. Frozen datasets remain frozen. Training,
validation, and test membership exists only in a model run's
`protocol/splits.json`.

## Where the executable definition lives

This document is the searchable logical contract. The executable SQLite DDL,
migrations, validation rules, and fingerprint implementation are maintained in
[`oracle_data_contracts/datasets/schema.py`](../oracle_data_contracts/datasets/schema.py).
Applications should use the dataset repository/schema APIs rather than relying
on undocumented SQLite implementation details.

## Common tables

| Table | Purpose |
|---|---|
| `ob_schema` | Schema name and semantic version. |
| `dataset` | Singleton identity, type, descriptive metadata, and lifecycle. |
| `assets` | Content-addressed encoded bytes or an external URI, with checksum and representation metadata. |
| `dataset_items` | Stable item identity, weight, source key, and item metadata. |
| `item_geometry` | Object bounding box and stored crop bounds in one named pixel coordinate space. |
| `metadata_documents` | Parsed JSON plus original sidecar text and checksum. |
| `import_events` | Reproducible importer options and summaries. |
| `dataset_events` | Lifecycle and checkpoint provenance. |
| `annotation_labels` | Shared label vocabulary, including optional hierarchy. |
| `item_label_annotations` | Append-only current/historical labels for any dataset item. |
| `annotation_reviews` | Independent reviewer decisions about label annotations. |
| `inference_runs` | Model-artifact and execution provenance for derived evidence. |
| `evidence_arrays` | Deduplicated typed NPY arrays for logits, embeddings, and outputs. |
| `model_evidence` | Indexed per-item model evidence from one inference run. |

Classification adds `classification_labels`, `classification_items`, and
`classification_annotations`. Mask refinement adds `mask_refinement_items` and
`mask_annotations`.

Annotations are append-only. At most one annotation per item is current, while
older accepted, rejected, or deprecated entries remain available for audit and
review.

`annotation_labels` and `item_label_annotations` are available in both dataset
types. This permits a mask-refinement ROI dataset to carry taxonomic or QC labels
without ceasing to be a mask-refinement dataset. Existing classification-specific
tables remain the training-class vocabulary for current classification datasets.

Model evidence is derived workspace material, not human ground truth. It is not
included in the semantic dataset fingerprint, is excluded from deterministic
training-dataset exchange, and should not be used as a training label without a
human annotation/review step.

Dataset and internally generated entity IDs are canonical UUID text in SQLite.
Folder-imported items use deterministic, dataset-scoped UUIDs. `item_id` remains
opaque text so upstream identifiers such as Pelagia detection IDs can be
preserved losslessly; PostgreSQL should key these by `(dataset_id, item_id)`.
JSON fields contain canonical JSON text and timestamps are UTC ISO-8601 text.
The other logical types map directly to PostgreSQL `uuid`, `jsonb`, and
`timestamptz`.

`item_geometry` uses integer `x, y, width, height` rectangles. For Pelagia ROI
datasets both rectangles use `source_frame_pixels`: `bbox` is the detected
object extent and `crop_bbox` is the possibly padded image crop stored as the
item asset. Importers that have only one compatible rectangle use it for both
concepts; when only a shaped ROI image is available, its full extent is used for
both. Application-specific spatial and provenance documents remain intact in
`dataset_items.metadata_json`; normalization details are recorded separately in
`item_geometry.metadata_json`.

## Lifecycle

`working` datasets are editable. `frozen` datasets are training/release
artifacts. Freeze triggers reject inserts, updates, or deletes to semantic
dataset tables. Provenance events and additive prediction tables may still be
written.

```bash
# Freeze the file in place.
oracle-dataset freeze working.sqlite

# Preferred: keep working.sqlite editable and create a frozen snapshot.
oracle-dataset checkpoint working.sqlite

# Reopen a frozen file only through an explicit command.
oracle-dataset thaw release.sqlite --reason "correct annotation"
```

Checkpointing uses SQLite's backup API, validates source and destination, adds a
UTC timestamp to the default filename, preserves `dataset_id`, assigns the
frozen copy a new `revision_id`, and leaves the source unchanged. The semantic
fingerprint excludes revision/lifecycle identity, timestamps, and events, so
equivalent working and frozen copies have the same content fingerprint.
Thawing branches to a new working `revision_id` whose parent is the frozen
revision, so the lineage remains explicit even though the SQLite file is reused.

Actual model training requires `lifecycle = "frozen"`. Configuration generation,
inspection, validation, and preflight checks may operate on a working dataset.

### Annotation workspace snapshots

Annotation workspaces remain editable. A snapshot is a full frozen SQLite backup
that retains annotations, reviews, inference runs, and evidence arrays:

```bash
oracle-dataset snapshot workspaces/roi-review.sqlite \
  --output snapshots/roi-review-2026-08-07.sqlite \
  --note "Reviewed Calanus batch 4"

oracle-dataset restore-workspace \
  snapshots/roi-review-2026-08-07.sqlite \
  workspaces/roi-review-restored.sqlite \
  --reason "Continue review"
```

Snapshots preserve the dataset UUID, receive a new frozen revision UUID, and
record a `workspace.snapshot.created` event. Restoring forks a new working
revision from the snapshot; it never edits the snapshot in place. In addition to
the canonical `dataset_fingerprint`, a `workspace_fingerprint` captures derived
inference/evidence state for snapshot integrity.

### Training releases from annotation workspaces

Create a separate, self-contained frozen dataset before training from an active
annotation workspace:

```bash
oracle-dataset release-training workspaces/roi-review.sqlite \
  datasets/roi-training-v1.sqlite \
  --name "roi-training-v1"
```

The release receives a new dataset and revision UUID, records the source
workspace ID/fingerprint in metadata, and excludes `inference_runs`,
`model_evidence`, and `evidence_arrays`. Human labels, reviews, masks, source
assets, and metadata are retained.

## Legacy ROI database migration

Mask-refinement entry points recognize the pre-V1 database layout containing
generic `samples` and legacy `mask_annotations` tables. Opening one in the mask
editor, validating it for U-Net use, resolving a segmentation training config,
loading segmentation arrays, checking a saved run's dataset identity for
evaluation or inference, or using the ROI visualizer automatically:

1. creates a timestamped `*.pre-v1-*.sqlite` backup with SQLite's backup API;
2. constructs a new V1 database in a temporary path;
3. separates ROI images and candidate masks into content-addressed assets;
4. migrates every item and every historical mask annotation;
5. makes the latest accepted annotation current and preserves its parent chain;
6. retains non-conflicting additive tables such as prediction tables;
7. validates schema, foreign keys, and item/annotation counts; and
8. atomically replaces the requested database only after validation succeeds.

Legacy train/validation/test values are discarded because split membership
belongs to a model-run protocol. The migration report records how many values
were removed. The migrated dataset remains `working`; create a frozen
checkpoint before training.

U-Net dataset analysis is likewise dataset-focused: it reports total and
annotated item counts, missing current masks, complete annotation-history
counts, candidate-mask coverage, usable samples, and observed shapes. Split
counts are not reported until a model run creates its own split manifest. The
analysis report also includes the dataset/revision UUIDs, schema version,
lifecycle state, semantic fingerprint, schema-validation result, and the
migration receipt when opening the database triggered an automatic upgrade.

Migration can also be requested explicitly:

```bash
python3 -m oracle_builder.datasets.cli migrate-roi \
  datasets/unet_training.sqlite
```

Use `--backup PATH` to choose the backup location. A timestamped JSON migration
report is written beside the migrated database.

## Metadata

The `dataset` row standardizes name, title, description, version, type,
lifecycle, schema version, UUID, and timestamps. Additional source metadata is
preserved as JSON. Top-level `.toml`, `.json`, `.yaml`, and `.yml` files are
stored losslessly in `metadata_documents` with both parsed JSON and original
text.

Metadata can also be attached to an existing working database independently of
an image import:

```bash
oracle-dataset metadata-add dataset.sqlite metadata.toml --actor "$USER"
```

This validates TOML, JSON, YAML, or YML input and either adds the source filename
as a new logical document or replaces the document selected with `--name`.
Original text, parsed JSON, source format, SHA-256, and an add/update provenance
event are retained. Because metadata is semantic dataset content, this operation
changes the dataset fingerprint and is rejected while the dataset is frozen.

A canonical folder metadata file begins:

```toml
[oracle_builder]
schema_name = "oracle_builder_dataset"
schema_version = "1.2.0"
export_format_version = "1.0.0"
dataset_id = "11111111-1111-4111-8111-111111111111"
revision_id = "22222222-2222-4222-8222-222222222222"
dataset_type = "classification"

[dataset]
name = "whoi-plankton-2012"
title = "WHOI Plankton 2012"
description = "Source and curation description."
version = "1"
```

TOML arrays require quoted strings, and every repeated creator must have its own
`[[dataset.creators]]` header.

## Deterministic folder exchange

```bash
oracle-dataset export release.sqlite release-folder
oracle-dataset import release-folder restored.sqlite
```

Exports never overwrite an existing directory. They contain:

- `metadata.toml` for people;
- `manifest.json` for exact reconstruction;
- `checksums.sha256` for transport verification;
- `metadata/source/` for original metadata sidecars;
- classification images under `images/{class}/`;
- mask-refinement images under `images/`, candidate masks under
  `candidate_masks/`, and every annotation revision under `annotations/`.

Import verifies all exported checksums, recreates exact identifiers and
annotation history, and verifies the source semantic fingerprint.

Train/validation/test membership belongs to a model run's experimental
protocol. It is stored in `protocol/splits.json` inside the model-run artifact,
not in the canonical folder exchange or semantic dataset fingerprint.

When the convenient class-folder importer receives a conventional top-level
`train/`, `validation/`, and/or `test/` layout, it records those names only as
per-item `source_partition` provenance. The run's `data.split_strategy` decides
whether to materialize that provenance into its own split manifest.

The named-class folder importer remains the convenient ingestion path for de
facto classification libraries. The canonical export format is the lossless,
deterministic interchange format.

## PostgreSQL warm-storage mapping

The logical tables and foreign keys are backend-neutral. A future PostgreSQL
adapter should:

- scope revision-owned rows by `(dataset_id, revision_id)`;
- map UUID text to `uuid`, JSON text to `jsonb`, timestamps to `timestamptz`,
  and embedded payloads to `bytea` or an object-store URI;
- enforce one current annotation with the same partial unique indexes;
- scope every revision-owned repository query by `dataset_id` and `revision_id`;
- retain the same schema version, validation, fingerprint, repository, and
  import/export interfaces.

Application and training code should depend on the repository contract, not raw
SQLite-specific row layout. SQLite remains the portable cold-storage artifact;
PostgreSQL becomes a multi-dataset warm store without changing dataset meaning.

## Breaking change from legacy databases

V1 intentionally does not create a compatibility `samples` view or silently
reinterpret a legacy generic database. Opening a database that has the old
`samples` table but no V1 schema raises a migration error. Re-import source
folders with `oracle-import-classification`, or export legacy content with the
older Oracle Builder revision before converting it into a V1 use-case dataset.
