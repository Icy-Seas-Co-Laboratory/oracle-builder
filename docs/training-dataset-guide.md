# Building training-set databases

Oracle Builder stores one typed dataset per SQLite file. Build an editable
working database first, validate and curate it, then make a frozen checkpoint
for training. Keep the working copy: a checkpoint is a reproducible release,
not the place to make corrections.

This guide covers the two supported starting points:

- A **classification** dataset from a locally maintained, labeled image-folder
  library.
- A **mask-refinement** dataset created from local ROIs or Pelagia detections
  and annotated in Mask Builder.

For an API or another application that creates datasets programmatically, use
the Dataset V1 repository/schema APIs rather than writing SQLite tables
directly. The schema reference is the contract for that integration.

## Choose the dataset type before collecting data

| Use case | Dataset type | One item contains | Target used for training |
| --- | --- | --- | --- |
| Assign exactly one category to each image or ROI | `classification` | Original image bytes, provenance, and a current class annotation | Class label |
| Correct or create an object mask for each ROI | `mask_refinement` | ROI image, optional candidate mask, provenance, and annotation history | Current accepted mask |

Do not use a classification folder hierarchy to represent multiple labels per
image, detection boxes, or segmentation masks. Duplicate membership across
class folders creates duplicate-content conflicts and can leak the same image
into more than one run split. Use a separate, clearly defined dataset or an
annotation-workspace workflow when the target is not one class per image.

## Build a classification database from scratch

### 1. Plan the label vocabulary

Decide what one label means before moving files. Labels should be mutually
exclusive at the intended training level and should describe the object or
scene, not an accidental acquisition condition. For example, use `copepod`,
`diatom`, and `detritus` only if every image can consistently be assigned one
of those categories. Add an explicit `unknown`, `other`, or `non_target` class
when that is a real operational outcome; do not silently discard such examples.

Use stable, machine-friendly folder names: lowercase ASCII words separated by
hyphens or underscores, such as `calanus_finmarchicus` or `non_target`.
Folder names become the initial label names. Renaming a folder after import
does not rename the existing database label.

Keep a short label-definition document with inclusion/exclusion examples,
source scope, and the person or rule that assigned labels. This is especially
important for visually adjacent classes.

### 2. Create the source folder library

Start with one directory per class:

```text
training-library/
├── calanus_finmarchicus/
│   ├── calanus_finmarchicus_000001.jpg
│   └── cruise07_station12_frame004182_roi003.png
├── detritus/
│   └── cruise07_station12_frame004199_roi001.png
├── non_target/
│   └── cruise07_station13_frame001011_roi004.jpg
└── metadata.toml
```

Image files may be nested inside a class folder, which is useful for grouping
by cruise, acquisition date, or batch. The class is still the first directory
under the library root. Only supported image files are collected; do not put
sidecar exports, thumbnails, or non-image attachments under a class folder.

Prefer filenames that retain a source identifier and are unique within their
class. A durable pattern is:

```text
<source-or-project>_<collection-or-batch>_<frame-or-image-id>_<roi-or-object-id>.<ext>
```

For example, `cruise07_station12_frame004182_roi003.png`. Avoid names such as
`final.jpg`, `copy 2.png`, or a sequence that is renumbered whenever files are
removed. The importer records the relative source path and creates a
deterministic item UUID from it, scoped to the dataset UUID. Moving a file or
renaming it therefore makes it a different source item on a later import.

### 3. Add optional metadata

Place `metadata.toml`, JSON, YAML, or YML sidecars at the library root. They
are preserved in the database as metadata documents. `metadata.toml` can also
populate the dataset name, title, description, version, and imaging metadata:

```toml
[dataset]
name = "cruise07-zooplankton"
title = "Cruise 07 zooplankton ROIs"
description = "Human-reviewed ROI crops collected in July 2026."
version = "0.1.0"

[imaging]
source_polarity = "light_on_dark"
```

Use `source_polarity` only when it is known for the complete library. Valid
values include `light_on_dark`, `dark_on_light`, `mixed`, and `unknown`. If it
is omitted, the importer makes a conservative estimate. `mixed` or `unknown`
does not cause automatic inversion during training.

Do not put secrets, access tokens, personally identifying information, or
unlicensed source content in a sidecar: its raw text is stored with the
dataset and travels with exports.

### Add or replace TOML records after import

You can attach a TOML record to an already formed **working** training set;
there is no need to re-import its images. This is useful for recording a label
guide, collection/batch provenance, acquisition settings, licensing notes, or
a curation decision made after initial intake.

```bash
uv run oracle-dataset metadata-add \
  datasets/cruise07-zooplankton.sqlite \
  records/collection-and-labels.toml \
  --actor "$USER"
```

The source filename becomes the logical record name by default. To replace a
specific previously attached record while retaining a stable logical name, use
`--name`:

```bash
uv run oracle-dataset metadata-add \
  datasets/cruise07-zooplankton.sqlite \
  records/collection-and-labels-v2.toml \
  --name collection-and-labels.toml \
  --actor "$USER"
```

Oracle Builder validates TOML and stores both its original text and parsed
content, along with a checksum and add/update provenance event. The operation
changes the dataset fingerprint, so validate and create a new checkpoint after
any metadata change:

```bash
uv run oracle-dataset validate datasets/cruise07-zooplankton.sqlite
uv run oracle-dataset checkpoint datasets/cruise07-zooplankton.sqlite \
  --output datasets/cruise07-zooplankton.v2.sqlite
```

Metadata cannot be added to a frozen training database. Preserve the old
frozen release for reproducibility, apply the record to its editable working
ancestor, and checkpoint a new version. If the frozen file is the only copy,
first make a safe copy and thaw that copy with `oracle-dataset thaw COPY.sqlite
--reason "add collection metadata"`; never thaw the release used by an
existing run merely to attach documentation.

`metadata-add` stores a metadata document; it does not revise the database's
top-level dataset name, title, description, or version fields after creation.
Choose the document's purpose and name accordingly, and use an application or
schema API when those identity fields themselves must change.

### 4. Review the library, then import

Run a dry run first. It reports class counts, unreadable images, duplicate
content, inferred polarity, and the generated label map without creating a
database.

```bash
uv run oracle-import-classification \
  --input /path/to/training-library \
  --output datasets/cruise07-zooplankton.sqlite \
  --minimum-images-per-class 20 \
  --dry-run
```

When the report is acceptable, run the same command without `--dry-run`:

```bash
uv run oracle-import-classification \
  --input /path/to/training-library \
  --output datasets/cruise07-zooplankton.sqlite \
  --minimum-images-per-class 20

uv run oracle-dataset validate datasets/cruise07-zooplankton.sqlite
uv run oracle-dataset info datasets/cruise07-zooplankton.sqlite
```

The importer preserves original encoded bytes by default, which is normally
the right choice. Use `--storage-mode materialized --input-shape H W C` only
when intentionally making a fixed, preprocessed representation. Use
`--require-rgb` when every input must be RGB; otherwise grayscale is accepted
by default. For a source library with known dark-on-light acquisition, pass
`--source-polarity dark_on_light` instead of relying on estimation.

For classification runs, `resize_mode = "fit_pad_max_2x"` or
`resize_mode = "fit_pad_max_3x"` preserves aspect ratio and pads to the
requested input shape, but limits enlargement of a small ROI to 2× or 3×.
Both still downscale an oversized ROI enough to fit. Use a capped mode when
strong enlargement would turn low-resolution ROIs into blurred full-frame
inputs; use ordinary `fit_pad` when all inputs should fill as much of the
network input as possible.

By default, identical image content is skipped. Content assigned to different
classes is an error unless `--duplicate-policy allow` is deliberately chosen;
in nearly all curated datasets it should be corrected at the source.

### 5. Freeze a training release

After label review and validation, checkpoint the working dataset:

```bash
uv run oracle-dataset checkpoint datasets/cruise07-zooplankton.sqlite \
  --output datasets/cruise07-zooplankton.v1.sqlite
```

Train from `cruise07-zooplankton.v1.sqlite`, not the editable file. Every run
records the frozen dataset identity and fingerprint together with its exact
split manifest.

## Import a pre-organized folder library

The same importer accepts an existing class-folder library. There are two
recognized layouts.

### Class folders only

```text
library/
├── copepod/
│   └── batch-a/copepod_001.tif
└── diatom/
    └── batch-a/diatom_001.tif
```

Each top-level directory is a class. Nested directories are organization only.

### Source partitions plus class folders

```text
library/
├── train/
│   ├── copepod/copepod_001.jpg
│   └── diatom/diatom_001.jpg
├── validation/
│   ├── copepod/copepod_002.jpg
│   └── diatom/diatom_002.jpg
└── test/
    ├── copepod/copepod_003.jpg
    └── diatom/diatom_003.jpg
```

`train`, `validation` (or `val`), and `test` are recognized only when at least
two of those top-level directories are present. They are stored as per-item
`source_partition` provenance, not as permanent dataset splits. A run may use
a complete source-partition layout with `data.split_strategy = "auto"`, or you
can request `"source_partitions"`; `"random"` deliberately ignores it. The
run artifact's split manifest—not these folders—is the authoritative training,
validation, and test assignment.

Keep related images together when defining source partitions. Near-duplicate
crops, adjacent video frames, or ROIs from the same physical object should not
be scattered across partitions, because that gives deceptively optimistic
evaluation results. If the provided partitions do not meet that standard,
import them as provenance and use a carefully designed run split instead.

For imports into an existing working database, preserve the current label map.
New class names require `--allow-new-classes` or a deliberately supplied
contiguous JSON `--label-map`. Review `--existing-policy` (`skip` by default)
before re-importing a changed library: the item identity derives from the
relative path, so a rename is not an in-place update.

## Build a mask-refinement database from scratch

Mask refinement is an annotation workflow, not a paired-image folder importer.
Start an editable workspace and add local ROIs or Pelagia detections; the
database retains original ROI assets, optional candidate masks, and an
append-only history of validated masks.

```bash
uv run python mask_builder.py \
  --image /path/to/roi-images \
  --database workspaces/roi-review.sqlite
```

Use a filename convention that preserves source and ROI identity, as for the
classification workflow. Keep original images in a stable input folder and
treat the SQLite workspace as the annotation record. Do not replace it with a
folder of exported masks: review history, source provenance, and candidate-mask
context are part of the dataset.

Review masks in Mask Builder, validate the workspace for the chosen U-Net input
and output shapes, then create a separate frozen training release:

```bash
uv run python mask_builder.py \
  --database workspaces/roi-review.sqlite \
  --validate-unet-dataset \
  --unet-input-shape 256,256,2 \
  --unet-output-shape 256,256,1

uv run oracle-dataset release-training workspaces/roi-review.sqlite \
  datasets/roi-training.v1.sqlite --name roi-training-v1
```

Use `checkpoint` for a direct mask dataset and `release-training` for an active
annotation workspace that contains derived inference/evidence material. See
the mask-refinement workflow for Pelagia imports, editing, and validation.

## Dataset quality checklist

Before freezing any training database, confirm:

- Each item has a clear target under the selected use case and labels/masks
  follow a written annotation rule.
- Classes are sufficiently represented for the intended evaluation, including
  operationally important rare or negative cases.
- Source duplicates and near-duplicates have been reviewed, especially across
  labels and proposed evaluation groups.
- Image mode, dimensions, polarity, acquisition changes, and preprocessing
  requirements have been inspected rather than assumed.
- Metadata identifies source, collection dates/batches, licensing or consent
  constraints, label definitions, and known limitations.
- The working database passes `oracle-dataset validate`; its frozen checkpoint
  or release has a versioned, descriptive filename.
- Evaluation will represent deployment conditions: source, time, location,
  instrument, and object-level leakage are all considered when choosing a
  split protocol.

## Useful references

- [Classification workflow](classification-workflow.md) for training and
  evaluation after import.
- [Mask-refinement workflow](mask-refinement-workflow.md) for ROI intake and
  annotation.
- [Dataset schema V1](dataset-schema-v1.md) for lifecycle, provenance, and
  programmatic-integration contracts.
