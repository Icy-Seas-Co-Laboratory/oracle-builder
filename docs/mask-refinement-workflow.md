# Mask-refinement workflow

This workflow adds Pelagia detections or local images to an editable
`mask_refinement` SQLite database, reviews candidate masks in napari, and
produces a frozen U-Net training dataset. The normal starting point is an
existing working database: source ROIs remain traceable through their Pelagia
detection IDs and are not overwritten merely by opening the editor.

## 1. Install the editor and choose a working database

```bash
python3 -m pip install -r requirements-gui.txt

# Start a local-image workspace, or open an existing one.
python3 mask_builder.py --database workspaces/roi-review.sqlite
```

`--database` is both the input and output workspace. With no image or Pelagia
option, it opens existing items. Add `--missing-masks-only` to focus review on
items that do not yet have a current validated mask:

```bash
python3 mask_builder.py \
  --database workspaces/roi-review.sqlite \
  --missing-masks-only
```

Use `--uuid ITEM_UUID` to open one known item. The bottom filmstrip is numbered
in queue order; left/right arrows move between images. In the editor, `A`, `F`,
`R`, and `L` apply threshold, fill holes, remove small objects, and keep the
largest component, respectively.

## 2. Connect to Pelagia safely

The Pelagia API base URL defaults to `http://localhost:8000`; set
`--api-base-url` when using another instance. Authentication supports either a
pre-issued bearer token or a login request. Keep credentials out of shell
history, command lines, notebooks, and committed files by placing them in your
session environment or your organization’s approved secret manager.

```bash
# Preferred when you already have a bearer token.
export PELAGIA_API_TOKEN='…'

# Or use a Pelagia login. Supply both values; the project key is optional.
export PELAGIA_USERNAME='…'
export PELAGIA_PASSWORD='…'
export PELAGIA_PROJECT_KEY='your-project-key'
```

The corresponding command-line options are `--api-token`, `--api-username`,
`--api-password`, and `--api-project-key`. CLI options take precedence over
environment variables. The token/login information is used only for the
Pelagia request; it is not deliberately stored in the SQLite dataset.

Before opening the editor, list the available detections and refine the query:

```bash
python3 mask_builder.py \
  --api-base-url https://pelagia.example.org \
  --list-api-rois \
  --api-run-id RUN_UUID \
  --api-collection COLLECTION_NAME \
  --api-limit 50 \
  --api-sort-by asset_frame --api-sort-dir asc
```

The list prints detection IDs plus asset, frame, ROI index, area, and mask
payload size. Copy a detection ID for a one-off import, or use the browsing
mode described next.

## 3. Add Pelagia ROIs to an existing database

For a known detection, open it and add it to the working database:

```bash
python3 mask_builder.py \
  --api-base-url https://pelagia.example.org \
  --api-roi-id DETECTION_UUID \
  --database workspaces/roi-review.sqlite
```

For routine annotation, browse a filtered queue. The first matching ROI opens
immediately; subsequent ROIs are fetched as you navigate. Saving an ROI writes
it to `--database`, together with Pelagia metadata and the candidate mask.

```bash
python3 mask_builder.py \
  --api-base-url https://pelagia.example.org \
  --api-browse-rois \
  --database workspaces/roi-review.sqlite \
  --api-run-id RUN_UUID \
  --api-collection COLLECTION_NAME \
  --api-min-area 500 \
  --api-min-bbox-w 32 --api-min-bbox-h 32 \
  --api-limit 100 \
  --api-sort-by asset_frame --api-sort-dir asc
```

Use `--random-api-roi` instead of `--api-browse-rois` to draw a random first
item from the filtered result set. Do not combine either option with
`--api-roi-id`.

Opening a detection whose UUID is already in the database updates its imported
image/candidate representation; saving adds a new annotation-history entry
rather than erasing previous annotation history. Use the editor’s **Duplicate
image** button only when a separate training example is intentional—it creates
a new UUID. **Delete image** removes the image and its dependent annotations
from the active database after confirmation.

### Common Pelagia filters

Use `--list-api-rois` with the same filters first whenever a query could return
many items. All filters are optional and are passed to Pelagia’s detections
endpoint.

| Purpose | Options |
| --- | --- |
| Restrict the source | `--api-run-id`, `--api-asset-id`, `--api-collection`, `--api-frame-id` |
| Limit position in a source | `--api-start-frame`, `--api-end-frame`, `--api-roi-index` |
| Restrict bounding-box location | `--api-min-bbox-x`, `--api-max-bbox-x`, `--api-min-bbox-y`, `--api-max-bbox-y` |
| Restrict size | `--api-min-bbox-w`, `--api-max-bbox-w`, `--api-min-bbox-h`, `--api-max-bbox-h`, `--api-min-area`, `--api-max-area`, `--api-min-perimeter`, `--api-max-perimeter` |
| Restrict encodings | `--api-roi-encoding`, `--api-roi-format`, `--api-mask-encoding`, `--api-mask-format` |
| Page and order results | `--api-limit`, `--api-offset`, `--api-sort-by`, `--api-sort-dir` |

For example, inspect rectangular ROIs from one asset before adding them:

```bash
python3 mask_builder.py \
  --list-api-rois \
  --api-asset-id ASSET_UUID \
  --api-min-bbox-w 64 --api-max-bbox-w 512 \
  --api-min-bbox-h 64 --api-max-bbox-h 512 \
  --api-offset 0 --api-limit 25 \
  --api-sort-by asset_frame --api-sort-dir asc
```

The editor’s `validated mask` layer is the editable result. `Save and next`,
`Skip`, the numbered thumbnail navigator, and arrow keys support queue review
and revisiting ROIs. Candidate masks and annotation history are retained for
audit.

## 4. Local-image imports

For local files, use an existing database exactly as above:

```bash
python3 mask_builder.py \
  --image images/to_mask \
  --database workspaces/roi-review.sqlite
```

Legacy ROI databases are automatically backed up and migrated to Dataset V1
when opened. Details are in the [dataset schema reference](dataset-schema-v1.md).

For an active review workspace, save a complete immutable milestone without
locking the editable source:

```bash
oracle-dataset snapshot workspaces/roi-review.sqlite \
  --output snapshots/roi-review-milestone.sqlite \
  --note "Completed first-pass review"
```

## 5. Validate, configure, and freeze

```bash
python3 mask_builder.py \
  --database workspaces/roi-review.sqlite \
  --validate-unet-dataset \
  --unet-input-shape 256,256,2 \
  --unet-output-shape 256,256,1

python3 mask_builder.py \
  --database workspaces/roi-review.sqlite \
  --write-unet-config configs/unet_training.toml \
  --unet-input-shape 256,256,2 \
  --unet-output-shape 256,256,1

oracle-dataset checkpoint workspaces/roi-review.sqlite \
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

## 6. Train

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

## 7. Inspect results

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
