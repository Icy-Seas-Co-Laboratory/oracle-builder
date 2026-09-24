# Oracle Builder pipeline options reference

This is the configuration reference for Oracle Builder's three operational
pipelines: model training, batch inference, and Oracle Serve. It is intended
to answer *what does this switch do?*; the workflow guides explain dataset
creation and end-to-end commands. Values shown as defaults are the resolved
defaults in `oracle_builder.config.DEFAULT_CONFIG`, unless noted otherwise.

## Before choosing options

Training always consumes a **frozen** SQLite dataset and writes a sealed,
portable run artifact. The resolved configuration (including inferred class
count, concrete polarity, and split assignment) becomes part of that artifact;
changing a TOML file later does not alter an existing run.

Use `python model_training.py --dry-run -c CONFIG --input DATASET.sqlite
--output NAME` to see the resolved configuration. Use `--preflight` for a
segmentation dataset compatibility check, and `--resume RUN_DIR` only for an
interrupted non-embedding run with a valid recovery snapshot.

## 1. Model training

### Options shared by training pipelines

| Section / option | Choices or default | Meaning and consideration |
|---|---|---|
| `run.task` | `segmentation`, `classification`, `embedding` | Selects the pipeline and required data contract. `embedding` is representation-only, not legacy clustering. |
| `run.model` | See task sections | Architecture identifier; required. |
| `run.seed` | `123` | Seeds split generation and stochastic training. Keep it fixed for comparable experiments. |
| `run.notes` | `""` | Free-text provenance stored with the artifact. |
| `data.input_shape` | required | `[height, width]` or `[height, width, channels]` for classification/embedding; segmentation requires channels. Larger images improve detail but sharply increase memory. |
| `data.batch_size` | `16` | Global batch size. With multi-GPU training it must divide evenly across replicas. |
| `shuffle_buffer` | `512` | Number of streaming examples used to randomize order; larger improves mixing and consumes host memory. |
| `validation_split`, `test_split` | `0.2`, `0.1` | Fractions for a deterministic random split when source partitions are not used. Test data is reserved for final evaluation. |
| `split_strategy` | `auto` | `auto` honors a complete imported `train`/`validation`/`test` layout, otherwise randomizes; `random` always randomizes; `source_partitions` requires a complete source layout. |
| `data.streaming.*` | enabled; 4 workers; 2 prefetch batches; deterministic; 65536 KiB cache | Classification/embedding loader controls. Turn `deterministic` off only when throughput matters more than repeatable ordering. Tune workers/prefetch for slow storage; both must be positive. |
| `distribution.strategy` | `auto` | `auto` selects visible GPUs/MirroredStrategy then can fall back; `single`/`none` use one device; `mirrored` requires distributed execution; `cpu` forces CPU. `devices`, `cross_device_ops` (`auto`, `nccl`, `hierarchical_copy`), `fallback_to_single`, and `memory_growth` refine it. |
| GPU selection controls | `unused_first`; safe fallbacks off | `require_unused_gpu`, `allow_busy_fallback`, `gpu_light_share_memory_mb`, `gpu_light_share_utilization_percent`, and `gpu_lease_directory` avoid stealing a busy GPU. Favor the defaults in shared environments. |
| `training.epochs`, `optimizer`, `learning_rate` | `10`, `adam`, `0.001` | Optimizer accepts `adam`, `sgd`, or a Keras optimizer name. Lower the rate for unstable/large models. |
| `training.display` | `rich` | `rich`, `text`, or `off`; display changes console output, not saved metrics. |
| `callbacks.*` | early stopping/LR reduction off | Enable `early_stopping` and/or `reduce_lr_on_plateau`; use `early_stopping_patience` and `checkpoint_monitor` (normally `val_loss`). |
| `recovery.*` | enabled, each epoch | Writes one rolling full-state snapshot. `save_every_epochs` must be positive. Keep enabled for long runs you may resume. |
| `output.*` | checkpoints off; predictions/figures/SavedModel on | `save_checkpoints` retains each epoch; `save_predictions`, `save_figures`, and `export_savedmodel` trade disk space/portability for artifacts. |
| `evaluation.segmentation_threshold` | `0.5` | Binary-mask decision threshold. Validate it on held-out data rather than assuming 0.5 is optimal. |
| `evaluation.benchmark.*` | on; 2 warmup, 10 measured batches | Measures serving throughput. Warmup may be zero; measured batches must be positive. |
| `evaluation.uncertainty.*` | off | Grouped bootstrap intervals: set `group_metadata_key`, replicates, confidence level, and seed. Use a real dependency group such as `cruise_id`; do not bootstrap correlated images independently. |

### Preprocessing and supervised augmentation

`[preprocessing]` is a serving contract: the same transformation is applied
during training and later inference. `resize_mode` is `fit_pad` (safe default),
`fit_pad_max_2x`, `fit_pad_max_3x` (cap enlargement of small ROIs),
`center_pad`/`center_roi_pad` (never enlarge; center the native-detail ROI on
the canvas), `fill_crop`, `center_crop`, `stretch`, `none`, or legacy `fit`.
Choose `fit_pad` when aspect ratio carries meaning; use crop/stretch only when
their distortion is accepted.

Padding and crops are independently anchored with `pad_anchor` and
`crop_anchor`: `center`, each compass edge, or `top_left`, `top_right`,
`bottom_left`, and `bottom_right`. `pad_mode` is `constant` (using
`pad_value`), `edge`, `reflect`, or `symmetric`; use `edge` or `constant` for
one-pixel ROIs. `upscale_limit` is a general positive cap for `fit_pad` when
the fixed 2×/3× aliases do not match the experiment. For example:

```toml
[preprocessing]
resize_mode = "center_pad" # preserve native ROI detail, then center/pad
pad_mode = "edge"
pad_anchor = "center"
```

`normalization` is `dtype`, `minmax`, `percentile`, or `none`. `rescale`,
`pad_value`, `percentile_low`, and `percentile_high` refine it. `invert` is
resolved to a Boolean (`auto` is accepted in source TOML and follows recorded
dataset polarity). `interpolation` is `nearest`, `bilinear`, `bicubic`, or
`lanczos`; masks normally need nearest-neighbor behavior while intensity images
usually benefit from bilinear. `channel_mode` is `auto`, `grayscale`, `rgb`,
or `rgba`.

For classification/embedding, `preprocessing.derived_channels.gradient_magnitude`
and `.local_contrast` add a Sobel morphology and/or local-texture channel;
`.local_contrast_sigma` controls texture scale. A three-channel input with both
enabled is grayscale + gradient + local contrast. The declared channel count
must match that result; callers never supply derived channels themselves.

`[augmentation]` applies only to the training split. `enabled`,
`repeats_per_epoch`, `invert`, `rotation`, `zoom`, `translation` (one number or
`[x,y]`), `skew`, horizontal/vertical flips, `brightness`, `contrast`,
`gaussian_noise`, and `fill_value` control image perturbations. In segmentation,
`mask_fill_value`, `photometric_channels`, and `mask_input_channels` protect
mask/candidate channels: geometric transforms are shared with masks, while
photometric transforms are not. Do not enable transformations that change a
label's meaning (for example, vertical flips when orientation is diagnostic).

Example: a small-ROI classifier that preserves native detail and adds texture:

```toml
[data]
input_shape = [128, 128] # resolves to three channels below

[preprocessing]
resize_mode = "fit_pad_max_2x"
channel_mode = "grayscale"

[preprocessing.derived_channels]
gradient_magnitude = true
local_contrast = true
local_contrast_sigma = 3.0
```

### 1.1 Segmentation

Segmentation refines ROI masks. Its `data.output_shape` is required and its
spatial dimensions must match `input_shape` when tiling is enabled. Models are
`unet`, `residual_unet`, and `unet_plus_plus`. Their shared model controls are
`base_filters` (capacity), `depth`, `dropout`, `activation`, and
`final_activation` (normally `sigmoid`); U-Net++ additionally supports
`deep_supervision`. Start with U-Net; use residual U-Net for a stronger encoder
and U-Net++ when nested skip connections justify the extra cost.

Input can be image-only, image plus candidate mask, or image + candidate mask
plus distance. `data.candidate_distance` is `none`, `euclidean_sdf`, or
`geodesic`; the older Boolean `candidate_sdf` remains compatible. Distance
requires a 3-channel segmentation input and positive `candidate_distance_clip`.
For geodesic distance, `[data.geodesic_distance]` has `epsilon`,
`intensity_weight`, `intensity_gamma`, `gradient_weight`, and connectivity `4`
or `8`. Geodesic distance costs more, but respects image boundaries better than
an ordinary signed-distance field.

`training.segmentation_target` is `validated_mask` (predict the accepted mask)
or `candidate_delta` (learn correction from a candidate). Delta mode requires
2 channels without distance or 3 with distance. Loss can be a standard Keras
binary loss, `bce_soft_dice` (with `bce_weight`, `soft_dice_weight`,
`soft_dice_smooth`), or `bce_soft_tversky` (with `bce_weight`,
`soft_tversky_weight`, `tversky_alpha`, `tversky_beta`, and smooth). A higher
Tversky beta penalizes false negatives more and tends to expand masks.
`spatial_edge_weighting`, `edge_weight_lambda`, and `edge_weight_sigma` focus
loss near boundaries; use it for thin/precise edges, not as a substitute for
correct masks.

`[tiling]` is segmentation-only: `enabled`, `overlap_fraction` `[0,1)`,
`blend_mode` (`uniform` or seam-reducing `hann`), `tile_large_rois_only`, and
`normalize_training_coverage`. It permits large ROI inference without changing
the trained receptive field.

```toml
[data]
input_shape = [256, 256, 3] # image, candidate mask, geodesic distance
output_shape = [256, 256, 1]
candidate_distance = "geodesic"

[training]
segmentation_target = "candidate_delta"
loss = "bce_soft_tversky"
tversky_alpha = 0.3
tversky_beta = 0.7

[tiling]
enabled = true
overlap_fraction = 0.5
blend_mode = "hann"
```

### 1.2 Classification

Classification predicts a dataset label and can preserve a serving embedding
and nearest-neighbor evidence. Available native architectures are `simple_cnn`,
`resnet_like`, `densenet_like`, `resnet`/`resnet18`…`resnet152`, `densenet`/
`densenet121`/`169`/`201`, `efficientnet`/`efficientnet_b0`…`b7`, and native
`efficientnet_v2`/`efficientnet_v2_b0`…`b3` plus `efficientnet_v2_s`/`m`/`l`.
SimpleCNN is the fast baseline; ResNet is a dependable default; DenseNet trades
more memory for feature reuse; EfficientNet targets accuracy per compute.

All families accept `embedding_dim`, `normalize_embeddings`, and usually
`dropout`; simple families expose `base_filters`. ResNet exposes `variant`,
`stem_kernel_size`, `stem_stride`, `stem_pool`, and `base_filters`. DenseNet
adds `variant`, `stem_*`, `growth_rate`, `initial_filters`,
`bottleneck_multiplier`, `compression`, and `block_config`. EfficientNet adds
`variant`, `width_coefficient`, `depth_coefficient`, `stem_filters`,
`top_filters`, `stem_kernel_size`, `stem_stride`, `dropout`, and `se_ratio`.
For ResNet and DenseNet, a custom `block_counts`/`block_config` is four positive
integers; their variants are respectively `18/34/50/101/152` and `121/169/201`.
EfficientNet variants are B0–B7. `model.auxiliary_features` can append standardized scalar ROI or
metadata features; each entry requires `name`, `source`, optional `transform`,
`standardize`, and missing-value policy. Auxiliary features cannot currently be
combined with self-supervised pretraining.

Use `sparse_categorical_crossentropy` for balanced labels or
`weighted_sparse_categorical_crossentropy` for imbalance. Weighted loss uses
`training.class_weights.mode` (`explicit`, `inverse_frequency`,
`effective_number`), `beta`, `normalize`, and `values`; weights are calculated
from the training split only. Metrics normally include `accuracy` and
epoch-wide `macro_f1`. `[evidence] enabled` and `knn_k` retain embeddings,
prototypes, and KNN context for predictions; turn off if storage/privacy costs
outweigh explanation value.

#### Optional self-supervised initialization

`[self_supervised]` (legacy `[pretraining]` still reads) initializes the same
classifier before supervised fitting. Set `enabled`; methods are `byol`
(or alias `student_teacher`), `simclr`, and for segmentation only
`grayscale_reconstruction`. BYOL is less sensitive to batch size; SimCLR uses
in-batch negatives and benefits from large effective batches. Options are
`epochs`, `verbose` (0/1/2), `learning_rate`, `teacher_momentum`,
`projection_dim`, `projection_hidden_dim`, `temperature`, `ssl_optimizer`
(`adam`/`adamw`), `weight_decay`, `minimum_global_batch_size`, collapse
thresholds, VICReg weights/target standard deviation, and SimCLR encoder
variance/covariance/target controls. `[self_supervised.augmentation]` has the
same perturbation controls as `[augmentation]`, but is independent of it.
`embedding_health` enables sampled effective-rank, pairwise-cosine, and
direction-norm safety checks; tune its sample/pair counts and thresholds only
when you understand the representation distribution.

### 1.3 Embedding-only

Embedding-only training writes a reusable encoder and training record; it does
not fit clusters or invent labels. Use `oracle-embed --config CONFIG --input
DATASET.sqlite --output DIR` (or the embedding path in `model_training.py`).
It requires a classification-style frozen dataset, `run.task = "embedding"`, a
native classifier-family encoder, positive `embedding_dim`, and enabled
`[self_supervised]`. BYOL is the practical default; SimCLR often needs a larger
batch. Configure preprocessing, distribution, self-supervision, inference
extraction batch, and output as above; supervised `training.loss` is still
present for config compatibility but embeddings are learned in the SSL phase.

### 1.4 Stratified classification

Resolution stratification trains one shared-weight classifier across square
input strata, routing each original ROI to the smallest configured dimension
that contains its largest side. It is classification-only, supports native CNN
families listed above, and cannot currently combine with self-supervision.

`classification.stratification.enabled` turns it on. `dimensions` must be
ascending, unique integers >= 2. The only supported `basis` is
`max_original_dimension`, `batch_size_policy` is `constant_input_tensor`, and
`weight_sharing` is `shared`. Use `schedule = "interleaved_steps"` to alternate
optimizer steps by stratum; `steps_per_stratum` controls the number of
consecutive batches a stratum receives before the next stratum's turn and
`supra_epochs` controls mixed global epochs between validation passes.
`normalization = "group"` avoids small-microbatch BatchNorm drift, while
`conditioning` gives the shared network a learned stratum cue without masking
classes. Keep training routing disabled for the initial canonical-routing
baseline. Larger strata get smaller microbatches to retain the
smallest-stratum tensor budget.

```toml
[classification.stratification]
enabled = true
dimensions = [32, 64, 128]
basis = "max_original_dimension"
weight_sharing = "shared"
schedule = "interleaved_steps"
steps_per_stratum = 1
supra_epochs = 1
normalization = "group"
group_norm_groups = 8

[classification.stratification.conditioning]
enabled = true
embedding_dim = 16

[classification.stratification.training_routing]
enabled = false

[classification.stratification.cycle_scheduler]
aggregation = "equal_strata"
guardrail_metric = "macro_f1"
max_stratum_drop = 0.03
```

## 2. Batch inference

`model_inference.py` runs a sealed artifact over SQLite records and writes a
prediction SQLite database. Required arguments are `--run`, `--input`, and
`--output`; `--split` is `all` (default), `train`, `validation`, or `test`;
`--prediction-set` labels the stored result set. A named split is allowed only
when the input is exactly the dataset revision recorded by the run. A sealed
run may not be used as its own output location.

The artifact's `[inference]` section controls extraction: `batch_size` is a
positive integer or `auto`; `minimum_batch_size`, `maximum_batch_size`, and
`memory_budget_mb` (at least 64) bound auto-probing; `progress` enables console
status. Auto finds a verified batch size on the actual inference host, so it is
the safer default when hardware differs from training. `output.prediction_commit_batches`
can reduce SQLite transaction overhead for large output sets.

Example: preserve a traceable result set while allowing safe host-specific
batch selection:

```bash
python model_inference.py --run runs/fish-v3 --input data/new.sqlite \
  --output predictions/fish-v3.sqlite --split all --prediction-set survey-2026-09
```

## 3. Oracle Serve

Oracle Serve is the HTTP process for resident inference bundles and optional
local compute work. It does not persist operational inputs/outputs; callers
own them. Start it with `uv run oracle-serve`.

| CLI option | Default | Use and consideration |
|---|---:|---|
| `--model ALIAS=RUN_DIR` | repeatable | Register selected sealed artifact(s) under a stable, path-safe operational selector. |
| `--models-root DIRECTORY` | repeatable | Recursively discover sealed model runs/products; malformed, active, and duplicate artifacts are skipped and reported. |
| `--host`, `--port` | `127.0.0.1`, `8100` | Bind address/port. Do not expose a non-local host without authentication/TLS controls. |
| `--root-path` | `ORACLE_BUILDER_ROOT_PATH` or empty | External prefix behind a reverse proxy, e.g. `/oracle-builder-api`; keeps OpenAPI URLs correct. |
| `--no-preload` | false | Lazy-load models. Useful during development; production should preload so readiness reflects actual availability. |
| `--max-batch-size` | 256 | Maximum combined items for one model execution. Startup warmup can lower a model's active limit if memory cannot fit it. |
| `--max-wait-ms` | 8 | Queueing window used to form a micro-batch. Larger improves throughput but adds latency. |
| `--queue-capacity` | 1024 | Pending requests per model. Bound it to prevent memory exhaustion and provide backpressure. |
| `--no-compute` | false | Inference-only mode; requires at least one registered model. |
| `--compute-queue-size` | 128 | Maximum queued compute jobs. |
| `--worker-id` | `local` or `ORACLE_BUILDER_WORKER_ID` | Stable identifier reported to an orchestrator. |

For ASGI/environment deployment, `ORACLE_BUILDER_MODELS_ROOT` accepts one or
more roots (platform path separator), while `ORACLE_BUILDER_MODELS` accepts
comma-separated `alias=/path` registrations. `ORACLE_BUILDER_PRELOAD`,
`ORACLE_BUILDER_MAX_PAYLOAD_BYTES`, `ORACLE_BUILDER_SERVING_MAX_BATCH_SIZE`,
`ORACLE_BUILDER_SERVING_MAX_WAIT_MS`, `ORACLE_BUILDER_SERVING_QUEUE_CAPACITY`,
`ORACLE_BUILDER_COMPUTE_ENABLED`, `ORACLE_BUILDER_COMPUTE_QUEUE_SIZE`, and
`ORACLE_BUILDER_WORKER_ID` supply the corresponding application settings.
`ORACLE_BUILDER_API_TOKEN` requires `Authorization: Bearer TOKEN` on model and
compute routes.

The service exposes `GET /health/live`, `GET /health/ready`, `GET /v1/models`
(optional `?task=segmentation`), `GET /v1/models/{selector}/evidence/{item_id}`,
and `POST /v1/models/{selector}:predict`. Inference request/response bodies
use `application/vnd.oracle-builder.inference+npz`: a JSON manifest plus typed
arrays, decoded without pickle. The resident worker batches concurrent requests
while preserving each caller's correlation and result set; catalog diagnostics
report whether a model is GPU-accelerated and its micro-batch limits.

With compute enabled, the protected `/compute/*` API provides workers, status,
submission, job inspection/events, and cooperative cancellation. Permitted
actions are `train`, `evaluate`, `model_ingest`, `run_validate`, and `run_pack`;
requests contain immutable parameters and a retained `resources` scheduling
intent. It is deliberately not an arbitrary shell-command service.
