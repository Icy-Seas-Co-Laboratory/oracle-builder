# Composable model architecture V2

V2 separates a CNN encoder from spatial pooling, image representation,
metadata representation, fusion, and the task head. It is the default for new
runs in Oracle Builder 0.2.0; use the declaration below to make that choice
explicit in a recipe:

## Recipe policy and preset catalog

Every mutable model-training TOML under `configs/` is an explicit V2 recipe:
it declares `[architecture]` with `version = 2`. Classification and embedding
recipes also identify their encoder through `[encoder]`. Do not rely on the
runtime default when authoring or copying a recipe. This keeps the source TOML,
resolved run configuration, and rendered model construction view in agreement.

The maintained preset catalog lives in
[`configs/classification_defaults/`](../configs/classification_defaults/).
There is one canonical family recipe and one distinct V2 preset for every
selectable architecture/variant.  Family recipes provide a sensible starting
point; a variant recipe is the reproducible choice when an experiment depends
on a particular capacity.  The currently supported encoder catalog is:

| Encoder family | Canonical variants |
|---|---|
| Simple CNN | `simple_cnn` |
| ResNet | `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152` |
| DenseNet | `densenet121`, `densenet169`, `densenet201` |
| EfficientNet | `efficientnet_b0` through `efficientnet_b7` |
| EfficientNetV2 | `efficientnet_v2_b0`, `efficientnet_v2_b1`, `efficientnet_v2_b2`, `efficientnet_v2_b3`, `efficientnet_v2_s`, `efficientnet_v2_m`, `efficientnet_v2_l` |
| ConvNeXt | `convnext_tiny`, `convnext_small` |
| MobileNetV3 | `mobilenet_v3_small`, `mobilenet_v3_large` |
| Compatibility lightweight encoders | `resnet_like`, `densenet_like` |

The sole checked-in exception is the legacy clustering reference, which is not
a mutable training architecture.  Archived artifacts remain reproducible from
their resolved configuration; they are not silently rewritten as V2.

```toml
[architecture]
version = 2

[input]
channels = ["intensity", "gradient_magnitude", "foreground_distance"]

[input.geometry]
type = "preserve_aspect_pad"

[encoder]
family = "resnet"
variant = "resnet18"

[stem]
kernel_size = 3
stride = 1
pool = "none"

[normalization]
type = "group"
groups = "auto"

[pooling]
type = "gem"
p = 3.0

[image_embedding.projection]
type = "linear"
output_dim = 256
l2_normalize = false

[[metadata.fields]]
name = "log_area"
source = "metadata.area"
transform = "log1p"

[metadata.encoder]
type = "mlp"
hidden_units = [64]
output_dim = 32

[metadata.augmentation]
# Applied only to scaled continuous metadata in the training dataset.
# It is never applied to validation, test, inference, exports, or caches.
gaussian_variance = 0.01
probability = 1.0
fields = ["log_area"]

[fusion]
type = "projected"
output_dim = 256

[classifier]
type = "cosine"
cosine_scale = 16.0
```

## Encoders and classifier heads

`efficientnet_v2`, `efficientnet_v2_b0`, `efficientnet_v2_b1`,
`efficientnet_v2_b2`, `efficientnet_v2_b3`, `efficientnet_v2_s`,
`efficientnet_v2_m`, and `efficientnet_v2_l` provide native fused-MBConv
EfficientNetV2 encoders. They accept the same arbitrary intensity/derived
channel inputs as the other native encoders; no RGB conversion or pretrained
ImageNet weights are assumed. Select one with `run.model` and optionally tune
`model.width_coefficient`, `model.depth_coefficient`, `model.stem_filters`,
and `model.top_filters`.

For classes with multimodal appearance, V2 can predict from learned class
prototypes instead of a linear weight per class:

```toml
[classifier]
type = "prototype"
prototypes_per_class = 2
prototype_metric = "cosine" # or "euclidean"
cosine_scale = 16.0
```

With multiple prototypes, each class receives the score of its closest
prototype. The layer is serializable with the model and its logits remain
compatible with the existing softmax, losses, exports, and feature artifacts.

## Geometry augmentation

The shared training augmentation policy now supports independent axis zoom,
random resized crops, and shape-safe right-angle rotations:

```toml
[augmentation]
zoom_x = 0.10                 # omit to inherit zoom
zoom_y = 0.20
random_resized_crop = 0.20    # retain 80--100% of each side, then resize
rotate_90 = true
```

`rotate_90` samples all four orientations for square data and 0/180 degrees
for rectangular data, so batched tensor dimensions never change. Segmentation
masks use nearest-neighbor geometry; mask-valued input channels and sample
weights follow the same spatial transform as their target.

Serving geometry is separate from stochastic augmentation. To center a small
ROI on the model canvas without upscaling it, use:

```toml
[input.geometry]
type = "center_roi_pad"

[preprocessing]
pad_mode = "constant"
pad_value = 0.0
```

The V2 geometry declaration resolves to `preprocessing.resize_mode =
"center_pad"`; all geometry settings are retained in the resolved run recipe.

The training model still returns probabilities for Keras compatibility. Its
feature view and SavedModel `embed` signature expose these named tensors when
applicable:

- `feature_map`
- `image_embedding`
- `metadata_embedding`
- `fused_embedding`
- `projection_embedding`
- `logits`
- `probabilities`

ConvNeXt Tiny/Small and MobileNetV3 Small/Large are native V2 encoder families.
For compatibility, the resolver still recognizes `run.model` and
`model.variant`; a V2 `[encoder]` declaration is normalized to those builder
keys in the sealed resolved configuration.  New TOMLs should declare both
`[architecture]` and `[encoder]` rather than depending on that bridge.

## Component conventions

V2 treats the following tables as one assembled graph.  Omit optional sections
only when the defaults are intentional; a checked-in recipe should spell out
choices that materially affect a comparison.

| Component | Primary choices |
|---|---|
| `[input]` | ordered `channels`; `geometry.type` such as `preserve_aspect_pad`, `center_roi_pad`, `resize_crop`, `center_crop`, or `direct_resize` |
| `[encoder]` | `family` and `variant`; architecture-specific capacity knobs remain under `[model]` |
| `[stem]` | `kernel_size`, `stride`, `width`, and `pool` where the family supports them |
| `[normalization]` | `batch`, `group`, `layer`, or `none` |
| `[pooling]` | `avg`, `max`, `avg_max`, or `gem` (`p` and optional `trainable_p`) |
| `[image_embedding.projection]` | `identity`, `linear`, or `mlp`; `output_dim`, optional `hidden_units`, `dropout`, and `l2_normalize` |
| `[metadata]` | declared fields, train-split scaling, optional encoder and training-only Gaussian noise |
| `[fusion]` | `concat` or `projected` (with `output_dim`) |
| `[classifier]` | `linear`, `mlp`, `cosine`, or `prototype` |

When a V2 recipe uses an input channel list, it is the authoritative source for
the preprocessing channel contract.  For a V2 geometry declaration, the
resolver records the equivalent serving `preprocessing.resize_mode` too.  These
normalizations are visible in `config/resolved.json` and should be reviewed
with `--dry-run` before a long training run.

## Derived channel ordering

`preprocessing.channels` accepts `intensity`, `gradient_magnitude`, `gx`,
`gy`, `gaussian_blur`, `laplacian`, `local_contrast`, `foreground_mask`, and
`foreground_distance`. For stochastic training, set
`preprocessing.derive_after_augmentation = true`: augmentation is then applied
to intensity before derived channels are regenerated.

## Representation artifacts

Classification runs write bounded-memory representation caches by default:

```toml
[output.intermediate_artifacts]
enabled = true
split = "test"
batch_size = 256
representations = ["image_embedding", "fused_embedding"]
```

Each cache directory contains `embeddings.npy` (readable with NumPy mmap),
`records.jsonl`, and `manifest.json`. The manifest is versioned as
`oracle_builder.embedding_cache` v1 and records dataset, run, artifact, model,
component configuration, representation name, shape, and dtype.
