# V2 classification preset catalog

Every TOML in this directory declares `[architecture] version = 2` and the
same explicit input, encoder, pooling, image-embedding, metadata, fusion, and
classifier contract. Dataset-specific facts—classes, labels, splits, and
metadata values—are read from the selected Oracle SQLite training set.

| Family | Family default | Concrete variant recipes |
| --- | --- | --- |
| Compact | `simple_cnn.toml` | `simple_cnn.toml` |
| Shallow residual | `resnet_like.toml` | `resnet_like.toml` |
| Shallow dense | `densenet_like.toml` | `densenet_like.toml` |
| ResNet | `resnet.toml` | `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152` |
| DenseNet | `densenet.toml` | `densenet121`, `densenet169`, `densenet201` |
| EfficientNet | `efficientnet.toml` | `efficientnet_b0` through `efficientnet_b7` |
| EfficientNetV2 | `efficientnet_v2.toml` (B0) | `efficientnet_v2_b0` through `efficientnet_v2_b3`, `efficientnet_v2_s`, `efficientnet_v2_m`, `efficientnet_v2_l` |
| ConvNeXt | `convnext.toml` (Tiny) | `convnext_tiny`, `convnext_small` |
| MobileNetV3 | `mobilenet.toml` (Small) | `mobilenet_v3_small`, `mobilenet_v3_large` |

The family aliases above document their concrete default explicitly; use a
concrete variant file for a durable, unambiguous run record. The V2 metadata
noise setting is intentionally zero by default. Raise
`metadata.augmentation.gaussian_variance` only for declared continuous
metadata fields and only when training-time regularization is desired.

All recipes are dataset-agnostic: the training command supplies the SQLite
dataset and records its inferred class order, split membership, and source
identity in the run artifact. They share grayscale `fit_pad` preprocessing,
effective-number class weighting, and a conservative geometry-safe
augmentation policy so family comparisons remain controlled. DenseNet and the
largest compound-scaled variants may need a smaller batch size because their
activation memory is higher.

`effective_number` class weighting is the standard imbalance policy; choose
`inverse_frequency` for stronger balancing, `explicit` only for a reviewed
hand-selected protocol, or `sparse_categorical_crossentropy` to opt out. The
family defaults also retain their disabled-by-default resolution-stratification
recipes. Enable them only after checking each configured stratum has adequate
class and split support.
