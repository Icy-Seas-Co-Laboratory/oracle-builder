# Oracle Builder model-product schema V1

This reference defines the portable artifact created by `oracle-model ingest`.
It supplements the shared [model-run artifact V1](run-artifact-v1.md) layout.

## Identity and lifecycle

`artifact.json` has `artifact_type = "model_product"` and the shared artifact
schema `oracle_builder_model_run` version `1.0.0`. It contains UUID
`artifact_id` and `run_id`, status/lifecycle, producer, optional dataset
provenance, the model contract, checksummed inventory, and semantic artifact
fingerprint. A completed product is sealed.

`config/source.toml` is the original user-supplied model-information file;
`config/resolved.json` is its portable resolved representation. Runtime source
paths are stored separately under `provenance/runtime.json`.

## Required TOML fields

```toml
[product]
name = "my-model"
task = "classification" # generic, classification, segmentation
version = "1.0.0"
description = "Purpose, source, and known limitations."
license = ""
tags = ["keras"]
```

`name` defaults to the source filename and other product fields are optional,
but supplying them is strongly recommended. Unknown TOML fields are preserved
in the resolved product metadata.

## Optional TOML sections

```toml
[[labels]]
name = "copepod"
label_id = "copepod"
# class_index = 0 # defaults to array order

[preprocessing]
resize_mode = "fit_pad"
channel_mode = "grayscale"
rescale = true
invert = false
embed_in_model = false
# raw_input_shape = [512, 512, 1]

[promotion]
enabled = true
# probabilities_layer = "predictions"
# logits_layer = "logits"
# embedding_layer = "features"
# activation = "softmax"
# output_name = "output_0" # SavedModel only, required for multiple outputs

[outputs]
primary = "class_probabilities"
# threshold = 0.5
```

`[[labels]]` is ordered by array position unless `class_index` is supplied;
indices must be contiguous and, when known, the count must equal the model class
axis. `embed_in_model` is an explicit assertion that input values are raw images.
When enabled, `raw_input_shape` is required and the importer embeds supported
resize, 1↔3 channel conversion, inversion, and rescaling operations.

## Model directory contract

```text
model/
├── source/original.keras|h5|hdf5|_savedmodel/
├── imported.keras                 # Keras products only
├── final.keras                    # Keras products only
├── weights.weights.h5             # Keras products only
├── export_savedmodel/             # standard products and SavedModel imports
├── inspection.json
├── model_summary.txt
├── model_manifest.json
└── load_test_report.json
```

`source/original.*` is an unchanged copy. `imported.keras` is the normalized
Keras import. `final.keras` is the preferred product model and may be a promoted
wrapper. `weights.weights.h5` requires the matching final architecture. A
TensorFlow SavedModel import preserves `source/original_savedmodel/` and writes
an `export_savedmodel/` adapter instead; it does not require that a historical
Keras configuration remain deserializable.

`inspection.json` records source hash, tensor/layer inspection, promotion
decision and assumptions, and reload/forward-pass results. Every saved file is
included in the sealed inventory and checksum file.

## Promotion contract

For known tasks, automatic promotion is attempted unless disabled with
`--no-promote` or `promotion.enabled = false`.

| Task | `final.keras` outputs |
|---|---|
| Classification | `logits`, `probabilities`, and `features` when an embedding is available. |
| Segmentation | `logits` and `probabilities`. |
| Generic/unpromoted | Original outputs; an explicit adapter is required. |

Explicit layer names take precedence. Softmax and sigmoid probabilities without
an accessible logits layer receive a numerically stable derived logit. Softmax
logits are canonical only up to an additive constant; the derivation is recorded
in `inspection.json`. A promotion failure preserves the product rather than
guessing its semantics.

SavedModel import currently supports a single-input classification signature.
It requires an explicit `promotion.activation` (`softmax` or `linear`) and
wraps the selected output as named `logits` and `probabilities`. It does not
invent an embedding when the source signature does not expose one.

## Dataset provenance

`--dataset DATASET.sqlite` is optional. When supplied, the manifest records the
dataset/revision UUIDs, type, schema/version, lifecycle, and semantic SHA-256.
No split manifest is inferred from an imported model: the artifact stores an
explicit unavailable split protocol instead.
