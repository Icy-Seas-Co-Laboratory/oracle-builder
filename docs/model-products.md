# External model products

`oracle-model ingest` turns a supplied model into a sealed, portable Oracle
Builder **model product**. It accepts native Keras `.keras` files, legacy
Keras `.h5` / `.hdf5` files, and TensorFlow SavedModel directories.

Use [Model-product schema V1](model-product-schema-v1.md) as the authoritative
reference for the artifact layout, TOML fields, promotion contract, identities,
and optional dataset provenance.

```bash
oracle-model ingest \
  --model /path/to/external-model.keras \
  --info configs/example_model_product.toml \
  --output products/external-model-v1
```

For a TensorFlow SavedModel, point `--model` at the directory containing
`saved_model.pb`. SavedModel import currently supports single-input
classification signatures. Set `[promotion].activation` explicitly to
`"softmax"` or `"linear"`; for multi-output signatures also set
`output_name`. Oracle Builder preserves the original directory unchanged and
exports an inference adapter with named `logits` and `probabilities` outputs.
This path is useful when an older Keras serialization can no longer be loaded
by the local Keras version.

The optional `--dataset` adds immutable SQLite dataset provenance to the
artifact. It does not create a training split, because importing a model is
not a training operation.

```bash
oracle-model ingest \
  --model /path/to/external-model.keras \
  --info /path/to/model-product.toml \
  --dataset datasets/example.sqlite \
  --output products/external-model-v1
```

The TOML uses `[product]` for identity, description, version, license, and
tags; `[[labels]]` supplies optional ordered classification labels.
`[preprocessing]` documents input preparation. Set `embed_in_model = true`
and provide `raw_input_shape` only when the supplied model expects raw image
values: Oracle Builder then includes resize, supported grayscale/RGB conversion,
inversion, and 0–255-to-0–1 rescaling in the saved promoted model.

For `classification` and `segmentation` products, `[promotion]` is enabled by
default. It produces standard Keras outputs named `logits` and `probabilities`
(plus normalized `features` when it finds or is given a vector embedding). A
softmax/sigmoid probability output gains a stable derived logit representation;
an explicit `logits_layer` takes precedence. Specify `probabilities_layer`,
`logits_layer`, `embedding_layer`, or `activation` when automatic inspection
would be ambiguous. Use `--no-promote` to retain a generic product unchanged.

See `configs/example_model_product.toml` for the complete annotated template.

Each product preserves the original source model, records SHA-256 hashes, and
writes `model/inspection.json`. Keras products also write a normalized
`model/final.keras`, weights, and a text summary. SavedModel products instead
write `model/export_savedmodel`, which is reload-tested with one zero-valued
forward pass when its input contract permits it.

When promotion succeeds, `model/final.keras` is an Oracle Builder
named-output inference bundle and `model/imported.keras` preserves the
normalized imported network. If promotion is not possible, the artifact remains
portable and the inspection report states why; provide the missing TOML
information or use an explicit adapter.
