# Oracle Builder configuration catalog

`reference_v2_exhaustive.toml` is the authoritative Architecture V2 authoring
reference. It is deliberately verbose: every maintained training setting is
either active with its default or shown immediately beside its active section
as a commented alternative. Start there when a run needs an option not exposed
by a focused preset.

The machine-readable counterpart is `oracle_builder.config_schema`. Its
versioned field catalog powers `/v1/config-schema`, including defaults, type,
finite choices, ranges, advanced flags, task applicability, and conditional
visibility. Do not create a second GUI-only option list. Runtime defaults are
projected into this catalog, while non-default V2 component fields and their
UI semantics are declared there. The API includes a fingerprint so a draft or
artifact can record which catalog interpreted it.

Each field also has an exposure tier. `standard` fields are routine protocol
choices, `advanced` fields are regularly useful deliberate tuning controls,
and `expert` fields remain hidden until an explicit expert toggle is enabled.
`internal` fields are dataset-derived, runtime-owned, or legacy-compatible and
are never normal authoring controls. New GUI code must use `exposure`; the
older boolean `advanced` remains only for API compatibility.

Run `python scripts/generate_classification_default_configs.py` after changing
the shared focused classification baseline or recipe catalog. It is idempotent
and regenerates only `classification_defaults/*.toml`; use `--check` in CI to
detect a stale checked-in preset.

Published training recipes use this order:

1. `run`, `architecture`, `input`, and `data`
2. preprocessing and input-channel derivation
3. encoder, stem, pooling, representation, metadata, fusion, and classifier
4. family-specific `model` settings and optional task-specific modules
5. self-supervised, supervised training, callbacks, recovery, and augmentation
6. execution, inference, evaluation, evidence, and output artifacts

`[data.materialization]` is an optional performance layer for frozen
classification datasets. It writes immutable NPY tensor shards plus a
Parquet-or-JSONL row manifest keyed by the dataset, split, preprocessing, and
storage signatures. Use `mode = "shared"` to reuse the cache across compatible
runs; leave it `off` when startup latency or cache disk use is more important.
Random augmentation intentionally remains online.

`classification_defaults/` contains focused V2 presets. Each names its exact
encoder family and variant, records an explicit native stem, and is intended
for routine runs; omitted advanced controls retain the documented defaults in
the exhaustive reference. The non-default architecture fields are kept close
to `[model]` so a DenseNet, ResNet, or EfficientNet can be reviewed without
searching through unrelated options.

The `example_*.toml` recipes demonstrate complete workflows. In particular,
`example_clustering.toml` now creates a V2 embedding artifact for downstream
clustering: `task = "clustering"` is not a supported training task.

Two TOMLs intentionally are not V2 training recipes: `example_model_product.toml`
is an external model-product descriptor, and `../datasets/metadata.example.toml`
is source-dataset metadata. Historical `runs/**/config/source.toml` files are
immutable provenance records and must never be edited to claim V2.
