# Oracle Builder architecture

Oracle Builder is organized as a reusable Python package with thin executable
entry points. Domain code should not import repository-root scripts.

## Package boundaries

| Package | Responsibility |
|---|---|
| `oracle_builder.datasets` | Versioned dataset contract, lifecycle, repositories, validation, and transfer. |
| `oracle_builder.data` | Backend-neutral decoding, preprocessing, splitting, tiling, and TensorFlow input adapters. |
| `oracle_builder.models` | Built-in architecture definitions. |
| `oracle_builder.training` | Distribution, augmentation, losses, metrics, pretraining, and training orchestration. |
| `oracle_builder.evaluation` | Predictions, thresholds, evidence, and reports. |
| `oracle_builder.classification` | Classification feature and evidence semantics. |
| `oracle_builder.inference` | Storage-neutral inference bundles, contracts, connectors, and sinks. |
| `oracle_builder.masking` | Mask editing, validation, API loading, and mask-refinement workflows. |
| `oracle_builder.saving` | Portable model serialization and load tests. |

The top-level `model_training.py`, `model_inference.py`, `model_evaluate.py`, and
`mask_builder.py` files are application entry points. New reusable behavior
belongs in the package domains above.

## Dependency direction

```text
CLI / applications
        |
        v
training, evaluation, masking workflows
        |
        v
datasets, data adapters, models, saving
        |
        v
TensorFlow / Keras / SQLite
```

Dataset code must not depend on a model architecture. Model code must not query
SQLite directly. Storage-specific SQL belongs in repository or adapter modules.

## Portability boundary

The dataset contract and stable identifiers are logical APIs. SQLite is the V1
cold-storage adapter. A PostgreSQL implementation should satisfy the same
repository behavior and typed records while mapping payloads to `bytea` or
external object storage.

The model registry references `oracle_builder.models`, ensuring that installed
packages contain every built-in architecture. Saved runs retain resolved model
configuration, dataset identity and fingerprint, environment information, and
load-test results.

## Extension rules

- Add a dataset use case by defining typed records and tables; do not overload
  generic input/output fields.
- Add storage through a repository adapter; do not scatter dialect checks.
- Add an architecture under `oracle_builder.models` and register its builder.
- Add a command as a thin adapter over package-level workflow functions.
- Version serialized contracts independently from application releases.

## Inference boundary

The canonical deployed unit is an inference bundle rather than a bare Keras
file. The bundle owns deterministic preprocessing and postprocessing while
retaining a separately exportable neural-network core. Operational systems
retain ownership of streamed inputs; inference is in-memory by default and
persistence is selected through an explicit sink.

The stable V1 contracts are documented in
[`inference-contract-v1.md`](inference-contract-v1.md).
