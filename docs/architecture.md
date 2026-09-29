# Oracle Builder architecture

Oracle Builder is organized as a reusable Python package with thin executable
entry points. Domain code should not import repository-root scripts.

## Package boundaries

| Package | Responsibility |
|---|---|
| `oracle_data_contracts` | Dependency-light Dataset V1 schema/lifecycle/workspace plus model-run artifact layout, manifests, split protocol, fingerprints, repositories, and deterministic transfer. |
| `oracle_data_contracts.work_units` | Versioned, path-free execution attempt contracts shared by the orchestrator and workers. |
| `oracle_tools.roi` | Dependency-light ROI image, mask morphology, thresholding, and validation primitives. It has no Builder, orchestration, training, SQLite-workspace, or GUI dependency. |
| `oracle_builder.datasets`, `oracle_builder.artifacts` | Backward-compatible imports and Oracle Builder CLI integration for the shared contracts. |
| `oracle_builder.data` | Backend-neutral decoding, preprocessing, splitting, tiling, and TensorFlow input adapters. |
| `oracle_builder.models` | Built-in architecture definitions. |
| `oracle_builder.training` | Distribution, augmentation, losses, metrics, self-supervised training, and training orchestration. |
| `oracle_builder.evaluation` | Predictions, thresholds, evidence, and reports. |
| `oracle_builder.classification` | Classification feature and evidence semantics. |
| `oracle_builder.inference` | Storage-neutral inference bundles, contracts, connectors, and sinks. |
| `oracle_builder.worker` | Stateless registered pull-worker runtime with fixed `train`, `infer`, and `package` executors plus disk-backed, resumable output publication. |
| `oracle_builder.orchestration` | Durable catalog, worker-pool admission, lease, artifact-store, and staged-publication authority. |
| `oracle_builder.masking` | Mask editing, API loading, SQLite workspace, and mask-refinement workflows. |
| `oracle_builder.saving` | Portable model serialization and load tests. |

The top-level `model_training.py`, `model_inference.py`, `model_evaluate.py`, and
`mask_builder.py` files are local/developer application adapters. Normal users
enter the system through the Orchestrator via the Web GUI, `oracle` CLI, or its
HTTP API; new reusable behavior belongs in the package domains above.

## Dependency direction

```text
CLI / applications
        |
        v
training, evaluation, masking workflows
        |
        v
data adapters, models, saving
        |
        v
oracle_data_contracts
        |
        v
SQLite / standard-library dependencies
```

Dataset code must not depend on a model architecture. Model code must not query
SQLite directly. Storage-specific SQL belongs in repository or adapter modules.

## Portability boundary

The dataset contract and stable identifiers live in `oracle_data_contracts`,
which intentionally has no TensorFlow, Keras, napari, Pelagia-client, or model
architecture dependency. SQLite is its V1 cold-storage adapter. A PostgreSQL
implementation should satisfy the same repository behavior and typed records
while mapping payloads to `bytea` or external object storage.

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

Batch inference is an Orchestrator-owned WorkUnit: callers create and start an
inference run through the Orchestrator, and an admitted worker executes the
fixed `infer` action. There is no supported direct-worker or `oracle-serve`
inference endpoint.

## Control-plane migration boundary

The Orchestrator owns durable work-unit construction and is the designated
artifact-store publication authority. `WorkUnit V1` contains portable artifact
references and a job-scoped staging reference; it never serializes host paths
or storage credentials. Workers materialize granted artifacts into disposable
scratch space and publish only staged output through a disk-backed resumable
transfer session. The Orchestrator alone validates, seals, and promotes the
result. This same worker protocol works over loopback or HTTPS to a remote
worker; S3-compatible stores remain an Orchestrator-owned replica/persistence
boundary rather than a credentialed worker write path. See
[`work-unit-v1.md`](work-unit-v1.md).
