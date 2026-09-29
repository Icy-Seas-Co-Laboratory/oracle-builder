# Oracle Builder documentation

## Workflow guides

| Goal | Guide |
|---|---|
| Operate the control plane from a terminal | [Oracle control CLI](oracle-cli.md) |
| Choose the supported GUI, CLI, or API entry point | [Using Oracle Builder](user-entrypoints.md) |
| Deploy registered pull workers | [Oracle worker](oracle-worker.md) |
| Deploy the single-host demo | [Non-container demo deployment](../deploy/demo/README.md) |
| Deploy the containerized single-host demo | [Docker demo deployment](../deploy/docker/README.md) |
| Use the control-plane protocol | [Orchestrator API](orchestrator-api.md) |
| Understand portable execution attempts | [WorkUnit V1](work-unit-v1.md) |
| Migrate historical artifacts or datasets | Offline migration readers in the source tree (not a supported runtime API) |

## Contract references

| Artifact | Authoritative reference |
|---|---|
| SQLite classification or mask-refinement dataset | [Dataset schema V1](dataset-schema-v1.md) |
| Imported/published model product and its TOML | [Model-product schema V1](model-product-schema-v1.md) |
| Training run, split protocol, integrity, and package | [Model-run artifact V1](run-artifact-v1.md) |
| Shared deployment/training artifact profiles | [Oracle Model Artifact Standard V2](model-artifact-standard-v2.md) |
| In-memory and persisted inference packet | [Inference contract V1](inference-contract-v1.md) |
| Orchestrator-to-worker execution attempt | [WorkUnit V1](work-unit-v1.md) |

## Configuration examples

The checked-in `configs/` files are runnable starting points. Classification
family defaults are in `configs/classification_defaults/`; segmentation examples
are `configs/example_segmentation_*.toml`; imported-model metadata begins with
`configs/example_model_product.toml`; self-supervised embedding training begins
with `configs/example_embedding.toml`.
