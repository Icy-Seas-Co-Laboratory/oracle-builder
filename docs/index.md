# Oracle Builder documentation

## Workflow guides

| Goal | Guide |
|---|---|
| Install and run a first model | [Getting started](getting-started.md) |
| Train an image classifier from labeled folders | [Classification workflow](classification-workflow.md) |
| Curate ROIs and refine masks | [Mask-refinement workflow](mask-refinement-workflow.md) |
| Ingest a pre-existing Keras model | [External model products](model-products.md) |
| Configure training, evaluation, or recovery | [Training, evaluation, and recovery](training-and-evaluation.md) |
| Diagnose installation or run problems | [Operations and troubleshooting](operations-and-troubleshooting.md) |

## Contract references

| Artifact | Authoritative reference |
|---|---|
| SQLite classification or mask-refinement dataset | [Dataset schema V1](dataset-schema-v1.md) |
| Imported/published model product and its TOML | [Model-product schema V1](model-product-schema-v1.md) |
| Training run, split protocol, integrity, and package | [Model-run artifact V1](run-artifact-v1.md) |
| In-memory and persisted inference packet | [Inference contract V1](inference-contract-v1.md) |

## Configuration examples

The checked-in `configs/` files are runnable starting points. Classification
family defaults are in `configs/classification_defaults/`; segmentation examples
are `configs/example_segmentation_*.toml`; imported-model metadata begins with
`configs/example_model_product.toml`.
