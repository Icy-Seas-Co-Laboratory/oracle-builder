# Oracle Builder

Oracle Builder builds, curates, trains, evaluates, preserves, and deploys
portable image-model products using local SQLite datasets and TensorFlow/Keras.
It supports image classification and ROI mask refinement.

The durable dataset and annotation-workspace contracts are also available in
the dependency-light `oracle_data_contracts` Python package within this source
tree, so other applications can share the same SQLite schema and lifecycle APIs.

## Start here

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install -e .
```

Choose the guide that matches what you want to do:

- [Getting started](docs/getting-started.md) — install, verify hardware, and run a small training job.
- [Classification workflow](docs/classification-workflow.md) — import a folder library, train, evaluate, infer, and compare classifier families.
- [Mask-refinement workflow](docs/mask-refinement-workflow.md) — build/edit an ROI mask dataset, train U-Net-family models, tile large ROIs, and visualize results.
- [External model products](docs/model-products.md) — ingest and promote existing Keras models.
- [Training, evaluation, and recovery](docs/training-and-evaluation.md) — configs, augmentation, streaming, multi-GPU, metrics, and resume behavior.
- [Operations and troubleshooting](docs/operations-and-troubleshooting.md) — validation, packages, common problems, and analysis helpers.
- [Inference API](docs/inference-api.md) — serve resident model bundles to Pelagia and other operational callers.

## Reference

- [Documentation index](docs/index.md)
- [Dataset schema V1](docs/dataset-schema-v1.md)
- [Model-product schema V1](docs/model-product-schema-v1.md)
- [Model-run artifact V1](docs/run-artifact-v1.md)
- [Inference contract V1](docs/inference-contract-v1.md)
- [Architecture and extension boundaries](docs/architecture.md)

The reference documents are authoritative for on-disk contracts. Workflow guides
intentionally repeat the commands and decisions needed for a single task.
