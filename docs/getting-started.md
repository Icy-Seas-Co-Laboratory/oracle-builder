# Getting started

[Documentation index](index.md) · [Repository overview](../README.md)

Oracle Builder's supported user boundary is the Orchestrator. The Web GUI,
`oracle` CLI, and HTTP API all create and inspect the same durable records;
workers connect outward and are never addressed directly.

## Install and launch the local stack

```bash
uv sync
scripts/start_oracle_stack.sh
```

The launcher selects the available CPU, CUDA, or Apple Metal extra. Override
auto-detection with `ORACLE_ACCELERATOR=cpu`, `cuda`, or `metal`. The local GUI
and Orchestrator use ports 5111 and 8110 by default. Open the GUI at
`http://127.0.0.1:5111/`; runtime state is kept under `.oracle-runtime/`.

For a containerized single-host deployment, use the [Docker demo guide](../deploy/docker/README.md).

## Choose a first workflow

Prepare a [frozen Oracle SQLite dataset](training-dataset-guide.md) and an
[admitted worker pool](oracle-worker.md) before queueing training. The GUI
Construction workspace creates model definitions; the CLI can list those
definitions and queue an immutable revision.

- In the GUI, upload a frozen SQLite dataset, create or import a model
  definition, verify the queued work, and start it only after verification
  succeeds. The Queue and Live Dashboard pages expose worker and artifact state.
  [Example dashboard view](assets/training-dashboard.jpg) (illustrative UI,
  not a deployment benchmark).
- From a terminal, configure the Orchestrator and use the supported [CLI](oracle-cli.md):

  ```bash
  export ORACLE_ORCHESTRATOR_URL=http://127.0.0.1:8110
  uv run oracle health
  uv run oracle dataset upload ./training.sqlite --ingest
  uv run oracle definition list
  ```

  Uploading a dataset only materializes it. To train, create or import a model
  definition, ensure an admitted worker pool is available (see [worker setup](oracle-worker.md)),
  then queue with `uv run oracle definition queue`, run
  `uv run oracle queue verify`, and start with `uv run oracle queue start`.
  Copy the dataset, definition, and pool IDs returned by the catalog into the
  commands below. Verification returns the queued run's state; start only after
  it passes:

  ```bash
  uv run oracle definition queue DEFINITION_ID --name first-classifier \
    --dataset DATASET_ID --worker-pool WORKER_POOL_ID \
    --gpu-count 0 --batch-size 16 --epochs 1
  uv run oracle queue verify QUEUED_RUN_ID
  uv run oracle queue list
  uv run oracle queue start QUEUED_RUN_ID --worker-pool WORKER_POOL_ID
  uv run oracle job list
  ```

  This example requests CPU execution; choose a pool and GPU count appropriate
  to your model and worker installation. For batch inference, use
  `uv run oracle inference submit` against a sealed model and frozen dataset.
- For programmatic control, use the [Orchestrator API](orchestrator-api.md).

## Developer-only local model example

The repository also contains a direct Python training adapter for controlled
development and contract tests. It is not the deployment boundary and should
not replace the Orchestrator workflow above:

```bash
uv run python -m oracle_builder.data.sqlite_dataset \
  --classification datasets/example_classification.sqlite
uv run oracle-dataset checkpoint \
  datasets/example_classification.sqlite \
  --output datasets/example_classification.training.sqlite
uv run python model_training.py \
  --config configs/example_classification.toml \
  --input datasets/example_classification.training.sqlite \
  --output example-classification-run \
  --overwrite
```

The checkpoint is a frozen dataset revision. For supported end-to-end training,
queue the frozen revision through the Orchestrator instead. Continue with the
[classification workflow](classification-workflow.md),
[mask-refinement workflow](mask-refinement-workflow.md), or
[external model products](model-products.md).
