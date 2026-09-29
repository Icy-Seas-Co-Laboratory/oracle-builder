# Oracle Builder

Oracle Builder is a durable control plane for portable image-model work. The
Orchestrator owns plans, artifact storage, worker admission, leases, and
published outputs; stateless `oracle-worker` processes pull sealed WorkUnits.

For normal users, the Orchestrator is the single system entry point: use the
Web GUI, the `oracle` CLI, or the Orchestrator HTTP API. All three operate on
the same control-plane records; none communicates directly with a worker.
See [Using Oracle Builder](docs/user-entrypoints.md) for supported workflows
and the current inference boundary.

The durable dataset and annotation-workspace contracts are also available in
the dependency-light `oracle_data_contracts` Python package within this source
tree, so other applications can share the same SQLite schema and lifecycle APIs.

## Start here

```bash
uv sync
```

Run supported control-plane commands through uv: `uv run oracle --help`,
`uv run oracle-orchestrator --help`, and `uv run oracle-worker --help`.

For the development/test environment with the optional GUI, run
`uv sync --extra gui`, then run `uv run pytest`. The Orchestrator API and
Uvicorn are base dependencies. Choose a GPU extra for your platform separately;
do not install every GPU extra at once.
For an S3/MinIO-compatible immutable artifact replica, add `--extra storage-s3`
and configure `oracle-orchestrator --s3-bucket ...`; the normal folder tree
remains the primary, human-readable artifact store.

For a containerized single-host demo (Caddy, production-built Web GUI,
Orchestrator, and Orchestrator-managed Docker workers), see the
[Docker demo deployment guide](deploy/docker/README.md).

Choose the guide that matches what you want to do:

- [Oracle control CLI](docs/oracle-cli.md) — upload data and queue portable work.
- [Oracle worker](docs/oracle-worker.md) — deploy stateless pull workers.
- [Orchestrator API](docs/orchestrator-api.md) — control-plane protocol.
- [Using Oracle Builder](docs/user-entrypoints.md) — choose the supported GUI,
  CLI, or API entry point.
- [WorkUnit V1](docs/work-unit-v1.md) — portable execution contract.

## Reference

- [Documentation index](docs/index.md)
- [Dataset schema V1](docs/dataset-schema-v1.md)
- [Model-product schema V1](docs/model-product-schema-v1.md)
- [Model-run artifact V1](docs/run-artifact-v1.md)
- [Oracle Model Artifact Standard V2](docs/model-artifact-standard-v2.md)
- [Inference contract V1](docs/inference-contract-v1.md)
- [Architecture and extension boundaries](docs/architecture.md)

## Web GUI

The SvelteKit orchestration interface lives in [`webgui/`](webgui/). It talks
to the `oracle-orchestrator` service through a same-origin proxy; see its
[setup guide](webgui/README.md) and the [Orchestrator API](docs/orchestrator-api.md).

Start the local orchestration and web GUI stack together with:

```bash
scripts/start_oracle_stack.sh
```

The launcher detects Apple Silicon Metal and NVIDIA CUDA on Linux or WSL2,
then synchronizes only the matching GPU extra. Set `ORACLE_ACCELERATOR=cpu`,
`cuda`, or `metal` to override detection. Run it from WSL2—not a native Windows
shell—for NVIDIA-backed Windows systems; a working `nvidia-smi` inside WSL is
required. `CUDA_VISIBLE_DEVICES` is honored when already set.

It creates a local `.oracle-runtime/` directory for the central SQLite
database, artifact storage, GUI-upload staging, and logs. Workers are separate
processes that connect outward to the control plane. Set `ORACLE_RUNTIME_DIR`,
`ORACLE_HOST`, or the `ORACLE_*_PORT` variables to override local defaults.

The reference documents are authoritative for on-disk contracts. Workflow guides
intentionally repeat the commands and decisions needed for a single task.
