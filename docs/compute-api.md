# Oracle Builder compute API

`oracle-serve` exposes a compute namespace alongside its inference API.  It is
an execution service for an orchestrator; it does not catalog artifacts, name
paths, construct experiments, or choose which work is scientifically useful.

The calling orchestrator supplies a UUID that it already owns.  It records the
durable job and experiment state, while `oracle-serve` retains transient local
queue, process, worker, and log-event state.

## Start a local compute worker

The compute worker is enabled by default.  Unlike inference-only operation,
it does not require a registered inference bundle:

```bash
oracle-serve --host 127.0.0.1 --port 8100
```

Use `--no-compute` for inference-only operation. `--compute-queue-size` and
`--worker-id` set the local queue bound and stable worker identifier.

The worker uses one process slot by default. To run compatible jobs in
parallel, configure both the process slots and, when useful, a CPU admission
limit:

```bash
oracle-serve --compute-worker-slots 2 --compute-cpu-capacity 8
```

The equivalent environment settings are `ORACLE_COMPUTE_WORKER_SLOTS` and
`ORACLE_COMPUTE_CPU_CAPACITY`; the legacy `ORACLE_BUILDER_COMPUTE_*` names are
also accepted. The stack launcher forwards the unprefixed settings
automatically. A slot does not bypass resource safety: each job's sealed CPU
and GPU request must still fit, and a leased GPU cannot be assigned twice.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /compute/workers` | Per-slot worker capabilities, active job, and status. |
| `GET /compute/status` | Queue capacity, job counts, per-slot state, configured CPU/slot capacity, CPU use, and GPU leases. |
| `POST /compute/jobs` | Submit an immutable orchestrator-issued job. |
| `GET /compute/jobs/{job_id}` | Inspect current execution state. |
| `GET /compute/jobs/{job_id}/events?after=N` | Poll structured log/lifecycle events. |
| `POST /compute/jobs/{job_id}/cancel` | Cooperatively cancel queued/running work. |

The same bearer token used by the inference API protects every compute route.

## Job request

```json
{
  "job_id": "c2516bd6-95c2-4f1c-aa8a-d51b35868d4e",
  "action": "train",
  "parameters": {
    "config": "/oracle/configs/experiment-a/run-01.toml",
    "input": "/oracle/datasets/7c8f/dataset.sqlite",
    "output": "run-01",
    "runs_dir": "/oracle/runs"
  },
  "resources": {"gpu_count": 1, "priority": "normal"}
}
```

Supported actions are `train`, `evaluate`, `model_ingest`, `run_validate`, and
`run_pack`.  They map only to existing Oracle Builder commands; callers cannot
submit arbitrary shell commands.  The `resources` object is retained and
returned with the job so the orchestrator can state scheduling intent.  The
local scheduler performs bounded multi-slot resource placement.

`GET /compute/status` is the scheduler-facing capacity snapshot. Its
`resources` object includes `worker_slots`, `cpu_capacity`, `cpu_in_use`, and
`gpu_leases`; its `workers` list identifies each occupied or idle process slot.

For a resumed training job, send `action: "train"` with `parameters.resume`.
For a new training job, send `config`, `input`, and `output`; add `runs_dir` or
`overwrite` when needed.

Terminal job responses include `started_at`, `finished_at`, `worker_id`,
`error`, and a structured `result` containing `output_path` and `exit_code`.
The output path is resolved by `oracle-serve` from the immutable request; it
does not choose or rename that path.

On compute success the orchestrator transitions model-producing work to
`validating`. It marks the durable job `indexed` only after the expected
Oracle Builder artifact type is complete, sealed, fingerprint-valid, and
linked into the catalog. Validation failure is retained as
`artifact_invalid` with a structured report. This preserves the separation
between compute execution and artifact/catalog ownership.
