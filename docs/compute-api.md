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

Use `--no-compute` for inference-only operation.  `--compute-queue-size` and
`--worker-id` set the local queue bound and stable worker identifier.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /compute/workers` | Worker capabilities, active job, and status. |
| `GET /compute/status` | Queue capacity, job counts, and worker readiness. |
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
single local worker currently advertises capabilities but does not yet perform
multi-worker resource placement.

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
