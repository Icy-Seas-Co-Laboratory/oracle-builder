# Oracle control CLI

`oracle` is the thin command-line client for the Oracle Orchestrator. It is
the supported interactive interface for creating control-plane records,
queueing work for worker pools, and observing jobs. It does not run training or inference,
choose output directories, or write derived artifacts locally.

It is the supported command-line system boundary: scripts must call the
Orchestrator through `oracle` or its HTTP API, never an `oracle-worker`. The
supported execution actions are `train` and durable batch `infer`. See [Using
Oracle Builder](user-entrypoints.md).

```bash
export ORACLE_ORCHESTRATOR_URL=http://control-plane.example:8110
export ORACLE_ORCHESTRATOR_TOKEN=OPERATOR_TOKEN
oracle health
oracle dataset upload ./training.sqlite --ingest
oracle definition list
oracle definition queue DEFINITION_ID --name baseline-compare --dataset DATASET_ID \
  --worker-pool GPU_POOL_ID --gpu-count 1 --batch-size 32 --epochs 12
oracle queue verify QUEUED_RUN_ID
oracle queue list
oracle queue start QUEUED_RUN_ID --worker-pool GPU_POOL_ID
oracle job list
oracle inference submit --name baseline-batch --model MODEL_ARTIFACT_ID --dataset DATASET_ID \
  --worker-pool CPU_POOL_ID --split test --start --wait
```

The local-I/O commands, `oracle dataset upload` and `oracle recipe upload`,
stream named source files to the control plane's owned upload staging. Their
respective `--ingest` and `--create` options then use only the server-returned
staging reference. `dataset ingest` and `recipe create --config-path` remain
compatibility commands for already-approved server-visible paths.

Useful read/control groups are `dataset`, `definition`, `spec`, `job`,
`queue`, `artifact`, `worker`, and `inference`. `oracle definition queue` pins a
definition and frozen dataset and adds an unverified entry without starting work.
`oracle queue verify ID [ID ...]` runs preflight on workers and leaves passing
runs ready for a separate `oracle queue start` command. Add `--start` to the
verify command only when you want each successful run to start automatically.
Failed verification never starts training. Use `--auto-batch-size` on definition
queueing to calibrate during verification, optionally with `--maximum-batch-size`.
Training keeps the verified batch size and worker; reverify before starting to
choose another worker. `oracle
worker` exposes worker-pool, worker, and durable-lease state; it deliberately
has no arbitrary remote command or worker-registration-token flag. `oracle
spec enqueue` is retained only for already-portable `train` specifications.
`oracle job list` reads durable pull-job state directly; the old remote
`--refresh` reconciliation flag has been removed.

`oracle inference submit` creates an immutable batch-inference request from a
sealed model artifact and frozen dataset. `--start` authorizes it for its
selected pool; `--wait` waits for a terminal durable state. Use `oracle
inference list`, `show`, `cancel`, and `download` to inspect and retrieve the
published result. The worker remains an internal pull client.

The control-plane CLI is the supported operational interface. Historical data
may be imported through one-way migration tooling, but no legacy execution or
compute-endpoint path is supported.

Read-only commands work anonymously against the default API policy. Commands
that create, upload, queue, cancel, or otherwise change state require an
`operator` or `admin` token supplied by `--token` or
`ORACLE_ORCHESTRATOR_TOKEN`.

When an S3-compatible replica is configured, use `oracle artifact replicas` to
inspect durable replica status. `oracle artifact replica-retry REF`,
`replica-verify REF`, and `replica-restore REF` enqueue safe maintenance using
an ArtifactRef such as `model_run:RUN_ID@ATTEMPT`; restore refuses to overwrite
an existing local artifact.
# Worker control

`oracle job command JOB_ID pause`, `yield`, `stop_now`, `restart`, or `resume`
submits an idempotent durable directive; `oracle job commands JOB_ID` shows its
received, accepted, applied, failed, or expired receipt. `restart` discards
uncommitted work for the current unit, while `resume` continues from the last
committed checkpoint. `oracle execution show RUN_ID` displays the resumable run
chain. A request is not completion: wait for its worker acknowledgement.
