# Oracle Builder Orchestrator API

The Orchestrator is the Pelagia-facing control plane for Oracle Builder.  It
owns one SQLite database containing catalog summaries, experiments, immutable
run specifications, and orchestration job state.  It does not replace Oracle
Builder artifact contracts: datasets and artifacts remain the scientific source
of truth and may be re-ingested after recovery.

It is the sole supported system API boundary. The Web GUI, `oracle` CLI, and
programmatic clients communicate with this service—not with `oracle-worker`.
Workers are lease-authenticated execution clients and expose no user-facing
control or inference API. See [Using Oracle Builder](user-entrypoints.md) for
the supported training and batch-inference workflows.

For normal interactive operation, prefer the thin [`oracle` control
CLI](oracle-cli.md). It maps common dataset, definition, queue, artifact, and
worker-pool actions to these API endpoints without taking execution or output
path ownership.

```bash
oracle-orchestrator --database /oracle/control/orchestrator.sqlite \
  --workspace-root /oracle/workspace \
  --artifact-root /oracle/runtime-artifacts \
  --runs-root /oracle/workspace/runs \
  --datasets-root /oracle/workspace/datasets \
  --training-catalog-root /oracle/training-sources \
  --role-tokens-sha256 '{"operator":"sha256:OPERATOR_TOKEN_SHA256"}' --port 8110
```

### Artifact storage

The default artifact store is an ordinary local folder tree.  That is the
authoritative workflow surface: sealed datasets and runs remain easy to
inspect, copy, snapshot, and restore without a proprietary database.  For
off-host durability, install `oracle-builder[storage-s3]` and add
`--s3-bucket BUCKET` (plus `--s3-endpoint-url` for MinIO or another compatible
service).  Each locally published artifact is replicated under
`{prefix}/{kind}/{artifact-id}/{revision}`; a content-hash manifest marker is
written last, so a backup process can ignore incomplete uploads.

Replica progress is durable in the control-plane database (`pending`,
`replicated`, or `failed`). Operators can inspect it with `GET
/v1/artifact-replicas` and enqueue retries, verification, or a restore of a
missing *canonical* local artifact through the corresponding
`:retry`, `:verify`, and `:restore` endpoints. Those operations accept only
location-free `ArtifactRef` values, run through the operation runner, and
never overwrite an existing local folder. A restore validates every downloaded
object against the remote completion marker before its atomic local publish.

The local folder is intentionally retained as the online authority in this
release.  It keeps atomic publication, catalog scanning, and grant delivery
simple, while S3-compatible storage provides a reliable immutable replica.

## Core workflow

1. Register a frozen SQLite dataset with `POST /v1/datasets:ingest`.
2. Register a validated TOML training recipe with `POST /v1/recipes`.
3. Create or update a typed model definition at `POST /v1/model-definitions`.
4. Queue it to an admitted worker pool with
   `POST /v1/model-definitions/{id}:queue`, supplying a frozen dataset and
   `worker_pool_id`. It returns `pending_verification` and creates no worker job.
5. Verify selected entries through `POST /v1/queued-runs:verify` with
   `queued_run_ids` and optional `start_after_verification` (default `false`).
   The worker resolves configuration/data, probes model execution, and determines
   the automatic batch size when requested. This is a verification-only job.
6. After successful verification, start `ready` entries through
   `POST /v1/queued-runs:start` with `queued_run_ids` and `worker_pool_id`.
   Training uses the verified batch size on the same worker execution instance. With explicit
   `start_after_verification: true`, successful verification creates the training
   job atomically; failed verification never starts training.
7. Create batch inference with `POST /v1/inference-runs`, using a sealed model
   artifact, frozen dataset, split, and admitted pool; authorize it with
   `POST /v1/inference-runs/{id}:start` and retrieve its sealed result through
   `GET /v1/inference-runs/{id}:download`.

There is no push-dispatch or compute-endpoint control plane. The Orchestrator
is the durable scheduling authority; workers are pull-only clients.

## Access policy

`GET`, `HEAD`, and `OPTIONS` requests are intentionally anonymous and
rate-limited per source address, so simple scripts can retrieve public catalog,
job, and artifact metadata without managing a login. The deployment proxy is
responsible for distributed rate limiting and client-address forwarding.

All writes require a configured bearer role token. `analyst` may submit the
explicit analytical queries; `operator` may queue, cancel, upload, and change
catalog state; `admin` is reserved for operational administration. Configure
only SHA-256 token digests through `--role-tokens-sha256` or
`ORACLE_ORCHESTRATOR_ROLE_TOKENS_SHA256`; the raw token is supplied by CLI
clients as `ORACLE_ORCHESTRATOR_TOKEN`. Worker registration and lease APIs use
separate pool and worker credentials. The local stack permits unauthenticated
mutations only when bound to loopback, and does so explicitly.

## Registered pull workers

An operator creates a worker pool with its allowed WorkUnit actions, receiving
a pool join token once.
A worker registers with that token, then receives its own bearer credential
once. It polls `POST /v1/workers/{worker_id}:lease`; an idle poll returns
`204`, while a successful poll returns an opaque lease token and a sealed,
path-free WorkUnit. Renewal and release are scoped to that lease token.

The durable pool/worker/lease records are inspectable through
`/v1/worker-pools`, `/v1/workers`, and `/v1/worker-leases`. Tokens are never
returned by list/detail endpoints or saved in plaintext. A browser or the
`oracle` CLI queues a definition only through
`POST /v1/model-definitions/{id}:queue`; the Orchestrator pins portable input
artifacts before verification. The old `:validate-and-queue` URL is a compatibility
alias for queue intake and now also returns an unverified entry. The
lower-level `enqueue_work_unit_for_pool` boundary stays internal, so a browser
or worker cannot choose an output location, inject a command, or construct
work for another pool.

`oracle-worker` registers once and continuously acknowledges leases,
requests/downloads one-use input grants, emits idempotent progress events,
uploads a staging archive, and completes the lease. Downloads require the
worker bearer, lease token, and grant token. Output publication uses a
lease-bound resumable session: create it with `POST
/v1/worker-leases/{lease_id}/output-uploads`, send authenticated fixed parts
to `PUT /v1/worker-output-uploads/{upload_id}/parts/{part_number}` with an
exact `Content-Range` and part SHA-256, then call `POST
/v1/worker-output-uploads/{upload_id}:finalize`. The control plane persists
accepted part hashes, so a worker may resend a part or resume after a dropped
connection without repeating execution. It validates the assembled archive
SHA-256 and stages it before normal lease completion publishes the artifact.
No worker receives an artifact-store path or object-store credential. A
`train` completion is validated as a model-run artifact and indexed
automatically.

The same HTTPS transfer protocol is used for local and remote workers. When
`--s3-bucket` is configured, the Orchestrator retains ownership of the S3 or
S3-compatible endpoint: sealed artifacts replicate there after local atomic
publication, with a content manifest marker written last. This keeps worker
credentials and artifact-store mutation authority out of remote deployments
while still supporting MinIO and other S3-compatible endpoints.

## Responsive operations and live updates

Long-running control-plane work has a durable operation record rather than
holding a browser request open. Schedule catalog scans, training-catalog scans,
catalog reindexing, worker reconciliation, replica maintenance, or legacy queue
intake with the corresponding `.../schedule` endpoints. Training preflight and
batch calibration run as worker jobs created by `POST /v1/queued-runs:verify`,
not as control-plane operations. Each
returns `202 Accepted` and an operation object
whose `operation_id` is available from `GET /v1/operations/{operation_id}`;
`GET /v1/operations` lists recent work. Operations retain `queued`, `running`,
`completed`, and `failed` state, timestamps, result/error data, and ordered
events across an orchestrator restart.

`GET /v1/events?after=<cursor>` is a resumable server-sent-event stream over
that durable event ledger. Events carry a globally ordered cursor in their SSE
`id`; clients should reconnect using the last received ID and retain adaptive
polling as a rolling-upgrade fallback. The orchestrator's single bounded
operation runner periodically reconciles pull-worker liveness and staged
publication, without contacting a remote compute service.

Worker-pool admission and leases enforce declared action and resource
compatibility before a worker can materialize a WorkUnit. The supplied local
pull-worker provider can start and stop only fixed `oracle-worker` argument
vectors from a typed deployment declaration; it reads a bootstrap token from a
permission-checked secret file rather than a command line. Container, VM, and
Kubernetes providers should map that same declaration to native deployment
primitives. The Orchestrator never opens an SSH or arbitrary-command channel
to workers.

For model-producing jobs, durable status progresses through `queued`,
`leased`, `running`, `validating`, and finally `indexed`. Worker failures use
`failed`; valid process output that fails the artifact contract uses
`artifact_invalid`. `GET /v1/jobs` returns worker and timing fields, structured
validation/reporting state, and linked artifact summary. The historical
path-based model-import endpoint and remote-job refresh behavior return `410
Gone`; model products enter the system only as sealed artifacts.

The orchestrator owns output locations. For training it assigns
`{runs-root}/{specification-id}` through `runs_dir` and `output`; for
evaluations, imported products, upload staging, and packages it assigns
subdirectories beneath `{artifact-root}`. UI clients may supply inputs and
configuration references, but cannot choose arbitrary derived-artifact paths.

When no roots are supplied, `{runs-root}` and `{datasets-root}` default to
`{workspace-root}/runs` and `{workspace-root}/datasets`. On application startup
the service safely scans the runs root for sealed artifacts and its dataset
catalog roots for frozen Oracle SQLite revisions. Missing database records are
registered; existing records are left intact. The resulting reconciliation
summary is available from `GET /health/ready`.

Queue intake pins the definition revision, dataset and baseline configuration.
Verification produces a separate sealed WorkUnit with `queue_execution.phase`
set to `verify`. A successful report records the selected batch size, worker,
checks, and timestamp in `preflight_report`; the final training WorkUnit uses
`queue_execution.mode: verified`. Its worker-local configuration applies that
batch size without changing the baseline artifact or recalibrating at start.

Queue statuses are `pending_verification` → `verifying` → `ready` → `queued`.
Verification failure becomes `needs_attention` and may be retried. Reverification
of an unstarted ready entry can select a different compatible worker. Start
requests reject unverified rows and duplicate starts. Selection validation and
job creation commit atomically within each request.

Deploy the updated Orchestrator and restart training workers together. Workers
advertise `training_verification_v1` on registration and authenticated lease
polls, including when reusing saved credentials. A fresh worker boot identifier and
capability digest bind verification to its environment. Restarting or changing
the worker invalidates unstarted verified work and requires reverification.
Older workers cannot claim
verification or verified-training units. Existing unstarted legacy entries must
be verified before starting; already running jobs are not migrated.

## Model workspace and training-source catalog

`POST /v1/artifacts/catalog/query` provides paginated, server-side filtering
and sorting of indexed artifacts. It exposes denormalized display facts such as
training set, classifier, stem size, macro F1, loss, and training duration,
while tags remain UI-owned annotations outside sealed artifacts. `POST
/v1/artifact-tags/assign` applies tags to one or more selected models; `GET
/v1/artifacts/{id}/architecture-view` derives a stable module graph from the
resolved configuration.

Model construction uses versioned drafts. Create, revise, validate, and clone
them under `/v1/model-drafts`. `POST /v1/model-drafts/{id}:plan-training`
validates the draft plus run-only overrides, records any initialization source,
and writes an immutable TOML snapshot into the owned experiment directory.

`--datasets-root` defaults to the project's `datasets` directory and is always
included as a read-only source root. `--training-catalog-root` may be repeated
to expose additional source directories. `POST /v1/training-catalog:scan`
indexes only Oracle Builder SQLite dataset revisions; image folders are never
treated as training sets and require an explicit conversion step. Catalog
entries are grouped into training-set families and immutable revisions. Detail,
comparison, lifecycle, and bounded JPEG preview endpoints let an operator
assess a revision. Only frozen revisions can be selected for training; an
operator may freeze a working revision explicitly with `POST
/v1/training-catalog/{catalog_id}:freeze`.

## Results and comparisons

`GET /v1/experiments/{id}/results` joins the experiment, dataset, run
specifications, latest jobs, and indexed artifacts. Evaluation metrics are read
from the standard sealed artifact summary (with the standard evaluation summary
file as a fallback). The response identifies primary task metrics, runtime,
recipe, seed, worker, and the recorded evaluation protocol.

`POST /v1/comparisons` accepts a name, description, and artifact IDs. A
comparison is persisted only when at least two artifacts have standard metrics
and share the same task, dataset fingerprint, and evaluation split. The saved
record snapshots the artifact evidence and common protocol; it never recomputes
metrics or modifies artifacts. Use `GET /v1/comparisons` and
`GET /v1/comparisons/{id}` to retrieve saved comparisons.

### Detailed artifact evidence

`GET /v1/artifacts/{artifact_id}/evidence` returns a bounded, display-oriented
view of evidence already present in the sealed artifact. Classification results
may include the confusion matrix, per-class metrics, top confusions, and standard
figures. Segmentation results may include sample metrics ordered from lowest Dice
score. Images under standard `figures`, `evaluation`, `overlays`, or `activations`
directories are cataloged as figures, prediction overlays, or activation/saliency
evidence.

`GET /v1/artifacts/{artifact_id}/evidence/files/{relative_path}` serves only
supported image files contained by that artifact directory. Absolute paths,
directory traversal, and non-image files are rejected. These endpoints never run
inference or calculate new evidence; missing products remain explicitly missing.

## File explorer

`GET /v1/files/roots` and `GET /v1/files?root=workspace&path=configs` support
the web GUI file explorer. It is read-only and confined to allow-listed roots:
the Orchestrator working directory (`workspace`), the owned artifact root
(`artifacts`), and any additional `--browse-root DIRECTORY` values. Paths are
relative to a selected root and requests that escape it are rejected.

`POST /v1/catalog:scan` accepts only directories within those same approved
roots. It validates and indexes standard `artifact.json` manifests, reports
skipped directories with reasons, and never rewrites the artifact itself.

The web GUI uses a resumable transfer protocol for files of any supported
size: create a session with `POST /v1/uploads/sessions`, send fixed 16 MiB
`PUT /v1/uploads/sessions/{id}/part` ranges with `Content-Range`, then atomically
publish it with `POST /v1/uploads/sessions/{id}:complete`. The session ledger
records received ranges, so selecting the same file and retrying after a
network interruption continues rather than restarts the upload. Partial paths
are never returned or eligible for registration.

`GET /v1/datasets/{id}:download` streams a registered SQLite training set.
`GET /v1/artifacts/{id}:download` streams a portable `.tar.gz` model-artifact
archive. Neither endpoint exposes a storage path. Supported upload kinds are
`datasets` (`.sqlite`), `configs` (`.toml`), and `models` (`.keras`, `.h5`, or
`.hdf5`); existing files are never overwritten. Use `--upload-limit-mib` to
set the service limit.

`POST /v1/uploads/{kind}/{filename}` remains as a streaming one-request
compatibility endpoint for the CLI and simple integrations; new browser work
should use resumable sessions.
