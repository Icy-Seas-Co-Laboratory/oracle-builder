# Oracle Builder Orchestrator API

The Orchestrator is the Pelagia-facing control plane for Oracle Builder.  It
owns one SQLite database containing catalog summaries, experiments, immutable
run specifications, and orchestration job state.  It does not replace Oracle
Builder artifact contracts: datasets and artifacts remain the scientific source
of truth and may be re-ingested after recovery.

```bash
oracle-orchestrator --database /oracle/control/orchestrator.sqlite \
  --workspace-root /oracle/workspace \
  --artifact-root /oracle/artifacts \
  --oracle-serve Local=http://127.0.0.1:8100 --port 8110
```

## Core workflow

1. Register a frozen SQLite dataset with `POST /v1/datasets:ingest`.
2. Register a validated TOML training recipe with `POST /v1/recipes`.
3. Create a typed training experiment at `POST /v1/experiments:train` with a
   dataset, recipe IDs, and explicit seeds.
4. Dispatch an immutable specification to a chosen `oracle-serve` endpoint at
   `POST /v1/specifications/{id}:dispatch`.
5. Reconcile transient compute state with `POST /v1/jobs/{id}:reconcile`. A
   completed training or import job is validated and indexed automatically.

Compute endpoints are configured by operators with repeatable
`--oracle-serve NAME=URL` arguments and persisted in the Orchestrator database.
Clients inspect live endpoint, queue, and worker state with
`GET /v1/compute/endpoints?refresh=true`. Before dispatch they call
`GET /v1/specifications/{id}/preflight?endpoint_id={endpoint-id}`. The same
preflight is enforced by the dispatch endpoint, including action support, GPU
capacity, endpoint readiness, and queue capacity.

For model-producing jobs, durable status progresses through `submitted`,
`queued`, `running`, `validating`, and finally `indexed`. Compute failures use
`failed`; valid process output that fails the artifact contract uses
`artifact_invalid`. `GET /v1/jobs` returns the resolved output path, worker and
timing fields, structured validation report, and linked artifact summary.

To import an external model, first stage or browse to a `.keras`, `.h5`, or
`.hdf5` file and a TOML file containing a `[product]` section. Create the
immutable import specification with `POST /v1/model-imports`, then dispatch it
through the same review gate. An optional registered dataset records provenance.

The orchestrator owns output locations. For training it assigns
`{artifact-root}/runs/{specification-id}` through `runs_dir` and `output`; for
evaluations, imported products, and packages it assigns their corresponding
subdirectories. UI clients may supply inputs and configuration references, but
cannot choose arbitrary derived-artifact paths.

Training experiments create one immutable specification per selected recipe and
seed. Every specification has a UUID, ordinal, generated configuration snapshot,
configuration hash, and exact compute request.

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

`POST /v1/uploads/{kind}/{filename}` stages raw uploaded bytes only in the
owned artifact root. Supported kinds are `datasets` (`.sqlite`), `configs`
(`.toml`), and `models` (`.keras`, `.h5`, or `.hdf5`). Existing files
are never overwritten. Use `--upload-limit-mib` to set the service limit.
