# Oracle WorkUnit V1

`WorkUnit V1` is the immutable, location-free contract produced by the Oracle
Orchestrator for one compute attempt. It is shared by the orchestrator and
future `oracle-worker` processes through `oracle_data_contracts`.

## Contract

```json
{
  "schema": {"name": "oracle_work_unit", "version": 1},
  "work_unit_id": "UUID",
  "attempt_id": "UUID",
  "specification_id": "UUID",
  "action": "train",
  "inputs": {
    "input": {
      "kind": "dataset",
      "artifact_id": "dataset-id",
      "revision": "revision-id",
      "fingerprint_sha256": "..."
    }
  },
  "configuration": {"kind": "configuration", "artifact_id": "..."},
  "staging": {"kind": "staging", "artifact_id": "job-id", "revision": "attempt-id"},
  "resources": {"gpu_count": 1}
}
```

Artifact references are portable tokens; they cannot contain paths or storage
credentials. The canonical JSON encoding is SHA-256 hashed and persisted with
the orchestration job at dispatch time.

## Portable input publication

New pull-lane plans carry portable refs in their parameters before a WorkUnit
is built:

```json
{
  "artifact_inputs": {"input": {"kind": "dataset", "artifact_id": "dataset-id", "revision": "1", "fingerprint_sha256": "..."}},
  "configuration_artifact": {"kind": "configuration", "artifact_id": "config-id", "fingerprint_sha256": "..."}
}
```

The orchestrator copies a submitted dataset SQLite file or TOML configuration
into its artifact store before it queues work. A file artifact is an immutable
directory with `payload` (the bytes the executor consumes) and `manifest.json`
(source-name metadata, SHA-256, and size). The original source path is never
put in the WorkUnit, lease, grant state, or worker request. New file refs are
SHA-pinned and the store rejects a content mismatch before publication.

`build_work_unit(..., require_portable=True)` requires these explicit refs for
any raw `input`, `model`, `info`, `run`, or `config` parameter. It refuses a
`legacy_file` fallback. `register_existing` is retained only for offline
historical migration and must not be used by active scheduling.

There is no local-path or push-compute compatibility mode in the supported
runtime. Active WorkUnits must use portable artifact references; host paths are
accepted only by offline migration readers and never enter scheduling state.

## Lease-scoped artifact materialization

When a worker receives a durable lease, the orchestrator can issue one
short-lived materialization grant for each input artifact. The immediate lease
response has this shape:

```json
{
  "ref": {"kind": "dataset", "artifact_id": "dataset-id", "revision": "revision-id"},
  "grant_id": "opaque-grant-id",
  "token": "opaque-bearer-token",
  "expires_at": "2026-01-01T00:10:00+00:00"
}
```

The grant is bound to both the lease and job, expires quickly, and may be used
once. The local artifact store persists only the token digest and portable
artifact reference; it does not put source paths, storage roots, or bearer
tokens in durable job/lease state. Normal grant metadata excludes `token`;
the token belongs only in the immediate authenticated worker response.

The HTTP delivery boundary is two-step: a worker requests a grant for an input
or configuration ref, then downloads the server-supplied path with its worker
bearer, lease token, and grant token. The response is a symlink-free gzip tar
with one `artifact/` root. Workers must safely extract it into worker-owned
scratch space; the transport root is not exposed to the action adapter.

## Pull execution and publication

Before working, a pull worker acknowledges its lease. The Orchestrator then
allocates the staging attempt. Workers may append bounded, idempotent progress
events, but cannot supply job status, output paths, or artifact identities.

After execution, the worker uploads a ZIP or TAR staging archive to its lease
endpoint using its worker bearer and lease token. The API spools the request on
the control-plane host, rejects traversal, links, special files, empty output,
and unsafe expansion, then extracts only into the allocated staging area. On a
successful completion report, the Orchestrator validates, seals, and publishes
the candidate as `job_output:<job-id>@<attempt-id>`. Failed completion never
publishes an artifact. Specialized training/model publication remains a later
action-specific validation and cataloging step.
