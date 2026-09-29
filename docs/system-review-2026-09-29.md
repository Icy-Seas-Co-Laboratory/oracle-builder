# Oracle Builder independent system review

**Review date:** 2026-09-29
**Reviewed snapshot:** the uncommitted workspace present during the queue-workflow and dashboard changes
**Review relationship:** independent of the implementation work requested in the same session

Line links identify the reviewed snapshot and may move as active changes are finalized. Findings outside the requested queue and dashboard work are recommendations; they were not automatically fixed.

## Scope and method

- the FastAPI control plane and SQLite state model;
- pull-worker leasing, execution, cancellation, and artifact transfer;
- dataset identity, split construction, training, recovery, and evaluation;
- the SvelteKit Web UI and its server-side API proxy;
- Docker/Caddy deployment, readiness, operations, and audit behavior;
- representative tests and the full Python test suite;
- frontend type checking and production build.

No repository files were changed as part of the review other than this report. The two access-control findings are **P0 only when the application is reachable outside a fully trusted local network**; they remain P1 hardening gaps for a private-lab deployment.

## What is already strong

Oracle Builder has a sound execution core:

- Work units are typed, sealed, path-free contracts rather than arbitrary remote commands.
- Worker and lease credentials are scoped, hashed at rest, and checked at the worker boundary.
- Lease claiming uses transactions and unique active-lease constraints.
- Archive extraction rejects traversal, links, and special files.
- Multipart output upload is bounded, resumable, and digest checked.
- Dataset identities, split manifests, run seals, and artifact checksums provide a strong basis for reproducible science.
- Startup publication reconciliation handles the narrow crash window after atomic artifact publication.
- The Python suite provides broad behavioral coverage across data, training, orchestration, transfer, and recovery.

## Prioritized findings

### 1. Conditional P0: the Web UI is an unauthenticated operator-token relay

**Confirmed.** The SvelteKit catch-all proxy accepts every supported verb, removes caller authorization, and injects the server's
shared operator token ([proxy](../webgui/src/routes/api/%5B...path%5D/+server.ts#L5)). There is no UI login, session, or authorization
hook in `webgui/src`. Caddy exposes the UI ([routing](../deploy/docker/Caddyfile#L21)), and the orchestrator container can hold the
host Docker socket ([compose](../deploy/docker/compose.yaml#L18)).

**Impact:** any visitor who can reach the UI can perform operator mutations, including model submission, data upload, and Docker worker
lifecycle actions.

**Action:** add authenticated UI sessions and authorization before the proxy. Replace the mutation-capable catch-all with explicit
backend-for-frontend routes, carry a human actor ID into audit records, and keep the current demo on a trusted network until this
boundary exists.

### 2. Conditional P0: sensitive control-plane reads are anonymous

**Confirmed.** GET and HEAD requests bypass role enforcement and receive rate limiting instead of authentication
([middleware](../oracle_builder/orchestration/api.py#L389)). Anonymous reads include system logs
([logs](../oracle_builder/orchestration/api.py#L475)), browse roots and files ([files](../oracle_builder/orchestration/api.py#L1090)),
dataset downloads ([datasets](../oracle_builder/orchestration/api.py#L1272)), and model artifact downloads
([artifacts](../oracle_builder/orchestration/api.py#L1393)). Caddy exposes `/v1/*` directly ([control
route](../deploy/docker/Caddyfile#L11)). A security test explicitly preserves this behavior
([test](../tests/test_orchestration_api_security.py#L19)).

**Impact:** training data, models, logs, local path metadata, job history, and worker topology can be disclosed without credentials.

**Action:** leave liveness public; require at least viewer authorization for catalog and metric reads, and stronger permissions for
payloads, logs, topology, and file browsing. Separate metadata permissions from dataset/model download permissions.

### 3. P1: local artifact fingerprints are not revalidated at use time

**Confirmed by runtime reproduction.** Ingest verifies a portable file fingerprint
([ingest](../oracle_builder/orchestration/storage.py#L599)), but local resolve, materialize, grant issue, and delivery only check
availability ([resolve](../oracle_builder/orchestration/storage.py#L667), [grant](../oracle_builder/orchestration/storage.py#L684),
[stream](../oracle_builder/orchestration/storage.py#L867)). The worker extracts the downloaded archive without comparing it with the
`ArtifactRef` fingerprint ([worker materialization](../oracle_builder/worker/pull.py#L938)).

After a published `payload` was changed, `materialize()` accepted and copied bytes with a different SHA-256 value.

**Impact:** accidental or unauthorized local modification can silently change the dataset or configuration executed under an immutable
reference.

**Action:** publish a deterministic content manifest, verify it before granting or streaming, include a delivery digest, and verify it
on the worker. Quarantine mismatches and add a post-publication tampering test.

### 4. Addressed in production paths: automatic GPU reservation lifetime

The initial review confirmed that auto-GPU flock handles survived for the worker process lifetime. The current owned, context-local scope
releases reservations in `finally` ([scope](../oracle_builder/training/distribution.py#L48)) around production training
([workflow](../oracle_builder/training/workflow.py#L66)) and probes ([probe](../oracle_builder/training/batch_tune.py#L48)). Tests cover success,
failure, nested ownership, and reacquisition ([tests](../tests/test_distribution_gpu_load.py#L22)). Low-level selector calls outside a scope still
retain reservations; three order-dependent selector tests expose this, so direct callers and tests must use the scope or explicit cleanup.

### 5. P1: cancellation and lost leases do not interrupt long training

**Confirmed.** Cancellation is checked before and after the blocking executor call ([execution
path](../oracle_builder/worker/pull.py#L1020)). Lease-renewal failure is raised only after execution returns, and one renew error
permanently stops the keepalive ([keepalive](../oracle_builder/worker/pull.py#L1116)). Expiry can immediately queue an infrastructure
retry ([retry policy](../oracle_builder/orchestration/service.py#L2035)).

**Impact:** an original training process can keep consuming compute after cancellation or lease loss while the control plane starts a
retry elsewhere.

**Action:** pass a cancellation event into the executor; check it through TensorFlow callbacks at bounded batch or epoch boundaries;
tolerate bounded transient renew errors; stop safely after sustained lease loss; and fence publication by execution-attempt generation.
A child process or process group provides bounded forced termination.

### 6. P1: a crash can strand durable operations permanently

**Confirmed.** Operation rows have no owner, heartbeat, attempt, or reclaim deadline
([schema](../oracle_builder/orchestration/database.py#L195)). The single operation runner marks a row running and then executes it
synchronously ([runner](../oracle_builder/orchestration/service.py#L388)). Startup reconciliation does not recover running operations
([startup](../oracle_builder/orchestration/service.py#L4064)).

**Impact:** interrupted scans, validation, or replication remain `running`; a stranded worker reconciliation row also suppresses future
reconciliation scheduling.

**Action:** lease operation claims with owner, heartbeat, attempt count, and retry policy; reclaim stale operations at startup; and
give lease/publication reconciliation a separate critical lane from user maintenance work.

### 7. P1: evaluation can silently substitute validation for test data

**Confirmed.** Streaming classification falls back when the test index is empty ([streaming
evaluation](../oracle_builder/evaluation/reports.py#L55)). The non-streaming path catches every `ValueError` and then loads validation
([array evaluation](../oracle_builder/evaluation/reports.py#L85)), which can hide corrupt data, decode failures, or shape errors. The
workflow labels the stage as held-out test evaluation ([workflow](../oracle_builder/training/workflow.py#L745)).

**Impact:** validation data used for model selection can become the reported final estimate, and unrelated data defects may be
misreported as an empty test split.

**Action:** catch only an explicit empty-split exception, require an opt-in fallback policy, prominently record `evaluated_split`, and
prevent production-quality sealing when no independent test split exists.

### 8. P1 scientific-risk opportunity: fallback splits ignore labels and groups

When complete source partitions are unavailable, the fallback ranks only `item_id` hashes
([assignment](../oracle_data_contracts/artifacts/splits.py#L42), [policy](../oracle_data_contracts/artifacts/splits.py#L131)). It
cannot preserve rare-class representation or keep related crops, subjects, cruises, or source images together.

**Impact:** some datasets may lose classes from validation/test or leak correlated observations across splits, inflating scientific
performance. The actual risk depends on dataset provenance.

**Action:** add stratified and group-stratified policies, require the relevant metadata key, publish class/group counts, and validate
zero group overlap.

### 9. Addressed during implementation: verification environment binding

The built-in worker now advertises a fresh execution-instance ID per boot ([CLI](../oracle_builder/worker/cli.py#L101)). Lease issuance
records a digest of the complete capability environment; verification evidence and the training policy pin that lease-issued digest rather
than later mutable worker state ([recording](../oracle_builder/orchestration/service.py#L2977)). Admission requires both worker and environment
to match ([matching](../oracle_builder/orchestration/service.py#L1966)). A capability refresh cannot occur during an active lease; afterward,
a mismatch cancels obsolete queued training, clears consent, and requires explicit re-verification
([invalidation](../oracle_builder/orchestration/service.py#L2116)). Crash recovery supplies the original lease ID and therefore uses its
recorded environment. Parametrized tests cover both ready and auto-start-queued invalidation ([tests](../tests/test_pull_queue.py#L230)).

### 10. Addressed during implementation: sealed output publication boundaries

The real CPU path exposed two pre-existing defects. The worker sanitized `provenance/runtime.json` after workflow sealing, invalidating the
artifact inventory; it now explicitly reopens, sanitizes, and reseals before upload ([packaging](../oracle_builder/worker/pull.py#L317),
[test](../tests/test_worker_batch_calibration.py#L95)). Completion also tried to resolve model and inference identity through the writable
staging API after validation/sealing had closed it. Identity is now read and validated before that boundary
([publication](../oracle_builder/orchestration/service.py#L3068), [test](../tests/test_pull_queue.py#L86)). The narrow regressions pass.

The focused queue review found no severe consent bypass, duplicate dispatch, environment-binding, or publication-recovery gap. Queueing creates no job;
verify and start are distinct transactions; no-start is the API default; failure and cancellation clear consent; auto-start survives restart;
recovery replays completion; and legacy workers are gated.

## Reliability and maintainability actions

### Current test baseline

The final full-suite run reported **821 passed and 79 failed**. The focused queue/worker/GPU set completed with **35 passed**.

A real one-epoch CPU run passed queue → manual/automatic forward-backward verification → explicit start → training → validated, sealed,
published, and indexed model artifact. The latest queue/batch/inference regression set reported **23 passed**; both publication regressions
also passed independently in this review.

- The 79 remaining failures predate the queue change: 74 stale classification preset assertions, three order-dependent tests that call the
  low-level GPU selector without a lease scope, one legacy ROI migration failure, and one unlabeled-pretraining failure.
- Two transient queue expectation failures from the initial snapshot are resolved; they had expected a newly queued run to be ready
  immediately. The new queue verification and environment-invalidation tests pass.

The V2 preset generator's `--check` passes, while [classification preset tests](../tests/test_classification_default_configs.py#L69)
still assert removed V1 aliases. Update the tests to resolve V2 authoring configuration before checking runtime aliases.

Two additional functional defects are directly exposed by the suite:

- unlabeled classification records call `int(None)` while being prepared for self-supervised pretraining ([record
  construction](../oracle_builder/data/sqlite_dataset.py#L287));
- split-manifest identity opens a legacy ROI database without first invoking the compatibility migration ([identity
  check](../oracle_data_contracts/artifacts/splits.py#L285)).

Frontend `npm run check` and `npm run build` passed at the reviewed snapshot.

### SQLite startup and request cost

Every service connection sets WAL, executes the full schema, inspects columns, and runs migration logic
([connect](../oracle_builder/orchestration/database.py#L414)); normal service methods obtain connections through this path ([service
connection](../oracle_builder/orchestration/service.py#L168)).

Move versioned migration to startup under an explicit transaction. Keep request connections lightweight, set an explicit busy timeout,
and load-test concurrent dashboard reads, worker heartbeats, lease claims, and upload parts.

### Readiness, audit, and deployment hardening

- Readiness returns HTTP 200 even when it reports `degraded` ([readiness](../oracle_builder/orchestration/api.py#L455)), while the
  launcher treats any 2xx as ready ([launcher](../scripts/start_oracle_stack.sh#L132)). Add dependency/capacity readiness, 503
  semantics where appropriate, and Docker health checks.
- Audit records are written only for successful mutations after authorization
  ([middleware](../oracle_builder/orchestration/api.py#L417)). Record denied and failed attempts, request ID, actor identity, outcome,
  and safe resource identifiers.
- Python and Node images run as root ([Python image](../deploy/docker/Dockerfile#L6), [Web
  image](../deploy/docker/webgui.Dockerfile#L9)). Use non-root users, read-only filesystems, dropped capabilities, resource limits, and
  a narrow Docker lifecycle sidecar.
- Materialization grants use a process-local lock around a JSON registry ([grant
  registry](../oracle_builder/orchestration/storage.py#L424)). Document the single-control-plane-replica invariant; move this state to
  transactional storage before horizontal scaling.

### Dashboard aggregation and UI verification

The refreshed dashboard now uses the shared operational refresh subscription, includes every active job, retains selected/recent jobs,
and uses per-request fallback so one failed detail call does not discard the whole snapshot ([dashboard
load](../webgui/src/lib/LiveDashboardView.svelte#L82)).

The remaining opportunity is server-side aggregation and pagination. The client still requests events, timing, and upload state
separately for each visible job. Add a dashboard snapshot or batch-detail endpoint, cursor recent history, and let SSE carry small
deltas between snapshots.

The frontend package has check/build scripts but no component or browser test command ([scripts](../webgui/package.json#L6)). Add
focused tests for queue verify/start transitions, selection persistence, modal focus/keyboard behavior, SSE reconnect, and partial API
failure.

## Recommended sequence

Before network exposure, implement human authentication and authenticated reads. Then close local artifact revalidation,
add cancellation/lease fencing and operation reclaim, make evaluation and split policies explicit, restore a green CI gate, separate
startup migration from request connections, harden deployment, and add dashboard aggregation plus browser workflow tests.


## Post-implementation disposition

The findings above describe the independently reviewed baseline. The subsequent
execution-contract implementation addresses artifact consumption verification
(finding 3), owned GPU reservation scope (4), supervised cancellation and lease
loss (5), silent test-split substitution (7), and deterministic class coverage
with optional explicit groups (8). V2 attempts retain verification provenance
and recheck the frozen batch on compatible replacement workers (9).

Review reconciliation also fixed stale pause acknowledgements during cancel,
malformed cache quarantine, nested Keras ZIP detection in output tar archives,
checkpoint hardlink packaging, and custom-layer loading in fresh worker
processes. New regressions cover these failures.

The broader access-control, durable operation recovery, request-cost, and
production observability recommendations remain follow-up work; this execution
change does not establish authenticated browser sessions or change deployment
exposure. See [implementation validation](execution-implementation-validation.md)
for the delivered execution surface and its scientific limits.
