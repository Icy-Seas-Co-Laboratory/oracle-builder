# Execution contracts, worker control, integrity, and split policies

Status: implementation record and remaining plan, 2026-09-29. The sections below
retain the original design rationale; the following status makes delivered scope
explicit.

## Implemented status

V2 run/unit/attempt contracts, durable directives and acknowledgements,
heartbeats, lease fencing, checkpointed epoch/cycle continuation, verified
artifact delivery, class-stratified split manifests, and ordered inference
shards are implemented. A default inference shard size of 1024 is selected by
the API; shard and merge workers require `inference_shards_v1` capability.

Each segment currently runs in a separate supervised process. The verified
worker cache reduces input transfer, but warm GPU/model residency is deferred.
Final evaluation and packaging are retried as finalization work and never cause
completed training to run again. CPU ordered-step interruption is available
only for units that advertise it; other work stops at epoch/cycle boundaries.
Recovery preserves supported continuation state but does not promise bitwise
RNG equivalence across hardware or unsupported custom loops.

Measured smoke coverage includes a two-epoch run executed as four work units
and a 24-item inference run executed as three shards plus a merge.

## Original objective and baseline (before implementation)

Keep user-facing queue → verify → start consent, while making execution interruptible, resumable, and independently verifiable. A user starts one logical run; scheduling its continuation units does not require repeated Start clicks. Pausing suspends continuation until Resume.

At planning time, the implementation had portable, hashed WorkUnit V1 contracts, transactional leases, staged publication, background lease renewal, epoch recovery snapshots, and immutable run split manifests. The gaps identified then were:

- `oracle_data_contracts/work_units.py` describes one worker attempt, effectively an entire training run.
- `oracle_builder/worker/pull.py` checks cancellation around a blocking executor; a renewal failure is surfaced only after execution returns.
- `oracle_builder/training/recovery.py` saves a full model and completed epoch but does not establish a complete continuation contract for data ordering, random state, and callback state.
- `oracle_builder/classification/stratified_training.py` owns shared-weight scheduling across resolution strata. These strata cannot be dispatched as independent concurrent model updates.
- `oracle_builder/orchestration/storage.py` verifies portable files at ingestion but does not verify local content at every consumption boundary. Worker extraction also lacks an expected-content check.
- `oracle_data_contracts/artifacts/splits.py` supports imported partitions or deterministic item-hash ranking, without class/group constraints.

## Design decisions

### 1. Separate run, work unit, attempt, and lease

A **run** is the complete user request, frozen scientific configuration, and ordered execution plan. A **work unit** is a deterministic, bounded advancement from an immutable input state to a declared output state. An **attempt** is one execution of that unit. A **lease** is temporary authority to execute and commit that attempt.

WorkUnit V2 should contain:

- schema version, run ID, stable unit ID, unit sequence, phase, and predecessor;
- dataset, effective configuration, split manifest, and input checkpoint references with explicit content digests;
- start cursor and stop boundary: epoch/cycle initially, optimizer step or item shard later;
- action-specific output contract and checkpoint schema;
- resource requirements, execution compatibility requirements, and supported interruption boundaries;
- deterministic seeds and versioned execution/data-order policies.

Put attempt ID, attempt generation, worker boot ID, lease token/expiry, and staging destination in a separate execution envelope. Retries reuse the same unit identity and inputs with a new attempt and generation. Runtime directives do not mutate the sealed unit.

Use explicit typed phase parameters, including a real verification phase, rather than adding more loosely interpreted flags under `queue_execution`. Hash strict canonical JSON and reject non-finite values and unsupported schemas.

Suggested run chain:

`verify → wait for Start → initialize → train segment 1 → train segment 2 → … → evaluate → package/publish`

Verification may overlap initialization work only when its output is explicitly reusable and immutable. It must not update model weights. Split assignment is resolved during verification and pinned for all later units.

Initially implement an ordered chain, with dependency fields that permit future branching. Do not introduce a general-purpose DAG scheduler in the first release.

### 2. Choose boundaries from resumable scientific state

Start with one completed epoch for ordinary supervised training and one complete shared-weight cycle for resolution-stratified training. A cycle includes its scheduler/validation decision; a stratum is not automatically an independent model.

These boundaries are correct initial checkpoints, but do not bound wall time when epochs are very long. Add step-bounded segments after sampler position, partial-epoch metric state, and callback behavior are resumable. A proposed operational target is 5–15 minutes per segment; benchmark before choosing the default. This is a soft scheduling target, not an epoch-mode guarantee.

Inference and bulk prediction should later use deterministic item-ID shards and an idempotent merge, with separate evaluation and packaging units. Training completion must not require redoing training if evaluation or packaging fails.

Preserve warm-worker locality: after checkpoint commitment, let the same process continue into the next authorized unit without unloading the model. Require fresh admission and a committed boundary even on this fast path. Cache pinned datasets/configurations across units and avoid retransferring them.

Use one active writer per training run. Scheduling fairness, priority, and resource availability are evaluated between units. Prefer locality but allow compatible-worker migration. The existing same-worker verification rule remains until a versioned compatibility contract supports safe migration; initially reverify on a new worker and never silently change an established training batch size.

### 3. Make checkpoints durable continuation artifacts

A checkpoint must include model weights, optimizer/slot and loss-scaling state, learning-rate scheduler state, early-stopping/best-model state, RNG state or a reproducible RNG derivation scheme, global step, epoch/cycle/stratum cursor, data shuffle/sampler position, partial metrics where applicable, and the pinned dataset/config/split identities. Record framework/model-code versions and compatibility requirements.

Audit each execution phase for support. Unsupported pretraining/custom loops must reject resumable scheduling or explicitly advertise whole-phase-only recovery; they must not claim guarantees they cannot satisfy.

Write immutable checkpoint generations and an atomic manifest, rather than overwriting a model file before its companion state file. Publish the checkpoint to durable storage before advancing the run cursor. Retain at least the prior committed checkpoint until the next one is validated and committed. Keep local uncommitted recovery separately; it is not authority to advance a run.

Use attempt generations to fence every checkpoint/output commit. Commit the selected output, run cursor, and next-unit eligibility in one database transaction, backed by reconciliation for the filesystem/database crash window. Repeating completion must return the original result rather than duplicate the next unit.

The guarantee is at-least-once execution with at-most-one accepted advancement per unit. A network partition can temporarily duplicate compute; stale attempts must never commit. Bitwise reproducibility across different hardware is not promised.

### 4. Worker control is independent of model execution

Run training in a supervised child process/process group. The worker parent owns credentials, control traffic, heartbeats, leases, and publication. Child callbacks communicate through bounded local IPC and do not perform blocking network requests. Cover materialization, execution, checkpoint upload, evaluation, and packaging, not just `model.fit`.

Use a worker-initiated authenticated long-poll command channel initially. It fits the existing HTTP/pull architecture and requires no inbound worker port. Keep periodic heartbeat/renewal requests separate; a long poll must not block renewals. A later WebSocket transport can reuse the same durable protocol if measurements justify it.

Proposed starting settings: heartbeat every 5 seconds, 30-second command long-poll, 30-second lease TTL with a conservative local stop margin. Command arrival wakes the pending poll immediately. Treat these as configurable values to tune under load, not universal guarantees.

Heartbeats report worker boot ID, attempt/generation, current phase and cursor, last progress time, last committed checkpoint, and last applied command sequence. Distinguish worker alive, executor progressing, and lease valid. Retry transient network failures with bounded jitter; derive a conservative monotonic local deadline from renewal and stop execution before authority is uncertain. Lease rejection or generation mismatch triggers immediate revocation handling.

Persist commands with command ID, target run/unit/attempt generation, sequence, issuer, reason, creation time, and optional deadline. Deliver at least once; handle duplicates idempotently. Persist received → accepted → applied/failed/expired acknowledgments, including checkpoint references and effective stop position. Reject stale commands for replaced attempts. Stop/revoke supersedes pending restart/continue commands; conflicting operations use an explicit transition table.

| User action | Execution semantics | Scheduling semantics |
| --- | --- | --- |
| Pause safely | Stop at the next supported safe boundary, publish a resumable checkpoint | Keep the run paused until Resume |
| Yield | Same checkpoint handoff as Pause | Release resources and make continuation eligible elsewhere |
| Stop now / cancel | Terminate the owned process group; do not wait for a new checkpoint | Cancel continuation; retain the last committed checkpoint and diagnostics |
| Restart current unit | Stop/fence the current attempt and discard its uncommitted advancement | New attempt from the unit's original input checkpoint |
| Resume | Start from the latest committed cursor/checkpoint | Continue the paused run |

Do not conflate Restart with Resume. A graceful pause can finish at an epoch/cycle boundary initially; the UI must report that latency. Once step-level recovery is supported, checkpoint at a batch callback. A long native GPU call can delay callbacks; the parent escalates termination after a configurable deadline. Forced termination has no fresh-checkpoint guarantee. A stopped but still uploading attempt must also be fenced.

Checkpoint-and-yield within a partially completed unit creates an explicit committed continuation cursor and a new immutable successor descriptor. Never mutate the old unit's start or stop boundary. Phase one avoids this complexity by yielding only at natural unit boundaries.

### 5. Verify content at consumption boundaries

Implement this first because all resumability depends on trustworthy inputs and checkpoints.

- Distinguish raw payload SHA-256, canonical dataset semantic fingerprint, directory content-manifest digest, and transport archive digest. Version/name the digest scheme; do not compare unlike identities.
- For portable file artifacts, verify the payload against the pinned raw digest. For directory artifacts, pin a canonical sorted inventory of relative paths, file sizes, and per-file hashes. Reject missing, extra, symlink, and special-file content.
- Verify local materialization and the exact snapshot used to prepare a delivery archive. A hash checked before copying is insufficient if the source can change during copying. Build/verify a private snapshot and serve those verified bytes; repeat the content check after worker extraction.
- Bind the expected content digest to the sealed work unit and authenticated grant. An archive-provided manifest alone is not a trusted expected identity. Transport hashes detect truncation/corruption but do not replace content identity.
- Validate worker cache entries before reuse, initially on every admission; consider a verified immutable cache optimization only after measurements and a clear immutability guarantee.
- Reject and quarantine/report mismatches without silently replacing an immutable artifact or retrying it as a transient compute error. Record expected/observed identity and affected consumers.
- Define an explicit validation/backfill path for old references lacking content digests; do not relabel historical semantic hashes as raw byte hashes.

### 6. Deterministic class coverage with optional group isolation

Treat **class**, **leakage group**, and **resolution stratum** as different concepts. Classes should be represented across splits; when grouping is explicitly configured, each group stays entirely within one split. Resolution routing occurs after split assignment. Tiles, crops, and augmentations inherit the original item's split.

Confirmed user priority: reliable grouping cannot currently be inferred; label representation matters most. Default newly generated ROI splits to **class-stratified**, without requiring or inventing group IDs. Retain stratified-group splitting as an explicit optional policy when reliable metadata becomes available. Do not infer grouping from filenames or use `dataset_items.source_key` directly as a group: the schema makes it unique per item. If optional grouping is selected, require its declared metadata key. When multiple independence relationships apply, group connected items together; a composite key must not accidentally divide the same specimen across sources.

Support explicit policies: imported/source partitions, stratified group, group-only, class-stratified, and deterministic random. Freeze existing manifests and run configurations. Preserve trusted imported benchmark partitions, validate them for leakage/coverage, and require an explicit decision when they conflict with constraints. Never silently resplit a holdout to satisfy a ratio.

Resolve the exact training population and effective class labels first, including filtering, annotation/concept mapping, and unlabeled/pretraining rules. For single-label classification, allocate each class's samples deterministically across enabled splits, first satisfying minimum coverage, then approximating requested ratios with stable integer allocation. Order samples by a versioned seeded item-ID hash, independently of database row order. Coverage takes priority over exact overall ratios, and deviations are visible. For optional grouping, assign whole groups using deterministic assignment plus repair; distinguish an unsatisfied search from proven infeasibility. A bounded deterministic constraint-solver fallback may be needed for hard grouped or multilabel cases and must report timeout/unknown honestly. Multilabel and segmentation targets need explicit task-specific coverage semantics; do not silently apply the single-label algorithm.

Require every requested class to have configured minimum support in each enabled split by default (initial minimum: one sample per class per enabled split). For ordinary three-way single-label splitting, each class therefore needs at least three samples; higher configured minima require more. Presence alone does not establish a statistically useful evaluation population, so report support counts prominently. With optional grouping, a necessary, but not sufficient, condition is that each class occurs in at least three independent groups. Mixed-class groups and other constraints can still prevent a feasible assignment.

Fail verification with a useful feasibility report when strict coverage or explicitly requested grouping cannot be met. Offer explicit alternatives such as fewer splits, relaxed coverage, or additional data. Never duplicate samples across splits, oversample validation/test, or break configured group isolation to manufacture coverage. Missing group metadata fails only a policy that explicitly requires grouping; it does not block the default class-stratified policy.

Report sample/class counts, class support per split, absent classes, ratio deviations, item-disjointness checks, and resolution-stratum coverage. Add group counts/zero-overlap checks only when grouping is configured; otherwise report group leakage as unassessed. Sparse resolution-by-class cells should be diagnostic by default; making every such cell mandatory is a separate policy and may be impossible. Unlabeled data needs an explicit deterministic pretraining split policy, optionally grouped, rather than a fabricated class.

Persist the manifest once during verification. Record dataset/label/group-policy identity, algorithm version, seed, assignments, and a stable semantic assignment digest excluding timestamps and random manifest IDs. Pin that same manifest across comparisons when evaluating candidate models. Training, resume, evaluation, and inference use it unchanged.

Remove silent validation-for-test substitution alongside this work: report the actual evaluated split and an explicit unavailable-test result. Any fallback must be opt-in and labeled.

## Delivery sequence and acceptance gates

| Slice | Main changes | Required evidence before proceeding |
| --- | --- | --- |
| 0. Contracts and fixtures | ADR for V2, lifecycle/command transition tables, digest semantics, effective-label audit; fixtures for ordinary and shared-weight training | Contract round trips, invalid transitions rejected, documented supported recovery phases |
| 1. Artifact integrity | Shared dependency-light verification utilities; store snapshot verification; worker extraction/cache checks; legacy validation tooling | Published-file tampering, manifest tampering, missing/extra files, mid-copy mutation, wrong ref, and cache corruption rejected before execution |
| 2. Split policy | Default effective-label class stratification; optional explicit group resolver; versioned allocator; feasibility/coverage report; verification-time manifest publication; UI preview | Row-order invariance, same-seed assignment identity, all-class coverage when feasible, rare-class failures, optional-group isolation/metadata failures, source-partition conflicts, tile inheritance, existing manifests unchanged |
| 3. Worker supervision and commands | Parent/child execution; independent heartbeat/long-poll; durable command/ack state; monotonic lease deadline; process-group stop; generation fencing | Pause/cancel while training and uploading; duplicate/stale commands; transient disconnect/reconnect; partition beyond lease; hung child; orchestrator restart; no stale output accepted |
| 4. Durable checkpoint state | Generation-based atomic snapshots; optimizer/callback/RNG/data cursor audit; checkpoint publication and replay-safe commit | Resume matches uninterrupted deterministic CPU reference within stated tolerance; corrupt/incomplete checkpoint rejected; crash before/after each commit step recoverable |
| 5. Sequential WorkUnit V2 | Run/unit/attempt tables and admission; epoch/cycle units; warm continuation; locality/fairness; evaluate/package separation | Mid-run worker death retries only uncommitted work; duplicate completion cannot duplicate advancement; checkpoint migration works on compatible workers; packaging retry never retrains |
| 6. Bounded step/shard units and UI | Step cursor/partial metrics; partial-unit yield continuation; inference shards; run/unit/attempt drill-down; command acknowledgments and checkpoint age | Long-epoch preemption, repeated pause/resume/restart correctness, no duplicate/missing shard output, interruption latency and transfer overhead benchmarks |

Primary files: `oracle_data_contracts/work_units.py`, `oracle_data_contracts/artifacts/splits.py`, shared artifact-contract helpers, `oracle_builder/orchestration/{database,service,api,storage,work_units}.py`, `oracle_builder/worker/{pull,cli}.py`, training workflow/recovery/callbacks, the shared stratified controller, split consumers, configuration schema, and queue/dashboard UI. Keep scientific continuation logic separate from worker transport and control-plane persistence.

Resolve the relevant existing test failures as touched, especially legacy split migration, unlabeled pretraining, and GPU scope tests. Track unrelated baseline failures explicitly; require a green focused gate for each slice and restore a green repository gate before broad rollout.

## Rollout

Use additive database migrations and capability negotiation for `work_unit_v2`, directive protocol version, checkpoint schema, and interruption boundaries. Existing in-flight V1 jobs retain their original contract; do not rewrite active work. New resumable jobs require upgraded workers and cannot fall back silently. Drain/restart workers for supervisor changes. Retain V1 readers for historical artifacts.

Enable in stages: integrity checks, new-run split policy, worker control, epoch/cycle scheduling, then step scheduling. Measure heartbeat delay, command acknowledgment/application delay, checkpoint duration/bytes, replayed work, artifact verification cost, and worker utilization. Establish checkpoint retention and disk-pressure handling before enabling frequent durable uploads.

The first milestone delivers integrity enforcement, trustworthy split previews, and reliable worker stop/control. The second delivers durable epoch/cycle scheduling. Step-level preemption follows demonstrated continuation correctness.
