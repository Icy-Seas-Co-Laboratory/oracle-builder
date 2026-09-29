# Execution implementation and validation

Date: 2026-09-29. Changes are in the working tree; no deployment is implied.

## Delivered behavior

- New queue intake requires V2-capable verifiers. Verification freezes split
  assignments and batch size; training waits for Start unless Verify and start
  was selected. Legacy in-flight work keeps its existing contract.
- Logical runs advance through immutable training units and separate attempts.
  Epoch/cycle checkpoints commit before a successor becomes eligible. Replaying
  completion cannot create another successor. Finalization never repeats fit.
- Worker parents own independent heartbeats, long-poll directives, lease
  deadlines, and child process-group termination. Pause, yield, stop, restart,
  and resume have durable receipts. Stop/restart fences uncommitted output.
- Compatible replacement workers recheck capacity at the frozen training batch;
  migration cannot silently change batch size. Inputs, split identities,
  checkpoints, and output bytes are validated at consumption/publication.
- New ROI assignments prioritize deterministic per-class split coverage, with
  optional explicit group isolation. Imported partitions are retained and
  validated; infeasible coverage fails verification. Evaluation does not
  silently substitute validation data for missing test data.
- Inference uses ordered item shards (API default 1024), then a coverage-checked
  merge. Shard workers advertise their capability before admission.
- Dashboard and CLI expose directives, receipts, paused continuations, attempts,
  committed progress, and checkpoint identity. Metrics open detail dialogs.

## Validation evidence

- Real CPU run: verify, two separately published epoch units, finalize, index.
  Repeated completion returned the original receipt.
- Real HTTP worker: fresh supervised execution processes, authenticated artifact
  delivery, verified cache reuse, heartbeats, replacement-worker continuation,
  and final publication passed.
- Real HTTP directives: safe pause committed a checkpoint; resume released the
  continuation; restart retried the same work-unit digest; final output indexed.
- Ordered CPU steps: six bounded units advanced two epochs, then finalized.
  A regression compares uninterrupted training against four one-batch resumed
  units, including model weights, every optimizer variable, and iterations.
- Inference: three disjoint eight-item shards merged into one 24-item result.
- Browser check on isolated preview data: metric details, paused-run visibility,
  committed cursor/checkpoint, and Resume → Applied → Queued.
- Final repository gate: **959 Python tests passed** (202 seconds).
  Svelte checking reported **0 errors and 0 warnings**; production build passed.
  The final independent read-only integration review found no new blockers.
  `git diff --check` passed.

## Supported limits and remaining work

Epoch/cycle recovery preserves supported model, optimizer, and callback state;
it does not promise identical RNG progression or bitwise results across
hardware. Exact batch-cursor continuation is restricted to the opt-in ordered
CPU path: finite non-streaming data, no random augmentation, no shared-weight
stratification, and no unsupported epoch callback schedules. Self-supervised
queued segmentation fails verification; the explicit whole-phase workflow
remains available.

Each unit uses a fresh process. Warm GPU/model residency, adaptive wall-time
budgets, generalized GPU sampler/RNG continuation, and separate evaluation and
packaging work units remain future slices. Evaluation and packaging currently
share one retriable finalization unit. Cache eviction is bounded by entry count;
production byte quotas, checkpoint garbage collection, and load/latency
benchmarks remain operational follow-ups. No multi-host GPU campaign was run.

The independent [system review](system-review-2026-09-29.md) retains unrelated
security and operational recommendations. Restart/drain workers to load the new
supervisor and capabilities before scheduling newly queued V2 work.
