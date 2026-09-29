# ADR 0002: Control-plane ownership and portable artifacts

## Status

Accepted for the compatibility-first migration.

## Decision

The Oracle Orchestrator is the durable control-plane authority. It owns job
history, immutable work-unit construction, artifact publication, and the
catalog. Compute processes are replaceable executors: they receive a sealed
work unit, use only job-scoped scratch/staging locations, and never publish or
catalog a result directly.

Durable scientific products remain self-describing folders and SQLite files.
The default `LocalArtifactStore` preserves the artifact-as-folder approach and
can register the existing `datasets/` and `runs/` layouts in place. A future
object-store implementation must preserve the same manifest, fingerprint, and
seal semantics rather than making a database the canonical location for model
bytes or datasets.

## Compatibility and migration

Existing path-based compute requests remain a local execution adapter during
the migration. The durable work-unit contract contains artifact references,
not host paths. The adapter is deliberately explicit and temporary: remote
workers must eventually materialize references through an artifact store.

Existing sealed runs and frozen datasets are registered in place. No bulk
move, format conversion, or mutable rewrite is part of this decision.

## Consequences

- Workers can later run on a different host, VM, or container.
- The catalog can be rebuilt by scanning sealed manifests.
- A worker cannot make a candidate result authoritative.
- Checkpoints and staged outputs require an explicit durable-storage policy;
  ephemeral scratch alone is not sufficient for recovery.
