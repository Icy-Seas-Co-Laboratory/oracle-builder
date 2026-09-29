# Oracle WorkUnit V2

## Current implementation limits

V2 supports ordered epoch/cycle units and explicitly advertised ordered CPU
batch steps. Each unit starts a supervised process; input cache reuse is
implemented, while warm model/GPU residency is deferred. Checkpoint resume is
scientifically continuous for supported workflows, not a bitwise-RNG promise
across devices. A compatible worker may migrate a run after fixed-batch
revalidation; it does not silently recalibrate the sealed batch size.

`WorkUnit V2` describes one immutable, bounded advancement of a logical run.
It replaces neither the existing V1 wire contract nor its historical records.
V1 remains the contract for one whole worker attempt while the scheduler is
migrated. V2 separates the replayable scientific instruction from temporary
authority to execute it.

## Unit descriptor

```json
{
  "schema": {"name": "oracle_work_unit", "version": 2},
  "run_id": "UUID",
  "work_unit_id": "UUID",
  "sequence": 3,
  "phase": "train",
  "action": "train",
  "predecessor_work_unit_id": "UUID or null",
  "inputs": {
    "dataset": {"kind": "dataset", "artifact_id": "...", "revision": "...", "fingerprint_sha256": "..."},
    "split_manifest": {"kind": "split_manifest", "artifact_id": "...", "fingerprint_sha256": "..."},
    "checkpoint": {"kind": "checkpoint", "artifact_id": "...", "fingerprint_sha256": "..."}
  },
  "configuration": {"kind": "configuration", "artifact_id": "...", "fingerprint_sha256": "..."},
  "resources": {"gpu_count": 1},
  "parameters": {"epochs": 1},
  "start_cursor": {"epoch": 2, "global_step": 200},
  "stop_boundary": {"kind": "epoch", "epoch": 3},
  "output_contract": {"checkpoint": {"required": true}},
  "compatibility": {"framework": "tensorflow", "minimum_worker_protocol": 2}
}
```

The allowed phases are `verify`, `initialize`, `train`, `evaluate`, `finalize`,
and `infer`. All input and configuration references require a
`fingerprint_sha256`. In V2 that field is the pinned byte/content identity
required by the referenced artifact contract; it is never inferred from a
worker-local path.

The canonical descriptor is strict JSON: object keys are strings and NaN,
Infinity, functions, paths, and arbitrary Python values are rejected. Its
SHA-256 is the stable identity of the unit's immutable inputs and boundaries.
The parser rejects missing or unknown top-level fields so a typo cannot change
execution semantics silently.

`action` is a derived, sealed compatibility field: `infer` maps to the
existing `infer` worker capability and all other V2 phases map to `train`
during the transition to phase-native admission. Parsers reject an action that
does not match its phase; consumers therefore use the exact sealed bytes.

## Execution envelope

An attempt attaches an independent envelope:

```json
{
  "schema": {"name": "oracle_work_unit_execution", "version": 1},
  "work_unit_id": "UUID",
  "work_unit_sha256": "...",
  "attempt_id": "UUID",
  "generation": 4,
  "worker_id": "worker-17",
  "worker_boot_id": "UUID",
  "lease_id": "UUID",
  "lease_expires_at": "2026-09-29T12:00:00+00:00"
}
```

The envelope fences retries: the same V2 unit can receive a new attempt ID and
generation after a failure, while an older attempt cannot commit a result once
its lease/generation is superseded. `work_unit_sha256` binds the authority to
the exact sealed descriptor.

Lease bearer tokens, staging paths, output destinations, and worker-local
paths are excluded. They are transient transport/control-plane state and must
not be stored in a unit or envelope. The worker validates the received unit
digest and current lease authority before accepting output.

Use `parse_work_unit` when a consumer must accept both V1 and V2. New V2-only
code should use `WorkUnitV2.from_dict`; it must not translate an unpinned V1
reference into a V2 fingerprint.
