# ADR 0004: Worker roles and lifecycle providers

## Status

Accepted for Phases 3 and 4.

## Decision

`oracle-worker` is the supported executable for compute work. It is an
outbound-only pull client with fixed `train` and `package` executors;
`oracle-serve` and role-based HTTP worker modes are retired.

The Orchestrator owns worker-pool admission, leases, and typed lifecycle
requests. It delegates privileged process creation to a named lifecycle
provider. The first provider is `PullWorkerProcessProvider`; future Docker,
systemd, Kubernetes, and externally managed providers use the same narrow
`PullWorkerSpec` interface.

## Safety constraints

- Web/API clients cannot provide arbitrary commands or shell snippets.
- The local provider receives an already-built argument vector and never uses
  a shell.
- Remote machine control is not implemented as SSH execution. Remote workers
  self-register or are controlled by an approved infrastructure provider.
- Stopping a worker is cooperative first, then bounded termination; the
  orchestrator records the resulting state rather than assuming success.

## Migration

Worker registration, leases, materialization grants, and staged publication are
the sole execution path. Historical endpoint records may remain readable for
offline migration, but are not a supported runtime API.
