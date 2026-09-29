# Oracle worker

`oracle-worker` is the disposable execution process for Oracle Builder. It has
no HTTP execution API, catalog, queue, artifact store, or output directory of
its own. It registers with an Orchestrator worker pool, pulls sealed work,
materializes only granted artifacts into private scratch space, and publishes
staged results back to the Orchestrator.

The supported invocation is the pull-worker command itself. Historical
`oracle-serve`, subcommands, and role-based HTTP worker modes are not part of
this release.

Workers heartbeat while their supervisor owns directives and leases. New
resumable workers should advertise the matching V2 control capability; inference
shard and merge units additionally require `inference_shards_v1`. Existing V1
workers should be drained and upgraded, never rewritten while active.

## Start a training worker

An operator creates a pool through the Orchestrator API or `oracle worker
pool-create`. The pool creation response returns a join token once. Store that
token in deployment secret management, not in a WorkUnit or browser.

```bash
# Read the one-time pool token from your deployment secret store; do not put it
# in shell history or a process argument list.
export ORACLE_BUILDER_WORKER_REGISTRATION_TOKEN="$(< /run/secrets/oracle-pool-token)"

oracle-worker \
  --orchestrator-url https://orchestrator.example \
  --pool-id POOL_ID \
  --name gpu-01 \
  --credentials-file /var/lib/oracle-worker/identity.json \
  --scratch-root /var/lib/oracle-worker/scratch \
  --executor train \
  --capability cpu_capacity=16
```

On its first start, the worker exchanges the join token for a server-issued
worker credential and stores it in `--credentials-file` with owner-only file
permissions. Later starts reuse that credential. Delete that file only when
intentionally replacing the worker identity.

`--executor train` is a fixed adapter for the versioned training contract. It
calls the typed `TrainingRequest` library boundary directly with a materialized
frozen dataset and resolved TOML configuration. It never mutates `sys.argv`,
constructs a command from a WorkUnit, or accepts an output path from the
control plane. `--executor package` is a deterministic smoke-test executor that
only publishes a manifest. A worker advertises exactly the actions enabled by
its fixed `--executor` options.

## Runtime guarantees

- A worker receives a lease only for an allowed action and compatible declared
  resources.
- Artifact download grants are one-use and bound to the worker and lease.
- The runtime acknowledges before execution, renews its lease while executing,
  records idempotent progress events, and uploads only a staging archive.
- Output publication is a separate, resumable phase. The worker writes its
  archive to private disk, sends bounded 8 MiB parts, and the Orchestrator
  records their digests before it accepts finalization. Neither side buffers a
  multi-gigabyte archive in memory.
- The Orchestrator validates, publishes, scans, and indexes a completed
  training run; workers never select a final output path.
- Scratch is private and disposable. Long-lived datasets, configurations,
  work-unit records, leases, and artifacts remain under Orchestrator control.

Each lease is also a durable execution attempt. Completed action failures are
scientific failures and are terminal. The control plane permits exactly one
automatic retry only after an infrastructure loss (lease expiry or release
without a sealed result); a second loss requires operator review. Workers
check for durable cancellation requests between materialization, execution,
and publication, and report a terminal cancelled attempt without publishing
output. Liveness reconciliation marks silent workers offline and uses that
same bounded recovery policy for their active lease.

## Recovering a completed output publication

If execution succeeds but publication cannot complete, the worker retains a
private recovery directory below `--scratch-root/recovery`. It contains the
archive and owner-only metadata, but never a credential or a host path. The
job must not be rerun solely for this condition. Once connectivity and the
lease credential are available, resume the transfer and finish the existing
lease:

```bash
oracle-worker \
  --orchestrator-url https://orchestrator.example \
  --credentials-file /var/lib/oracle-worker/identity.json \
  --recover-output /var/lib/oracle-worker/scratch/recovery/RECOVERY_ID
```

The registered worker identity receives a fresh short-lived recovery token for
its own pending lease; no raw lease token is kept in recovery metadata. The
command queries the durable upload session, skips already accepted parts,
verifies the final archive digest, and only then completes the lease. It
removes recovery files only after successful publication. Use this for a
publication interruption; a failed scientific execution remains a distinct,
terminal failure.

`--once` performs at most one polling/execution cycle, and `--max-jobs N`
makes a bounded batch worker. These are useful for CI, containers, and managed
platform jobs. A normal deployment runs continuously under the platform's
process supervisor. The local pull-worker lifecycle provider can start this
fixed process shape from an operator-owned deployment declaration; remote
providers should translate that declaration to Docker, VM, or Kubernetes
resources without adding SSH or arbitrary shell access to the Orchestrator.

## Managed deployments

The Orchestrator can own the desired lifecycle of fixed pull-worker profiles.
Profiles are an operator-local JSON file passed to
`oracle-orchestrator --worker-deployment-profiles FILE` (or the
`ORACLE_WORKER_DEPLOYMENT_PROFILES` environment variable). They are never sent
from the browser: the deployment API records only `profile_id`, pool, name,
and desired state.

```json
{
  "profiles": [{
    "profile_id": "demo-local-train",
    "provider": "local-process",
    "orchestrator_url": "http://127.0.0.1:8110",
    "credentials_root": "/srv/oracle/worker-identities",
    "scratch_root": "/srv/oracle/worker-scratch",
    "registration_token_file": "/run/secrets/oracle-pool-token",
    "executors": ["train"],
    "capabilities": {"cpu_capacity": "4"}
  }]
}
```

The local provider starts only the fixed `oracle-worker` vector described by
this profile. The raw token is absent from SQLite, CLI arguments, and the
deployment API. Start it with `oracle deployment create --name demo-local-1
--worker-pool POOL_ID --profile demo-local-train`, then `oracle deployment
start DEPLOYMENT_ID`. Keep the token file owner-readable only and remove it
after initial registration. See the [non-container demo deployment guide](../deploy/demo/README.md)
for persistent paths, TLS, backup, and recovery steps.

For Docker, use an operator-owned profile with `provider: "docker"`, a fixed
image, private Docker network, absolute host roots for credentials/scratch,
and a read-only bootstrap-token mount. The Orchestrator's Docker provider may
then create only the fixed worker image and argument vector selected by that
profile. The complete Compose configuration and its Docker-socket tradeoff are
documented in the [Docker demo deployment guide](../deploy/docker/README.md).
# Supervision and upgrades

Pull workers send routine heartbeats while a supervisor controls model execution.
Pause and yield stop at the next supported checkpoint: epoch/cycle boundaries
initially, and ordered CPU batch steps only where the work unit advertises that
support. `stop_now` may not create a new checkpoint. Drain and upgrade existing
V1 workers before enabling resumable work; active V1 jobs keep their original
contract and are never forcibly rewritten.
