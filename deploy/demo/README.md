# Non-container demo deployment

This is a deliberately small, single-host deployment shape: Caddy terminates
TLS, the Orchestrator and Web GUI listen on loopback, and an
Orchestrator-managed `local-process` worker runs on the same host. Docker is
not used. It is suitable for a demo, not a multi-host high-availability setup.

For the equivalent Compose-based topology, including Orchestrator-managed
Docker workers, see the [Docker demo deployment](../docker/README.md).

## 1. Prepare the host and secrets

Create an `oracle-builder` service account and these private, persistent
directories (all owned by that account):

```text
/var/lib/oracle-builder/control             # SQLite control database
/var/lib/oracle-builder/artifacts           # uploads, staging, local replicas
/var/lib/oracle-builder/runs                # sealed model-run folders
/var/lib/oracle-builder/datasets            # frozen SQLite datasets
/var/lib/oracle-builder/worker-identities   # worker credentials, 0700
/var/lib/oracle-builder/worker-scratch      # disposable worker files, 0700
/var/lib/oracle-builder/workspace           # approved recipes/workspace
/var/log/oracle-builder
/run/oracle-builder                         # one-time pool join token, tmpfs
```

Install the project with `api` and `storage-s3` extras on the host. Copy
`oracle-builder.env.example` to `/etc/oracle-builder/orchestrator.env`, copy
`worker-deployments.local-process.json.example` to
`/etc/oracle-builder/worker-deployments.json`, and replace every placeholder.
Both files must be owner-readable only. Use an AWS service identity when
possible; otherwise store credentials in the separate 0600 file named by
`AWS_SHARED_CREDENTIALS_FILE`.

Generate a high-entropy operator token, retain the raw value in the operator
secret store, and place **only its SHA-256 digest** in
`ORACLE_ORCHESTRATOR_ROLE_TOKENS_SHA256`. For example:

```bash
TOKEN="$(openssl rand -hex 32)"
printf '%s' "$TOKEN" | shasum -a 256
```

Set the JSON value as `{"operator":"sha256:THE_DIGEST"}`. Never place the
raw token in the Orchestrator environment file, command line, browser, or
worker profile. The isolated Web GUI server environment below is the one
exception: it needs the raw token to proxy browser mutations and must remain
restricted to the service account.

The profile's `registration_token_file` does not exist until a worker pool is
created. Write its one-time join token to `/run/oracle-builder/demo-pool-token`
with `install -m 0600 -o oracle-builder -g oracle-builder`; remove it after
the worker first registers. The service-issued identity remains under
`worker-identities/` for later restarts.

## 2. Start the control plane and TLS proxy

Install `systemd/oracle-orchestrator.service` as
`/etc/systemd/system/oracle-orchestrator.service`, adapt its project/venv
paths, then start it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now oracle-orchestrator
```

Use `Caddyfile.example` as the Caddy site configuration. It deliberately
routes public `/v1/*` and `/health/*` requests directly to the Orchestrator so
anonymous, rate-limited retrieval scripts can use the documented API. The Web
GUI is proxied separately; its `/api/*` server-side route adds the operator
token for mutations without exposing it to browsers.

The sample Caddyfile has no distributed rate limit because stock Caddy does
not provide one. Put a CDN/WAF or another edge rate-limit mechanism in front
of it, keyed by the original client address, and configure trusted proxy
headers correctly. Do not expose port 8110 or 5111 publicly. Set the body and
read/write timeouts to cover your largest intended upload; the supplied values
match the default 10 GiB application upload limit.

The included systemd service starts the Orchestrator only. The present Web GUI
uses SvelteKit's development server (`npm run dev`), so it is acceptable for a
time-bounded supervised demo but is **not** yet a production web-server
artifact. Start it under a separate restricted service with
`ORCHESTRATOR_URL=http://127.0.0.1:8110` and its server-side operator token.
`systemd/oracle-webgui-demo.service` and `webgui.env.example` provide that
demo-only service shape; the environment file contains the raw token and must
remain 0600. Moving the GUI to a production SvelteKit adapter is a required
follow-up before anything beyond the demo.

Run safe checks before starting:

```bash
ORACLE_DEMO_DATA_ROOT=/var/lib/oracle-builder \
ORACLE_WORKER_DEPLOYMENT_PROFILES=/etc/oracle-builder/worker-deployments.json \
scripts/demo_preflight.sh --strict
```

## 3. Start one local managed worker and smoke-test training and inference

Use a narrow `train`/`infer` pool, then save the single-use response token in the
profile's token file:

```bash
export ORACLE_ORCHESTRATOR_URL=https://demo.example.com
export ORACLE_ORCHESTRATOR_TOKEN='raw-token-from-secret-store'
oracle worker pool-create --name demo-compute --action train --action infer --max-workers 1
# Save the returned registration_token to /run/oracle-builder/demo-pool-token
oracle deployment create --name demo-local-1 --worker-pool POOL_ID --profile demo-local-train
oracle deployment start DEPLOYMENT_ID
oracle worker list
oracle deployment show DEPLOYMENT_ID
```

Remove the token file after the worker is `idle`; an initial registration
failure may safely be retried while it remains. Upload a known small *frozen*
SQLite dataset in the Web GUI, create a CPU-safe model definition, then queue
it to `POOL_ID`. Confirm the lifecycle in this order: queued lease, worker
acknowledgement, progress events, staged output, indexed artifact, and
`replicated` status. Then create one inference run from the sealed model and
the same frozen dataset; use `oracle inference submit --name smoke-inference
--model ARTIFACT_ID --dataset DATASET_ID --worker-pool POOL_ID --start --wait`
and download the resulting sealed predictions archive. Use the GUI Operations page or `oracle worker leases`,
`oracle job list`, and `oracle artifact replicas`.

## 4. Backup, retention, and recovery drill

The local folders are the primary online copy. Take a nightly filesystem or
volume snapshot of `control/`, `artifacts/`, `runs/`, and `datasets/`; include
the project deployment configuration, but exclude worker scratch. Retain at
least seven daily snapshots for the demo period. Verify that SQLite snapshots
are filesystem-consistent (or stop the service briefly / use a storage-level
snapshot protocol); do not copy a live SQLite database with a naive file copy.

S3/MinIO holds immutable artifact replicas, not the control database. Apply a
bucket lifecycle that retains current demo objects for the whole demo window,
keeps noncurrent versions for at least 30 days if bucket versioning is enabled,
and aborts incomplete multipart uploads after seven days. Do not expire the
only remote copy before the local backup retention window.

Recovery drill, using a completed training artifact:

1. Record its `ArtifactRef` and run `oracle artifact replica-verify REF`.
2. Wait for the returned operation to complete and confirm it remains
   `replicated` in `oracle artifact replicas`.
3. On a separate clean restore host/root, configure the same S3 replica and
   an empty canonical artifact directory.
4. Run `oracle artifact replica-restore REF`; wait for completion, then
   validate/download the restored artifact.
5. Repeat the restore without clearing the target and confirm it refuses to
   overwrite the canonical local artifact.

Record the operation IDs, checksums, and elapsed time in the demo runbook.
This proves remote artifact recovery; restoring the Orchestrator database is a
separate volume-backup procedure.

## Demo gate

The demo is ready only when TLS, token-protected writes, worker heartbeat, a
small completed training run, a replicated artifact, and the clean-root
restore have all been observed from a fresh service restart. See
`docs/operations-and-troubleshooting.md` for general troubleshooting.
