# Docker demo deployment

This is the containerized counterpart to the single-host demo. It runs Caddy,
the production-built Web GUI, and the Orchestrator in Compose. The
Orchestrator can launch fixed-profile `oracle-worker` containers through the
host Docker socket after an operator creates a pool and deployment.

It is a single-host demo shape, not an HA or multi-tenant platform. The
durable folder tree remains host-owned and human-readable; Docker containers
are replaceable.

## What runs where

| Service | Responsibility | Persistent state |
| --- | --- | --- |
| Caddy | TLS, public routing | named Caddy certificate/config volumes |
| Web GUI | UI and server-side authenticated `/api` proxy | none |
| Orchestrator | API, SQLite, artifacts, leases, catalogs | `${ORACLE_DOCKER_DATA_ROOT}` |
| `oracle-worker` | pull-only training/inference compute | disposable scratch under the data root |

The worker does not publish a port and no browser or CLI connects to it.
This Compose stack creates the `oracle-builder` network. Pelagia's Docker
stack joins it as an external network by default, so Pelagia containers can
reach the Orchestrator at `http://orchestrator:8110`. Start this stack first.
The Docker socket is mounted only into the Orchestrator because it owns the
fixed-profile worker lifecycle. If that authority is unacceptable, remove the
socket/profile mount and run workers from an external orchestrator instead.

## 1. Prepare the host

Use a Linux host with Docker Compose v2, enough disk for TensorFlow images and
model artifacts, and a real DNS name if public TLS is required. Choose an
absolute data root without spaces, for example `/srv/oracle-builder`:

```bash
cd deploy/docker
./prepare.sh /srv/oracle-builder
cp .env.example .env
chmod 600 .env
```

The script creates `control`, `artifacts`, `runs`, `datasets`, `workspace`,
`logs`, worker identities, worker scratch, and the one-time worker-bootstrap
directory under that root. It also produces
`runtime/worker-deployments.json`, whose host paths are deliberately not
stored in SQLite or exposed by the API.

Generate an operator token, retain the raw token only in `.env`, and put its
SHA-256 digest in the role map:

```bash
TOKEN="$(openssl rand -hex 32)"
printf '%s' "$TOKEN" | sha256sum
```

Set `ORCHESTRATOR_OPERATOR_TOKEN=$TOKEN` and replace
`REPLACE_WITH_SHA256` in `ORACLE_ORCHESTRATOR_ROLE_TOKENS_SHA256`. Never
commit `.env`, the generated runtime profile, or `worker-bootstrap/pool-token`.

For public TLS, set `ORACLE_PUBLIC_HOST` to the host's DNS name and allow
inbound TCP 80/443 (and UDP 443 when using HTTP/3). For a local-only demo,
leave it as `localhost`. Configure a CDN/WAF or edge rate limiter before
Internet exposure: the application limits anonymous reads per process but is
not a distributed edge limiter.

## 2. Build and start

Build the worker image explicitly: it is intentionally not a continuously
running Compose service because the Orchestrator creates the actual worker
container from the fixed profile.

```bash
docker compose --profile image build
docker compose up -d orchestrator webgui caddy
docker compose ps
docker compose logs -f orchestrator
```

The Web GUI is production-built with SvelteKit's Node adapter. Caddy exposes
only ports 80 and 443; Orchestrator and Web GUI ports remain inside the Docker
network. Check readiness through Caddy:

```bash
curl --fail https://YOUR_HOST/health/ready
```

Use `http://localhost/health/ready` or the locally trusted Caddy certificate
when testing the default local host.

## 3. Create and start an Orchestrator-managed worker

Use the raw operator token only in this shell or in a secret manager:

```bash
export ORACLE_ORCHESTRATOR_URL=https://YOUR_HOST
export ORACLE_ORCHESTRATOR_TOKEN='raw-token'
oracle worker pool-create --name docker-compute --action train --action infer --max-workers 1
```

Copy the returned `registration_token` to
`${ORACLE_DOCKER_DATA_ROOT}/worker-bootstrap/pool-token`, mode `0600`. Then
create and start the deployment:

```bash
oracle deployment create --name docker-cpu-1 --worker-pool POOL_ID --profile docker-cpu
oracle deployment start DEPLOYMENT_ID
oracle worker list
```

The Orchestrator asks Docker to create a container named
`oracle-worker-DEPLOYMENT_ID` on the private `oracle-builder` network. Its
join token is mounted read-only and is not in a Docker environment value,
command line, WorkUnit, or SQLite record. After the worker is `idle`, remove
the bootstrap token file. Docker retains the worker-issued credential under
`worker-identities/` for its next start.

Run the same small frozen-input training and inference smoke test described in
the [non-container demo guide](../demo/README.md). Inspect progress with the
Web GUI Operations page, `oracle worker leases`, and `docker logs
oracle-worker-DEPLOYMENT_ID`.

## Storage, replicas, and upgrades

Back up the entire host data root; it contains the SQLite database and the
primary artifact tree. Worker scratch is disposable, but retaining it during a
demo aids diagnosis. Caddy volumes contain certificate material and should be
backed up separately if TLS continuity matters.

Set the optional S3 variables in `.env` only after configuring the standard
AWS credential provider chain for the Orchestrator container. For Docker
secrets or workload identity, mount the credential source into the
Orchestrator service; never pass static access keys through a WorkUnit or the
Web GUI. The S3 replica supplements, rather than replaces, the host data-root
backup.

For an image/configuration upgrade: stop Compose, take a consistent backup of
the data root, build the new images, start the control-plane services, verify
`/health/ready`, then reconcile worker deployments and run one small job. When
Oracle Builder runs with Pelagia's Docker stack, use Pelagia's
`deploy/docker/update-dev.py` for updates from all three `main` branches. It
restarts idle Docker deployments from the rebuilt worker image and records the
deployed commits and image IDs.

## Guardrails

- Do not mount the Docker socket into Web GUI or worker containers.
- Do not expose ports 8110 or 3000 directly.
- Do not use the socket-mounted Orchestrator on a shared/untrusted Docker
  host; Docker-socket access is effectively host administration.
- Do not use the generated profile with a different data root. Re-run
  `prepare.sh` intentionally after moving persistent storage.
