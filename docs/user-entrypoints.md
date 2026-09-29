# Using Oracle Builder

For normal use, the **Orchestrator is the only Oracle Builder entry point**.
Choose the interface that fits the user, but all three use the same durable
control plane:

| Interface | Use it for | Talks to |
| --- | --- | --- |
| Web GUI | Dataset/model transfer, definition authoring, queueing, evidence, and operations | Orchestrator only, through its same-origin proxy |
| `oracle` CLI | Scripts, uploads, catalog retrieval, queueing, and operational automation | Orchestrator API only |
| Orchestrator HTTP API | Programmatic retrieval and controlled mutations | Orchestrator only |

The Orchestrator owns identities, authorization, immutable input
materialization, worker-pool admission, leases, artifact publication, catalog
state, and replica recovery. `oracle-worker` is deliberately not a user-facing
server: it is a stateless, outbound pull process that receives sealed work and
returns staged output. Users and browser clients do not address workers,
choose worker output paths, or access worker scratch storage.

## Supported execution

Supported end-to-end execution is a versioned model definition plus a frozen
SQLite dataset queued as a `train` WorkUnit, and batch inference over a sealed
model artifact plus a frozen SQLite dataset queued as an `infer` WorkUnit. A
worker executes either under an Orchestrator lease; the Orchestrator validates
and publishes the resulting immutable artifact.

The old `oracle-serve` push protocol, resident HTTP inference service,
path-based model import, and remote-job refresh API are retired. They must not
be used by new scripts or deployments.

## Inference

Batch inference is supported through the Orchestrator API, `oracle inference`
CLI commands, and the **Run inference** action on a sealed model in the Web
GUI. Select a sealed model, frozen dataset, split, and admitted worker pool;
the resulting evidence artifact is immutable, downloadable, and replicated
like a training artifact. Do not send inference requests to a worker or revive
`oracle-serve`. Local inference libraries remain useful for controlled
developer tooling, but are not a deployment API.

## Security and retrieval

In token-configured deployments, anonymous, rate-limited `GET` requests are intentionally available for
catalog, artifact, and dataset retrieval scripts. Uploads, queueing,
deployments, replica operations, and every other mutation require an
authorized role token. The Web GUI keeps its operator token server-side; it
does not expose it to browsers. The explicitly unauthenticated loopback
development launcher does not apply this production read limiter.
