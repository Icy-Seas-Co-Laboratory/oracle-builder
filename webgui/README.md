# Oracle Builder web GUI

SvelteKit frontend for the Oracle Builder Orchestrator. It uses a same-origin
server-side proxy, so browsers never need direct access to the orchestration
service address.

```bash
cp .env.example .env
npm install
npm run dev
```

Set `ORCHESTRATOR_URL` to the running `oracle-orchestrator` backend. Set
`ORCHESTRATOR_OPERATOR_TOKEN` when its mutation API is protected: SvelteKit
keeps this token server-side, strips browser-supplied authorization, and adds
it only when proxying to the control plane. Anonymous direct `GET` requests
remain available for lightweight retrieval scripts.

The GUI is one of three supported client surfaces—the others are `oracle` and
the Orchestrator HTTP API. All communicate only with the Orchestrator;
`oracle-worker` is not browser- or user-addressable.

The main workspace focuses on worker pools, sealed definition queueing,
durable leases, and published output references. Dataset/config/model uploads
use resumable 16 MiB chunks (a retry resumes the same selected file), and the
asset library can download frozen datasets or complete portable model-artifact
archives without connecting the browser to worker storage.

The Queue page separates **Add to queue → Verify selected → Start selected**.
Select one or more entries; the optional **Start training after successful
verification** checkbox is unchecked by default. Worker preflight checks the
configuration, dataset and model, and calibrates automatic batch sizes. Passing
entries wait for Start unless the checkbox was selected; failures remain
inspectable and can be verified again. Restart updated workers to advertise
verification support. Training uses the same worker instance and selected batch
size; a worker restart or capability change requires reverification.

The Workers page creates pool admission boundaries and shows fleet state. It
never connects to worker hosts or reads artifact storage. Pool join tokens are shown once after creation; use them with
`oracle-worker` outside the browser.

The root route is a thin workspace controller. Worker-fleet state is kept in
`WorkerFleetView.svelte`; the browser does not own scheduling or artifact
state.

Operational state is coordinated by one visibility-aware browser refresh loop:
it consumes the orchestrator's resumable event stream when available, then
falls back to polling every 5 seconds for active work, 20 seconds when idle,
and 60 seconds while hidden. This prevents the queue and workspace shell from
independently reconciling the same jobs.

For local development, run `scripts/start_oracle_stack.sh` from the repository
root. The GUI defaults to `http://127.0.0.1:5111` and the Orchestrator defaults
to port `8110`. Override bind ports with `ORACLE_WEBGUI_PORT` or
`ORACLE_ORCHESTRATOR_PORT`. Workers are deployed independently and pull work
from the Orchestrator; the browser never needs their addresses.

The stack keeps transient control-plane state under `.oracle-runtime/`, while
durable runs and training datasets use the repository's `runs/` and `datasets/`
directories. Each new Orchestrator process reconciles those directories: it
indexes sealed run artifacts and registers frozen Oracle SQLite datasets that
are absent from its runtime catalog. The training-set catalog accepts only
Oracle SQLite revisions (never arbitrary image folders), groups related
revisions into a selectable family, and provides bounded previews from the
SQLite image assets.

`ArtifactEvidence.svelte` adds read-only confusion matrices, class and sample
detail, segmentation evidence, and a gallery for sealed figures, overlays, and
activation or saliency products. The application shell follows the scientific
study workflow: **Prepare → Design → Run → Review**.

The Live Dashboard shares the workspace refresh coordinator, supports run
selection, batch/epoch charts, and metric, timing, fleet, queue, and publication
detail dialogs. Charts expose sampled values and individual point tooltips.
Updates can be paused; refresh failures retain the last snapshot with a visible
stale-data notice. Native dialogs support Escape and restore focus on close.

For resumable work units, the selected-run panel also shows the run/unit/attempt
cursor, latest committed checkpoint, and worker heartbeat. Its controls submit
durable `pause`, `yield`, `restart`, `resume`, or `stop now` requests; they show
the command's received/accepted/applied state and do not claim completion when
the request is merely delivered. Safe pause and yield appear only for jobs that
advertise an execution segment. Queue verification details include the frozen
split policy and class-by-split coverage report when available.
