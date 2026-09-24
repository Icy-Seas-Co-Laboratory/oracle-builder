# Oracle Builder web GUI

SvelteKit frontend for the Oracle Builder Orchestrator. It uses a same-origin
server-side proxy, so browsers never need direct access to the orchestration
service address.

```bash
cp .env.example .env
npm install
npm run dev
```

Set `ORCHESTRATOR_URL` to the running `oracle-orchestrator` backend. The main
workspace has five connected pages: **Model Runs** (catalog, tags, inspector),
**Comparison**, **Training Sets** (read-only source catalog and previews),
**Construction** (versioned V2 model drafts), and **Training** (immutable plan
creation). It also supports asset upload and registration, dispatch preflight,
and the full compute-to-catalog job lifecycle.

The root route is a thin workspace controller. Page-level feature state is
kept in focused components (`models/ModelRunsView.svelte`,
`models/ModelComparisonView.svelte`, `TrainingSetCatalogView.svelte`,
`ModelConstructionView.svelte`, and `TrainingStudio.svelte`) while shared
operational data is refreshed centrally without a global client-side store.

For local development, run `scripts/start_oracle_stack.sh` from the repository
root. The GUI defaults to `http://127.0.0.1:5111`; Oracle Serve and the
Orchestrator retain their defaults at ports `8100` and `8110`. The script checks
their health endpoints first and reuses either service when it is already
healthy. Override any bind port with `ORACLE_WEBGUI_PORT`, `ORACLE_SERVE_PORT`,
or `ORACLE_ORCHESTRATOR_PORT`.

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
