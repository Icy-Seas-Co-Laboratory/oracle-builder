# Oracle Builder web GUI

SvelteKit frontend for the Oracle Builder Orchestrator. It uses a same-origin
server-side proxy, so browsers never need direct access to the orchestration
service address.

```bash
cp .env.example .env
npm install
npm run dev
```

Set `ORCHESTRATOR_URL` to the running `oracle-orchestrator` backend. The GUI
supports asset upload and registration, allow-listed artifact scans, typed
training experiments, external-model imports, dispatch preflight, and the full
compute-to-catalog job lifecycle.

The route is a thin workspace controller. Feature state and actions are grouped
in `src/lib/AssetsView.svelte`, `ExperimentWizard.svelte`, and
`RunQueue.svelte`; `ExperimentResults.svelte` provides contract-backed result
review and persisted model comparisons. Shared operational and empty states are separate components.
This keeps polling and shared catalog data centralized without introducing a
global client-side store.

`ArtifactEvidence.svelte` adds read-only confusion matrices, class and sample
detail, segmentation evidence, and a gallery for sealed figures, overlays, and
activation or saliency products. The application shell follows the scientific
study workflow: **Prepare → Design → Run → Review**.
