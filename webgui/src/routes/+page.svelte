<script lang="ts">
	import { onMount } from 'svelte';
	import { api, type RecordValue } from '$lib/api';
	import AssetsView from '$lib/AssetsView.svelte';
	import ExperimentWizard from '$lib/ExperimentWizard.svelte';
	import ExperimentResults from '$lib/ExperimentResults.svelte';
	import RunQueue from '$lib/RunQueue.svelte';
	import ServiceStatus from '$lib/ServiceStatus.svelte';

	type Section = 'overview' | 'prepare' | 'design' | 'run' | 'review';
	let section: Section = 'overview';
	let datasets: RecordValue[] = [];
	let artifacts: RecordValue[] = [];
	let recipes: RecordValue[] = [];
	let experiments: RecordValue[] = [];
	let specifications: RecordValue[] = [];
	let jobs: RecordValue[] = [];
	let computeEndpoints: RecordValue[] = [];
	let systemHealth: RecordValue | null = null;
	let selectedDatasetId = '';
	let notice = '';
	let failure = '';
	let loading = true;

	const activeStatuses = ['queued', 'running', 'submitted', 'dispatching', 'validating'];
	const isActiveJob = (job: RecordValue) => activeStatuses.includes(String(job.status));
	const serviceReady = () => systemHealth?.database === 'ready' && computeEndpoints.some((endpoint) => endpoint.status === 'ready');
	const indexedJobs = () => jobs.filter((job) => job.status === 'indexed').length;
	const plannedRuns = () => specifications.filter((item) => item.status === 'planned').length;
	const failedRuns = () => jobs.filter((job) => ['failed', 'dispatch_failed', 'artifact_invalid'].includes(String(job.status))).length;
	const workflow = [
		{ id: 'prepare', step: '01', label: 'Prepare', description: 'Datasets, recipes, and models' },
		{ id: 'design', step: '02', label: 'Design', description: 'Define a controlled study' },
		{ id: 'run', step: '03', label: 'Run', description: 'Review and dispatch compute' },
		{ id: 'review', step: '04', label: 'Review', description: 'Inspect and compare evidence' }
	] as const;
	const stageComplete = (id: string) => id === 'prepare' ? datasets.length > 0 && recipes.length > 0 : id === 'design' ? experiments.length > 0 : id === 'run' ? jobs.length > 0 : indexedJobs() > 0;
	const recommendedSection = (): Section => !datasets.length || !recipes.length ? 'prepare' : !experiments.length ? 'design' : plannedRuns() || jobs.some(isActiveJob) ? 'run' : 'review';
	const titles: Record<Section, [string, string]> = {
		overview: ['Study workspace', 'Build models through traceable experiments.'],
		prepare: ['Prepare · 01', 'Organize reusable inputs.'],
		design: ['Design · 02', 'Define a controlled experiment.'],
		run: ['Run · 03', 'Move approved work through compute.'],
		review: ['Review · 04', 'Compare results and inspect evidence.']
	};

	async function refresh() {
		loading = true; failure = '';
		try {
			const [datasetResult, artifactResult, recipeResult, experimentResult, specificationResult, jobResult, endpointResult, healthResult] = await Promise.all([
				api.datasets(), api.artifacts(), api.recipes(), api.experiments(), api.specifications(), api.jobs(true), api.computeEndpoints(true), api.health()
			]);
			datasets = datasetResult.datasets; artifacts = artifactResult.artifacts; recipes = recipeResult.recipes; experiments = experimentResult.experiments;
			specifications = specificationResult.specifications; jobs = jobResult.jobs; computeEndpoints = endpointResult.endpoints; systemHealth = healthResult;
			if (!selectedDatasetId && datasets.length) selectedDatasetId = String(datasets[0].dataset_id);
		} catch (error) { failure = error instanceof Error ? error.message : 'Could not reach the Orchestrator.'; }
		finally { loading = false; }
	}

	async function changed(message: string) { notice = message; failure = ''; await refresh(); }
	function failed(message: string) { failure = message; notice = ''; }
	function useDataset(datasetId: string) { selectedDatasetId = datasetId; section = 'design'; }
	async function experimentCreated(message: string) { section = 'run'; await changed(message); }
	async function modelImportCreated(message: string) { section = 'run'; await changed(message); }
	function viewArtifact() { section = 'review'; notice = 'The indexed model is ready for evidence review.'; }

	async function refreshActiveJobs() {
		if (!jobs.some(isActiveJob)) return;
		try {
			const [jobResult, artifactResult, specificationResult, endpointResult] = await Promise.all([api.jobs(true), api.artifacts(), api.specifications(), api.computeEndpoints(true)]);
			jobs = jobResult.jobs; artifacts = artifactResult.artifacts; specifications = specificationResult.specifications; computeEndpoints = endpointResult.endpoints;
		} catch { /* Keep durable state visible while services restart. */ }
	}

	onMount(() => { refresh(); const timer = window.setInterval(refreshActiveJobs, 5000); return () => window.clearInterval(timer); });
</script>

<svelte:head><title>Oracle Builder · Pelagia</title><meta name="description" content="Design reproducible model experiments and inspect durable evidence" /></svelte:head>

<div class="app-shell">
	<aside class="app-sidebar">
		<button class="brand" on:click={() => section = 'overview'} aria-label="Oracle Builder home"><span class="brand-mark">OB</span><span><strong>Oracle Builder</strong><small>Pelagia model laboratory</small></span></button>
		<div class="sidebar-group"><p class="nav-label">MODEL WORKFLOW</p><nav class="workflow-nav">{#each workflow as item}<button class:active={section === item.id} on:click={() => section = item.id}><span class="nav-step">{item.step}</span><span><strong>{item.label}</strong><small>{item.description}</small></span>{#if stageComplete(item.id)}<i aria-label="Stage has activity">✓</i>{/if}</button>{/each}</nav></div>
		<div class="sidebar-context"><p class="nav-label">WORKSPACE</p><dl><div><dt>Studies</dt><dd>{experiments.length}</dd></div><div><dt>Active runs</dt><dd>{jobs.filter(isActiveJob).length}</dd></div><div><dt>Indexed results</dt><dd>{indexedJobs()}</dd></div></dl></div>
		<ServiceStatus ready={serviceReady()} {loading} />
	</aside>

	<div class="app-frame">
		<header class="topbar"><div><p class="eyebrow">{titles[section][0]}</p><h1>{titles[section][1]}</h1></div><div class="topbar-actions"><span class:online={serviceReady()} class="system-pill"><i></i>{serviceReady() ? 'System ready' : 'Compute unavailable'}</span><button class="quiet-button" on:click={refresh} disabled={loading}>{loading ? 'Refreshing…' : 'Refresh'}</button></div></header>
		<main class="workspace">
			{#if failure}<div class="banner error" role="alert"><strong>Action needed</strong><span>{failure}</span><button aria-label="Dismiss" on:click={() => failure = ''}>×</button></div>{/if}
			{#if notice}<div class="banner notice" role="status"><strong>Updated</strong><span>{notice}</span><button aria-label="Dismiss" on:click={() => notice = ''}>×</button></div>{/if}

			{#if section === 'overview'}
				<section class="welcome-panel"><div class="welcome-copy"><p class="eyebrow">REPRODUCIBLE MODEL DEVELOPMENT</p><h2>From frozen data to reviewed evidence.</h2><p>Oracle Builder keeps the dataset revision, configuration, execution history, evaluation protocol, and portable model artifact connected through one study workflow.</p><div class="hero-actions"><button on:click={() => section = recommendedSection()}>Continue workflow <span>→</span></button><button class="secondary" on:click={() => section = 'design'} disabled={!datasets.length || !recipes.length}>New experiment</button></div></div><div class="workflow-map">{#each workflow as item, index}<button class:complete={stageComplete(item.id)} on:click={() => section = item.id}><span>{item.step}</span><div><strong>{item.label}</strong><small>{item.description}</small></div><i>{stageComplete(item.id) ? '✓' : index < 3 ? '→' : '·'}</i></button>{/each}</div></section>
				<section class="stat-grid"><article><span>Frozen datasets</span><strong>{datasets.length}</strong><small>{recipes.length} validated recipes</small></article><article><span>Experiment studies</span><strong>{experiments.length}</strong><small>{plannedRuns()} runs await dispatch</small></article><article><span>Compute activity</span><strong>{jobs.filter(isActiveJob).length}</strong><small>{computeEndpoints.filter((item) => item.status === 'ready').length} ready endpoints</small></article><article class:attention={failedRuns() > 0}><span>Reviewed artifacts</span><strong>{artifacts.length}</strong><small>{failedRuns() ? `${failedRuns()} runs need attention` : 'No execution failures'}</small></article></section>
				<section class="panel next-action"><div><p class="eyebrow">RECOMMENDED NEXT STEP</p><h2>{recommendedSection() === 'prepare' ? 'Prepare the first reusable inputs' : recommendedSection() === 'design' ? 'Turn prepared inputs into a study' : recommendedSection() === 'run' ? 'Review and dispatch planned work' : 'Inspect the evidence behind your results'}</h2><p>{recommendedSection() === 'prepare' ? 'Register a frozen dataset and validate at least one training recipe.' : recommendedSection() === 'design' ? 'Choose a dataset, recipes, and explicit seeds for a reproducible experiment.' : recommendedSection() === 'run' ? 'Run preflight against Oracle Serve before releasing each immutable request.' : 'Compare compatible candidates, then inspect class, sample, and visual evidence.'}</p></div><button on:click={() => section = recommendedSection()}>Open {recommendedSection()} <span>→</span></button></section>
			{:else if section === 'prepare'}
				<AssetsView {datasets} {artifacts} {recipes} bind:selectedDatasetId onchanged={changed} onfailure={failed} onuseDataset={useDataset} onmodelcreated={modelImportCreated} />
			{:else if section === 'design'}
				<ExperimentWizard {datasets} {recipes} bind:selectedDatasetId oncreated={experimentCreated} onfailure={failed} ongoassets={() => section = 'prepare'} />
			{:else if section === 'review'}
				<ExperimentResults {experiments} {jobs} onchanged={changed} onfailure={failed} oncreateexperiment={() => section = 'design'} />
			{:else}
				<RunQueue {specifications} {jobs} {computeEndpoints} onchanged={changed} onfailure={failed} oncreateexperiment={() => section = 'design'} onviewartifact={viewArtifact} />
			{/if}
		</main>
	</div>
</div>
