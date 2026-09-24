<script lang="ts">
	import { onMount } from 'svelte';
	import { api, type RecordValue } from '$lib/api';
	import ModelRunsView from '$lib/models/ModelRunsView.svelte';
	import ModelComparisonView from '$lib/models/ModelComparisonView.svelte';
	import TrainingSetCatalogView from '$lib/TrainingSetCatalogView.svelte';
	import ModelConstructionView from '$lib/ModelConstructionView.svelte';
	import TrainingStudio from '$lib/TrainingStudio.svelte';

	type Page = 'models' | 'comparison' | 'datasets' | 'construction' | 'training';
	let datasets: RecordValue[] = [];
	let recipes: RecordValue[] = [];
	let artifacts: RecordValue[] = [];
	let jobs: RecordValue[] = [];
	let computeEndpoints: RecordValue[] = [];
	let systemHealth: RecordValue | null = null;
	let activePage: Page = 'models';
	let comparisonIds: string[] = [];
	let trainingDraftId = '';
	let loading = true;
	let notice = '';
	let failure = '';

	const pageMeta: { id: Page; number: string; title: string; detail: string }[] = [
		{ id: 'models', number: '01', title: 'Model Runs', detail: 'Catalog & inspect' },
		{ id: 'comparison', number: '02', title: 'Comparison', detail: 'Compare evidence' },
		{ id: 'datasets', number: '03', title: 'Training Sets', detail: 'Browse sources' },
		{ id: 'construction', number: '04', title: 'Construction', detail: 'Compose a model' },
		{ id: 'training', number: '05', title: 'Training', detail: 'Plan execution' }
	];
	const activeStatuses = ['queued', 'running', 'submitted', 'dispatching', 'validating'];
	const activeJobs = () => jobs.filter((job) => activeStatuses.includes(String(job.status))).length;
	const healthEndpoints = () => systemHealth?.compute_endpoints && typeof systemHealth.compute_endpoints === 'object' ? systemHealth.compute_endpoints as RecordValue : null;
	const serviceReady = () => systemHealth?.database === 'ready' && (Number(healthEndpoints()?.ready ?? 0) > 0 || computeEndpoints.some((endpoint) => endpoint.status === 'ready'));

	async function refresh(silent = false) {
		if (!silent) loading = true;
		try {
			const [datasetResult, recipeResult, artifactResult, jobResult, endpointResult, healthResult] = await Promise.all([
				api.datasets(), api.recipes(), api.artifacts(), api.jobs(true), api.computeEndpoints(true), api.health()
			]);
			datasets = datasetResult.datasets; recipes = recipeResult.recipes; artifacts = artifactResult.artifacts;
			jobs = jobResult.jobs; computeEndpoints = endpointResult.endpoints; systemHealth = healthResult;
		} catch (error) { failure = error instanceof Error ? error.message : 'Could not reach the Oracle Builder control plane.'; }
		finally { if (!silent) loading = false; }
	}
	function navigate(page: Page) { activePage = page; if (typeof window !== 'undefined') window.history.replaceState(null, '', `#${page}`); }
	function compared(ids: string[]) { comparisonIds = ids; navigate('comparison'); }
	function useDraft(draftId: string) { trainingDraftId = draftId; navigate('training'); }
	function changed(message: string) { notice = message; failure = ''; void refresh(true); }

	onMount(() => {
		const page = window.location.hash.slice(1) as Page;
		if (pageMeta.some((item) => item.id === page)) activePage = page;
		void refresh(); const timer = window.setInterval(() => void refresh(true), 10_000);
		return () => window.clearInterval(timer);
	});
</script>

<svelte:head><title>Oracle Builder</title><meta name="description" content="A scientific workspace for model evidence, data, and reproducible training." /></svelte:head>

<div class="app-shell">
	<aside class="app-sidebar">
		<button class="brand" on:click={() => navigate('models')} aria-label="Oracle Builder home"><span class="brand-mark">OB</span><span><strong>Oracle Builder</strong><small>Scientific model workspace</small></span></button>
		<div class="sidebar-group"><p class="nav-label">WORKSPACE</p><nav class="workflow-nav" aria-label="Primary navigation">{#each pageMeta as page}<button class:active={activePage === page.id} on:click={() => navigate(page.id)}><span class="nav-step">{page.number}</span><span><strong>{page.title}</strong><small>{page.detail}</small></span>{#if page.id === 'models' && artifacts.length}<i>{artifacts.length}</i>{:else if page.id === 'training' && activeJobs()}<i>{activeJobs()}</i>{/if}</button>{/each}</nav></div>
		<div class="sidebar-context"><p class="nav-label">SYSTEM</p><dl><div><dt>Models</dt><dd>{artifacts.length}</dd></div><div><dt>Frozen data</dt><dd>{datasets.filter((dataset) => dataset.lifecycle === 'frozen').length}</dd></div><div><dt>Active jobs</dt><dd>{activeJobs()}</dd></div></dl></div>
		<div class="side-foot"><span class:offline={!serviceReady()} class="pulse"></span>{loading ? 'Updating workspace…' : serviceReady() ? 'Compute connected' : 'Plan offline; dispatch later'}</div>
	</aside>
	<div class="app-frame">
		<header class="topbar"><div><p class="eyebrow">{pageMeta.find((page) => page.id === activePage)?.number} · {pageMeta.find((page) => page.id === activePage)?.title}</p><h1>{pageMeta.find((page) => page.id === activePage)?.detail}</h1></div><div class="topbar-actions"><span class:online={serviceReady()} class="system-pill"><i></i>{loading ? 'Connecting…' : serviceReady() ? 'Compute ready' : 'Compute unavailable'}</span><button class="quiet-button" on:click={() => refresh()} disabled={loading}>{loading ? 'Refreshing…' : 'Refresh'}</button></div></header>
		<main class="workspace">
			{#if failure}<div class="banner error" role="alert"><strong>Action needed</strong><span>{failure}</span><button aria-label="Dismiss" on:click={() => failure = ''}>×</button></div>{/if}
			{#if notice}<div class="banner notice" role="status"><strong>Updated</strong><span>{notice}</span><button aria-label="Dismiss" on:click={() => notice = ''}>×</button></div>{/if}
			{#if activePage === 'models'}<ModelRunsView oncompare={compared} onfailure={(message) => failure = message} />
			{:else if activePage === 'comparison'}<ModelComparisonView artifactIds={comparisonIds} onback={() => navigate('models')} onfailure={(message) => failure = message} />
			{:else if activePage === 'datasets'}<TrainingSetCatalogView onuse={() => changed('Catalog sources are read-only. Register and freeze a revision before training.')} onfailure={(message) => failure = message} />
			{:else if activePage === 'construction'}<ModelConstructionView {artifacts} onnotice={changed} onfailure={(message) => failure = message} ontrain={useDraft} />
			{:else}<TrainingStudio {datasets} {recipes} {artifacts} initialDraftId={trainingDraftId} onnotice={changed} onfailure={(message) => failure = message} onplanned={() => { trainingDraftId = ''; void refresh(true); }} />{/if}
		</main>
	</div>
</div>
