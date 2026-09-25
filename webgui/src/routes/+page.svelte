<script lang="ts">
	import { onMount } from 'svelte';
	import { api, type RecordValue } from '$lib/api';
	import ModelRunsView from '$lib/models/ModelRunsView.svelte';
	import ModelComparisonView from '$lib/models/ModelComparisonView.svelte';
	import TrainingSetCatalogView from '$lib/TrainingSetCatalogView.svelte';
	import ModelConstructionView from '$lib/ModelConstructionView.svelte';
	import ValidatedQueueView from '$lib/ValidatedQueueView.svelte';

	type Page = 'models' | 'comparison' | 'datasets' | 'construction' | 'queue';
	let datasets: RecordValue[] = [];
	let artifacts: RecordValue[] = [];
	let jobs: RecordValue[] = [];
	let queuedRuns: RecordValue[] = [];
	let computeEndpoints: RecordValue[] = [];
	let systemHealth: RecordValue | null = null;
	let activePage: Page = 'models';
	let comparisonIds: string[] = [];
	let loading = true;
	let notice = '';
	let failure = '';

	const pageMeta: { id: Page; number: string; title: string; detail: string }[] = [
		{ id: 'models', number: '01', title: 'Model Runs', detail: 'Catalog & inspect' },
		{ id: 'comparison', number: '02', title: 'Comparison', detail: 'Compare evidence' },
		{ id: 'datasets', number: '03', title: 'Training Sets', detail: 'Browse sources' },
		{ id: 'construction', number: '04', title: 'Construction', detail: 'Compose a model' },
		{ id: 'queue', number: '05', title: 'Queue', detail: 'Validate & schedule' }
	];
	const activeStatuses = new Set(['preparing', 'queued', 'running', 'paused', 'submitted', 'dispatching', 'training', 'validating', 'sealing', 'indexing']);
	$: activeJobCount = new Set([
		...jobs.filter((job) => activeStatuses.has(String(job.status))).map((job) => String(job.queued_run_id ?? job.job_id)),
		...queuedRuns.filter((run) => activeStatuses.has(String(run.status))).map((run) => String(run.queued_run_id ?? run.job_id))
	]).size;
	const healthEndpoints = () => systemHealth?.compute_endpoints && typeof systemHealth.compute_endpoints === 'object' ? systemHealth.compute_endpoints as RecordValue : null;
	const serviceReady = () => systemHealth?.database === 'ready' && (Number(healthEndpoints()?.ready ?? 0) > 0 || computeEndpoints.some((endpoint) => endpoint.status === 'ready'));

	async function refresh(silent = false) {
		if (!silent) loading = true;
		try {
			const [datasetResult, artifactResult, jobResult, queuedResult, endpointResult, healthResult] = await Promise.all([
				api.datasets(), api.artifacts(), api.jobs(true), api.queuedRuns(), api.computeEndpoints(true), api.health()
			]);
			datasets = datasetResult.datasets; artifacts = artifactResult.artifacts; jobs = jobResult.jobs; queuedRuns = queuedResult.queued_runs; computeEndpoints = endpointResult.endpoints; systemHealth = healthResult;
		} catch (error) { failure = error instanceof Error ? error.message : 'Could not reach the Oracle Builder control plane.'; }
		finally { if (!silent) loading = false; }
	}
	function syncPageFromLocation() {
		if (typeof window === 'undefined') return;
		const page = window.location.hash.slice(1) as Page;
		if (pageMeta.some((item) => item.id === page)) activePage = page;
	}
	function navigate(page: Page) { activePage = page; if (typeof window !== 'undefined') window.history.replaceState(null, '', `#${page}`); }
	function compared(ids: string[]) { comparisonIds = ids; navigate('comparison'); }
	function useDraft(_definitionId: string) { navigate('queue'); }
	function changed(message: string) { notice = message; failure = ''; void refresh(true); }

	onMount(() => {
		syncPageFromLocation();
		void refresh(); const timer = window.setInterval(() => void refresh(true), 10_000);
		window.addEventListener('hashchange', syncPageFromLocation);
		window.addEventListener('popstate', syncPageFromLocation);
		return () => {
			window.clearInterval(timer);
			window.removeEventListener('hashchange', syncPageFromLocation);
			window.removeEventListener('popstate', syncPageFromLocation);
		};
	});
</script>

<svelte:head><title>{pageMeta.find((page) => page.id === activePage)?.title ?? 'Workspace'} · Oracle Builder</title><meta name="description" content="A scientific workspace for model evidence, data, and reproducible training." /></svelte:head>

<div class="app-shell">
	<aside class="app-sidebar">
		<button class="brand" on:click={() => navigate('models')} aria-label="Oracle Builder home"><img class="brand-mark" src="/brand/oracle-builder-mark.webp" alt="" /><span><strong>Oracle Builder</strong><small>Scientific model workspace</small></span></button>
		<div class="sidebar-group"><p class="nav-label">WORKSPACE</p><nav class="workflow-nav" aria-label="Primary navigation">{#each pageMeta as page}<button class:active={activePage === page.id} on:click={() => navigate(page.id)}><span class="nav-step">{page.number}</span><span><strong>{page.title}</strong><small>{page.detail}</small></span>{#if page.id === 'models' && artifacts.length}<i>{artifacts.length}</i>{:else if page.id === 'queue' && activeJobCount}<i>{activeJobCount}</i>{/if}</button>{/each}</nav></div>
		<div class="sidebar-context"><p class="nav-label">SYSTEM</p><dl><div><dt>Models</dt><dd>{artifacts.length}</dd></div><div><dt>Frozen data</dt><dd>{datasets.filter((dataset) => dataset.lifecycle === 'frozen').length}</dd></div><div><dt>Active jobs</dt><dd>{activeJobCount}</dd></div></dl></div>
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
			{:else if activePage === 'construction'}<ModelConstructionView onnotice={changed} onfailure={(message) => failure = message} ontrain={useDraft} />
			{:else}<ValidatedQueueView {datasets} {computeEndpoints} onchanged={changed} onfailure={(message) => failure = message} />{/if}
		</main>
	</div>
</div>
