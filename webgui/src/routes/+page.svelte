<script lang="ts">
	import { onMount } from 'svelte';
	import TransientBanner from '$lib/TransientBanner.svelte';
	import { api, type RecordValue } from '$lib/api';
	import { OperationalRefreshCoordinator, type RefreshReason, publishOperationalRefresh } from '$lib/operational-refresh';
	import ModelRunsView from '$lib/models/ModelRunsView.svelte';
	import ModelComparisonView from '$lib/models/ModelComparisonView.svelte';
	import TrainingSetCatalogView from '$lib/TrainingSetCatalogView.svelte';
	import ModelConstructionView from '$lib/ModelConstructionView.svelte';
	import WorkerFleetView from '$lib/WorkerFleetView.svelte';
	import LiveDashboardView from '$lib/LiveDashboardView.svelte';
	import OperationsView from '$lib/OperationsView.svelte';

	type Page = 'models' | 'comparison' | 'datasets' | 'construction' | 'queue' | 'workers' | 'dashboard' | 'operations';
	let datasets: RecordValue[] = [];
	let artifacts: RecordValue[] = [];
	let jobs: RecordValue[] = [];
	let systemHealth: RecordValue | null = null;
	let computerName = '';
	let activePage: Page = 'models';
	let comparisonIds: string[] = [];
	let queuedDefinition: { id: string; revision: number } | null = null;
	let queuedDatasetId = '';
	let constructionView: { confirmLeave: () => boolean } | undefined;
	let loading = true;
	let notice = '';
	let failure = '';
	let refreshFailure = '';
	let serverLogsOpen = false;
	let serverLogs: RecordValue[] = [];
	let serverLogsLoading = false;
	let serverLogsError = '';

	const pageMeta: { id: Page; number: string; title: string; detail: string }[] = [
		{ id: 'models', number: '01', title: 'Model Runs', detail: 'Catalog & inspect' },
		{ id: 'comparison', number: '02', title: 'Comparison', detail: 'Compare evidence' },
		{ id: 'datasets', number: '03', title: 'Training Sets', detail: 'Browse sources' },
		{ id: 'construction', number: '04', title: 'Construction', detail: 'Compose a model' },
		{ id: 'queue', number: '05', title: 'Queue', detail: 'Validate & schedule' },
		{ id: 'workers', number: '06', title: 'Workers', detail: 'Remote execution fleet' },
		{ id: 'dashboard', number: '07', title: 'Live Dashboard', detail: 'Watch jobs & workers' },
		{ id: 'operations', number: '08', title: 'Operations', detail: 'Demo health & recovery' }
	];
	const navigationGroups: { label: string; pages: Page[] }[] = [
		{ label: 'PREPARE', pages: ['datasets'] },
		{ label: 'DESIGN', pages: ['construction'] },
		{ label: 'RUN', pages: ['queue', 'dashboard'] },
		{ label: 'REVIEW', pages: ['models', 'comparison'] },
		{ label: 'SYSTEM', pages: ['workers', 'operations'] }
	];
	const activeStatuses = new Set(['queued', 'leased', 'acknowledged', 'running', 'paused', 'sealing', 'indexing', 'uploading', 'publishing', 'publication_pending', 'publication_paused', 'cancel_requested']);
	const record = (value: unknown): RecordValue => value && typeof value === 'object' && !Array.isArray(value) ? value as RecordValue : {};
	const records = (value: unknown): RecordValue[] => Array.isArray(value) ? value as RecordValue[] : [];
	$: activeJobCount = new Set([
		...jobs.filter((job) => activeStatuses.has(String(job.status))).map((job) => String(job.job_id))
	]).size;
	$: serviceReady = systemHealth?.database === 'ready';

	let coordinator: OperationalRefreshCoordinator | undefined;
	async function refresh(silent = false, signal?: AbortSignal, reason: RefreshReason = 'manual') {
		if (!silent) loading = true;
		try {
			const [datasetResult, artifactResult, jobResult, healthResult] = await Promise.allSettled([
				api.datasets({ signal }), api.artifacts({ signal }), api.jobs(true, { signal }), api.health({ signal })
			]);
			if (signal?.aborted) return;
			if (datasetResult.status === 'fulfilled') datasets = datasetResult.value.datasets;
			if (artifactResult.status === 'fulfilled') artifacts = artifactResult.value.artifacts;
			if (jobResult.status === 'fulfilled') jobs = jobResult.value.jobs;
			if (healthResult.status === 'fulfilled') {
				systemHealth = healthResult.value;
				computerName = typeof healthResult.value.computer_name === 'string' ? healthResult.value.computer_name : '';
			}
			if (healthResult.status === 'rejected') systemHealth = null;
			publishOperationalRefresh(reason);
			const failed: string[] = [];
			if (datasetResult.status === 'rejected') failed.push('datasets');
			if (artifactResult.status === 'rejected') failed.push('artifacts');
			if (jobResult.status === 'rejected') failed.push('jobs');
			if (healthResult.status === 'rejected') failed.push('service health');
			if (failed.length) refreshFailure = `Could not refresh ${failed.join(', ')}. Showing the last available data.`;
			else refreshFailure = '';
		} catch (error) {
			if (!(error instanceof DOMException && error.name === 'AbortError')) refreshFailure = error instanceof Error ? error.message : 'Could not reach the Oracle Builder control plane.';
		}
		finally { if (!silent) loading = false; }
	}
	function syncPageFromLocation() {
		if (typeof window === 'undefined') return;
		const page = window.location.hash.slice(1) as Page;
		if (!pageMeta.some((item) => item.id === page)) return;
		if (page !== activePage && activePage === 'construction' && !constructionView?.confirmLeave()) {
			window.history.replaceState(null, '', '#construction');
			return;
		}
		activePage = page;
	}
	function navigate(page: Page) {
		if (page !== activePage && activePage === 'construction' && !constructionView?.confirmLeave()) return;
		activePage = page;
		if (typeof window !== 'undefined') window.history.replaceState(null, '', `#${page}`);
	}
	function compared(ids: string[]) { comparisonIds = ids; navigate('comparison'); }
	function useDraft(definitionId: string, revision: number) { queuedDefinition = { id: definitionId, revision }; navigate('queue'); }
	function useDataset(dataset: RecordValue) {
		const datasetId = String(dataset.dataset_id ?? dataset.id ?? '');
		if (datasetId) queuedDatasetId = datasetId;
		changed('Frozen revision is selected in Queue and ready to pair with a definition.');
		navigate('queue');
	}
	function changed(message: string) { notice = message; failure = ''; void coordinator?.trigger('mutation', true); }
	async function loadServerLogs() {
		serverLogsLoading = true; serverLogsError = '';
		try {
			const result = await api.serverLogs();
			serverLogs = records(result.logs);
		} catch (error) { serverLogsError = error instanceof Error ? error.message : 'Could not load server diagnostics.'; }
		finally { serverLogsLoading = false; }
	}
	function openServerLogs() { serverLogsOpen = true; void loadServerLogs(); }

	onMount(() => {
		syncPageFromLocation();
		coordinator = new OperationalRefreshCoordinator(
			(signal, reason) => refresh(!(reason === 'initial' || reason === 'manual'), signal, reason),
			() => activeJobCount > 0
		);
		coordinator.start();
		window.addEventListener('hashchange', syncPageFromLocation);
		window.addEventListener('popstate', syncPageFromLocation);
		return () => {
			coordinator?.stop(); coordinator = undefined;
			window.removeEventListener('hashchange', syncPageFromLocation);
			window.removeEventListener('popstate', syncPageFromLocation);
		};
	});
</script>

<svelte:head><title>{pageMeta.find((page) => page.id === activePage)?.title ?? 'Workspace'} · Oracle Builder</title><meta name="description" content="A scientific workspace for model evidence, data, and reproducible training." /></svelte:head>

<div class="app-shell">
	<aside class="app-sidebar">
		<button class="brand" on:click={() => navigate('models')} aria-label="Oracle Builder home"><img class="brand-mark" src="/brand/oracle-builder-mark.webp" alt="" /><span><strong>Oracle Builder</strong><small>Scientific model workspace</small></span></button>
		<div class="sidebar-group"><nav class="workflow-nav" aria-label="Primary navigation">{#each navigationGroups as group}<p class="nav-label">{group.label}</p>{#each group.pages as pageId}{@const page = pageMeta.find((item) => item.id === pageId)!}<button class:active={activePage === page.id} on:click={() => navigate(page.id)}><span class="nav-step">{page.number}</span><span><strong>{page.title}</strong><small>{page.detail}</small></span>{#if page.id === 'models' && artifacts.length}<i>{artifacts.length}</i>{:else if page.id === 'queue' && activeJobCount}<i>{activeJobCount}</i>{/if}</button>{/each}{/each}</nav></div>
		<div class="sidebar-context"><p class="nav-label">SYSTEM</p><dl><div><dt>Models</dt><dd>{artifacts.length}</dd></div><div><dt>Frozen data</dt><dd>{datasets.filter((dataset) => dataset.lifecycle === 'frozen').length}</dd></div><div><dt>Active jobs</dt><dd>{activeJobCount}</dd></div></dl></div>
		<div class="sidebar-bottom"><div class="side-foot"><span class:offline={!serviceReady} class="pulse"></span>{loading ? 'Updating workspace…' : serviceReady ? 'Control plane connected' : 'Control plane unavailable'}<button class="server-logs-button" on:click={openServerLogs}>Service logs</button></div></div>
	</aside>
	<div class="app-frame">
		<header class="topbar"><div><p class="eyebrow">{pageMeta.find((page) => page.id === activePage)?.number} · {pageMeta.find((page) => page.id === activePage)?.title}</p><h1>{pageMeta.find((page) => page.id === activePage)?.detail}</h1></div><div class="topbar-actions"><span class="computer-pill" title={`Computer: ${computerName || 'Unavailable'}`}><span>Computer</span><strong>{computerName || 'Unavailable'}</strong></span><span class:online={serviceReady} class="system-pill"><i></i>{loading ? 'Connecting…' : serviceReady ? 'Orchestrator ready' : 'Orchestrator unavailable'}</span><button class="quiet-button" on:click={() => void coordinator?.trigger('manual', true)} disabled={loading}>{loading ? 'Refreshing…' : 'Refresh'}</button></div></header>
		<main class="workspace">
			{#if refreshFailure}<TransientBanner message={refreshFailure} title="Refresh interrupted" tone="error" ondismiss={() => refreshFailure = ''} />{/if}
			{#if failure}<TransientBanner message={failure} title="Action needed" tone="error" ondismiss={() => failure = ''} />{/if}
			{#if notice}<TransientBanner message={notice} title="Updated" tone="notice" ondismiss={() => notice = ''} />{/if}
			{#if activePage === 'models'}<ModelRunsView oncompare={compared} onfailure={(message) => failure = message} />
			{:else if activePage === 'comparison'}<ModelComparisonView artifactIds={comparisonIds} onback={() => navigate('models')} onfailure={(message) => failure = message} />
			{:else if activePage === 'datasets'}<TrainingSetCatalogView onuse={useDataset} onchanged={changed} onfailure={(message) => failure = message} />
			{:else if activePage === 'construction'}<ModelConstructionView bind:this={constructionView} onnotice={changed} onfailure={(message) => failure = message} ontrain={useDraft} />
			{:else if activePage === 'queue'}<WorkerFleetView mode="queue" preferredDefinitionId={queuedDefinition?.id ?? ''} preferredDefinitionRevision={queuedDefinition?.revision ?? 0} preferredDatasetId={queuedDatasetId} onchanged={changed} onfailure={(message) => failure = message} />
			{:else if activePage === 'workers'}<WorkerFleetView mode="workers" onchanged={changed} onfailure={(message) => failure = message} />
			{:else if activePage === 'dashboard'}<LiveDashboardView onfailure={(message) => failure = message} />
			{:else}<OperationsView onfailure={(message) => failure = message} />{/if}
		</main>
	</div>
</div>

{#if serverLogsOpen}<div class="server-logs-backdrop" role="presentation" on:click={(event) => event.target === event.currentTarget && (serverLogsOpen = false)}><div class="server-logs-modal" role="dialog" aria-modal="true" aria-labelledby="server-logs-title" tabindex="-1"><header><div><p class="eyebrow">CONTROL PLANE HEALTH</p><h2 id="server-logs-title">Oracle Builder service logs</h2><p>Bounded operational logs from the Orchestrator and Web GUI.</p></div><div><button class="secondary small" disabled={serverLogsLoading} on:click={loadServerLogs}>{serverLogsLoading ? 'Refreshing…' : 'Refresh'}</button><button class="icon-button" aria-label="Close server logs" on:click={() => serverLogsOpen = false}>×</button></div></header>{#if serverLogsLoading && !serverLogs.length}<p class="empty">Loading service logs…</p>{:else if serverLogsError}<p class="server-logs-error">{serverLogsError}</p>{:else}<div class="server-log-grid">{#each serverLogs as log}<article><header><div><h3>{String(log.name ?? log.service ?? 'Service')}</h3><small>{log.available ? `${String(log.line_count ?? 0)} recent lines · ${String(log.updated_at ?? 'timestamp unavailable')}` : 'Unavailable'}</small></div><span class:available={Boolean(log.available)}>{log.available ? 'Available' : 'Unavailable'}</span></header>{#if log.available}<pre>{String(log.text ?? '') || 'No log output yet.'}</pre>{:else}<p>{String(log.message ?? 'No log file was found for this service.')}</p>{/if}</article>{/each}</div>{/if}</div></div>{/if}
