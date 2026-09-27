<script lang="ts">
	import { onMount } from 'svelte';
	import { api, type RecordValue } from '$lib/api';
	import { OperationalRefreshCoordinator, type RefreshReason, publishOperationalRefresh } from '$lib/operational-refresh';
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
	let serverLogsOpen = false;
	let serverLogs: RecordValue[] = [];
	let schedulerDiagnostics: RecordValue[] = [];
	let schedulerFindings: string[] = [];
	let serverLogsLoading = false;
	let serverLogsError = '';

	const pageMeta: { id: Page; number: string; title: string; detail: string }[] = [
		{ id: 'models', number: '01', title: 'Model Runs', detail: 'Catalog & inspect' },
		{ id: 'comparison', number: '02', title: 'Comparison', detail: 'Compare evidence' },
		{ id: 'datasets', number: '03', title: 'Training Sets', detail: 'Browse sources' },
		{ id: 'construction', number: '04', title: 'Construction', detail: 'Compose a model' },
		{ id: 'queue', number: '05', title: 'Queue', detail: 'Validate & schedule' }
	];
	const activeStatuses = new Set(['preparing', 'queued', 'running', 'paused', 'submitted', 'dispatching', 'training', 'validating', 'sealing', 'indexing']);
	const record = (value: unknown): RecordValue => value && typeof value === 'object' && !Array.isArray(value) ? value as RecordValue : {};
	const records = (value: unknown): RecordValue[] => Array.isArray(value) ? value as RecordValue[] : [];
	const strings = (value: unknown): string[] => Array.isArray(value) ? value.map(String) : [];
	$: activeJobCount = new Set([
		...jobs.filter((job) => activeStatuses.has(String(job.status))).map((job) => String(job.queued_run_id ?? job.job_id)),
		...queuedRuns.filter((run) => activeStatuses.has(String(run.status))).map((run) => String(run.queued_run_id ?? run.job_id))
	]).size;
	const healthEndpoints = () => systemHealth?.compute_endpoints && typeof systemHealth.compute_endpoints === 'object' ? systemHealth.compute_endpoints as RecordValue : null;
	const serviceReady = () => systemHealth?.database === 'ready' && (Number(healthEndpoints()?.ready ?? 0) > 0 || computeEndpoints.some((endpoint) => endpoint.status === 'ready'));
	const tensorDevicesFor = (endpoints: RecordValue[]) => endpoints.flatMap((endpoint) => {
		const queue = record(endpoint.queue);
		const resources = record(endpoint.resources ?? queue.resources ?? endpoint.scheduler_resources);
		const leases = new Set(strings(resources.gpu_leases));
		const endpointName = String(endpoint.name ?? endpoint.base_url ?? 'Compute');
		const gpus = Array.from(new Map(
			records(endpoint.workers).flatMap((worker) => records(record(worker.capabilities).gpus))
				.filter((gpu) => gpu.id !== undefined && gpu.id !== null)
				.map((gpu) => [String(gpu.id), gpu])
		).values());
		if (gpus.length) return gpus.map((gpu) => ({
			key: `${endpoint.endpoint_id}:${String(gpu.id)}`,
			endpoint: endpointName,
			label: gpu.backend === 'metal' ? 'GPU:0 · Metal' : `GPU:${String(gpu.id)} · CUDA`,
			inUse: leases.has(String(gpu.id)),
			available: endpoint.status === 'ready'
		}));
		if (endpoint.status !== 'ready') return [];
		return [{
			key: `${endpoint.endpoint_id}:cpu`, endpoint: endpointName, label: 'CPU:0',
			inUse: Number(resources.cpu_in_use ?? 0) > 0, available: true
		}];
	});
	$: tensorflowDevices = tensorDevicesFor(computeEndpoints);

	let coordinator: OperationalRefreshCoordinator | undefined;
	async function refresh(silent = false, signal?: AbortSignal, reason: RefreshReason = 'manual') {
		if (!silent) loading = true;
		try {
			const [datasetResult, artifactResult, jobResult, queuedResult, endpointResult, healthResult] = await Promise.all([
				api.datasets({ signal }), api.artifacts({ signal }), api.jobs(true, { signal }), api.queuedRuns({ signal }), api.computeEndpoints(true, { signal }), api.health({ signal })
			]);
			datasets = datasetResult.datasets; artifacts = artifactResult.artifacts; jobs = jobResult.jobs; queuedRuns = queuedResult.queued_runs; computeEndpoints = endpointResult.endpoints; systemHealth = healthResult;
		publishOperationalRefresh(reason);
		} catch (error) {
			if (!(error instanceof DOMException && error.name === 'AbortError')) failure = error instanceof Error ? error.message : 'Could not reach the Oracle Builder control plane.';
		}
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
	function changed(message: string) { notice = message; failure = ''; void coordinator?.trigger('mutation', true); }
	async function loadServerLogs() {
		serverLogsLoading = true; serverLogsError = '';
		try {
			const result = await api.serverLogs();
			serverLogs = records(result.logs);
			schedulerDiagnostics = records(record(result.scheduler).endpoints);
			schedulerFindings = strings(record(result.scheduler).findings);
		} catch (error) { serverLogsError = error instanceof Error ? error.message : 'Could not load server diagnostics.'; }
		finally { serverLogsLoading = false; }
	}
	function openServerLogs() { serverLogsOpen = true; void loadServerLogs(); }

	onMount(() => {
		syncPageFromLocation();
		coordinator = new OperationalRefreshCoordinator(
			(signal, reason) => refresh(reason === 'initial' || reason === 'manual', signal, reason),
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
		<div class="sidebar-group"><p class="nav-label">WORKSPACE</p><nav class="workflow-nav" aria-label="Primary navigation">{#each pageMeta as page}<button class:active={activePage === page.id} on:click={() => navigate(page.id)}><span class="nav-step">{page.number}</span><span><strong>{page.title}</strong><small>{page.detail}</small></span>{#if page.id === 'models' && artifacts.length}<i>{artifacts.length}</i>{:else if page.id === 'queue' && activeJobCount}<i>{activeJobCount}</i>{/if}</button>{/each}</nav></div>
		<div class="sidebar-context"><p class="nav-label">SYSTEM</p><dl><div><dt>Models</dt><dd>{artifacts.length}</dd></div><div><dt>Frozen data</dt><dd>{datasets.filter((dataset) => dataset.lifecycle === 'frozen').length}</dd></div><div><dt>Active jobs</dt><dd>{activeJobCount}</dd></div></dl></div>
		<div class="sidebar-bottom"><section class="tensorflow-devices" aria-live="polite"><p class="nav-label">TENSORFLOW DEVICES</p>{#if tensorflowDevices.length}{#each tensorflowDevices as device}<div class="tensorflow-device" title={device.endpoint}><i class:busy={device.inUse} class:offline={!device.available}></i><span>{device.label}</span><small>{device.inUse ? 'In use' : 'Available'}</small></div>{/each}{:else}<small class="device-empty">No device telemetry available</small>{/if}</section><div class="side-foot"><span class:offline={!serviceReady()} class="pulse"></span>{loading ? 'Updating workspace…' : serviceReady() ? 'Compute connected' : 'Plan offline; dispatch later'}<button class="server-logs-button" on:click={openServerLogs}>Server health & logs</button></div></div>
	</aside>
	<div class="app-frame">
		<header class="topbar"><div><p class="eyebrow">{pageMeta.find((page) => page.id === activePage)?.number} · {pageMeta.find((page) => page.id === activePage)?.title}</p><h1>{pageMeta.find((page) => page.id === activePage)?.detail}</h1></div><div class="topbar-actions"><span class:online={serviceReady()} class="system-pill"><i></i>{loading ? 'Connecting…' : serviceReady() ? 'Compute ready' : 'Compute unavailable'}</span><button class="quiet-button" on:click={() => void coordinator?.trigger('manual', true)} disabled={loading}>{loading ? 'Refreshing…' : 'Refresh'}</button></div></header>
		<main class="workspace">
			{#if failure}<div class="banner error" role="alert"><strong>Action needed</strong><span>{failure}</span><button aria-label="Dismiss" on:click={() => failure = ''}>×</button></div>{/if}
			{#if notice}<div class="banner notice" role="status"><strong>Updated</strong><span>{notice}</span><button aria-label="Dismiss" on:click={() => notice = ''}>×</button></div>{/if}
			{#if activePage === 'models'}<ModelRunsView oncompare={compared} onfailure={(message) => failure = message} />
			{:else if activePage === 'comparison'}<ModelComparisonView artifactIds={comparisonIds} onback={() => navigate('models')} onfailure={(message) => failure = message} />
			{:else if activePage === 'datasets'}<TrainingSetCatalogView onuse={() => { changed('Frozen revision is registered and ready in Queue.'); navigate('queue'); }} onchanged={changed} onfailure={(message) => failure = message} />
			{:else if activePage === 'construction'}<ModelConstructionView onnotice={changed} onfailure={(message) => failure = message} ontrain={useDraft} />
			{:else}<ValidatedQueueView {datasets} {computeEndpoints} onchanged={changed} onfailure={(message) => failure = message} />{/if}
		</main>
	</div>
</div>

{#if serverLogsOpen}<div class="server-logs-backdrop" role="presentation" on:click={(event) => event.target === event.currentTarget && (serverLogsOpen = false)}><div class="server-logs-modal" role="dialog" aria-modal="true" aria-labelledby="server-logs-title" tabindex="-1"><header><div><p class="eyebrow">SERVER HEALTH</p><h2 id="server-logs-title">Oracle Builder service logs</h2><p>Scheduler diagnosis and bounded tails from the local Orchestrator, Serve, and Web GUI processes.</p></div><div><button class="secondary small" disabled={serverLogsLoading} on:click={loadServerLogs}>{serverLogsLoading ? 'Refreshing…' : 'Refresh'}</button><button class="icon-button" aria-label="Close server logs" on:click={() => serverLogsOpen = false}>×</button></div></header>{#if serverLogsLoading && !serverLogs.length}<p class="empty">Loading service diagnostics…</p>{:else if serverLogsError}<p class="server-logs-error">{serverLogsError}</p>{:else}{#if schedulerDiagnostics.length}<section class="scheduler-diagnostics" aria-label="Scheduler diagnosis"><h3>Scheduler diagnosis</h3>{#if schedulerFindings.length}<div class="scheduler-findings">{#each schedulerFindings as finding}<p>{finding}</p>{/each}</div>{/if}<div class="scheduler-diagnostic-grid">{#each schedulerDiagnostics as endpoint}<article><h4>{String(endpoint.name ?? endpoint.endpoint_id ?? 'Compute endpoint')}</h4><small>{String(endpoint.idle_workers ?? 0)} idle / {String(endpoint.worker_slots ?? 0)} worker slots · {String(endpoint.authorized_runs ?? 0)} authorized waiting</small><p>{String(endpoint.diagnosis ?? '')}</p></article>{/each}</div></section>{/if}<div class="server-log-grid">{#each serverLogs as log}<article><header><div><h3>{String(log.name ?? log.service ?? 'Service')}</h3><small>{log.available ? `${String(log.line_count ?? 0)} recent lines · ${String(log.updated_at ?? 'timestamp unavailable')}` : 'Unavailable'}</small></div><span class:available={Boolean(log.available)}>{log.available ? 'Available' : 'Unavailable'}</span></header>{#if log.available}<pre>{String(log.text ?? '') || 'No log output yet.'}</pre>{:else}<p>{String(log.message ?? 'No log file was found for this service.')}</p>{/if}</article>{/each}</div>{/if}</div></div>{/if}
