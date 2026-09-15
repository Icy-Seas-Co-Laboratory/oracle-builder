<script lang="ts">
	import { onMount } from 'svelte';
	import { api, type RecordValue } from '$lib/api';
	import ControlCenter from '$lib/ControlCenter.svelte';
	import ServiceStatus from '$lib/ServiceStatus.svelte';

	let datasets: RecordValue[] = [];
	let recipes: RecordValue[] = [];
	let artifacts: RecordValue[] = [];
	let specifications: RecordValue[] = [];
	let jobs: RecordValue[] = [];
	let experiments: RecordValue[] = [];
	let computeEndpoints: RecordValue[] = [];
	let systemHealth: RecordValue | null = null;
	let selectedDatasetId = '';
	let loading = true;
	let notice = '';
	let failure = '';

	const activeStatuses = ['queued', 'running', 'submitted', 'dispatching', 'validating'];
	const activeCount = () => jobs.filter((job) => activeStatuses.includes(String(job.status))).length;
	const healthEndpoints = () => systemHealth?.compute_endpoints && typeof systemHealth.compute_endpoints === 'object' ? systemHealth.compute_endpoints as RecordValue : null;
	const serviceReady = () => systemHealth?.database === 'ready' && (Number(healthEndpoints()?.ready ?? 0) > 0 || computeEndpoints.some((endpoint) => endpoint.status === 'ready'));

	async function refresh(silent = false) {
		if (!silent) loading = true;
		try {
			const [datasetResult, recipeResult, artifactResult, specificationResult, jobResult, experimentResult, endpointResult, healthResult] = await Promise.all([
				api.datasets(), api.recipes(), api.artifacts(), api.specifications(), api.jobs(true), api.experiments(), api.computeEndpoints(true), api.health()
			]);
			datasets = datasetResult.datasets; recipes = recipeResult.recipes; artifacts = artifactResult.artifacts;
			specifications = specificationResult.specifications; jobs = jobResult.jobs; experiments = experimentResult.experiments;
			computeEndpoints = endpointResult.endpoints; systemHealth = healthResult;
			if (!selectedDatasetId && datasets.length) selectedDatasetId = String(datasets[0].dataset_id);
		} catch (error) { failure = error instanceof Error ? error.message : 'Could not reach the Orchestrator.'; }
		finally { if (!silent) loading = false; }
	}
	async function changed(message: string) { notice = message; failure = ''; await refresh(); }
	function failed(message: string) { failure = message; notice = ''; }

	onMount(() => { void refresh(); const timer = window.setInterval(() => void refresh(true), 5000); return () => window.clearInterval(timer); });
</script>

<svelte:head><title>Oracle Builder</title><meta name="description" content="Train, monitor, and organize Oracle Builder models." /></svelte:head>

<main class="control-app">
	<header class="control-topbar"><a class="control-brand" href="/" aria-label="Oracle Builder home"><span>OB</span><strong>Oracle Builder</strong></a><div><span class:online={serviceReady()} class="system-pill"><i></i>{loading ? 'Connecting…' : serviceReady() ? 'Compute ready' : 'Compute unavailable'}</span><button class="quiet-button" on:click={() => refresh()} disabled={loading}>{loading ? 'Refreshing…' : 'Refresh'}</button></div></header>
	{#if failure}<div class="banner error" role="alert"><strong>Action needed</strong><span>{failure}</span><button aria-label="Dismiss" on:click={() => failure = ''}>×</button></div>{/if}
	{#if notice}<div class="banner notice" role="status"><strong>Updated</strong><span>{notice}</span><button aria-label="Dismiss" on:click={() => notice = ''}>×</button></div>{/if}
	<ControlCenter {datasets} {recipes} {artifacts} {specifications} {jobs} {experiments} {computeEndpoints} bind:selectedDatasetId onchanged={changed} onfailure={failed} />
	<footer class="control-footer"><span>{datasets.length} datasets · {recipes.length} recipes · {activeCount()} active runs</span><ServiceStatus ready={serviceReady()} {loading} /></footer>
</main>
