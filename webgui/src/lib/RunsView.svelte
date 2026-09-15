<script lang="ts">
	import { onMount } from 'svelte';
	import { api, type RecordValue } from '$lib/api';
	import ArtifactEvidence from '$lib/ArtifactEvidence.svelte';
	import EmptyState from '$lib/EmptyState.svelte';

	export let artifacts: RecordValue[] = [];
	export let onfailure: (message: string) => void;
	export let onnewrun: () => void;
	let catalog: RecordValue[] = [];

	let query = '';
	let task = '';
	let selectedId = '';
	let detail: RecordValue | null = null;
	let history: RecordValue[] = [];
	let loadingDetail = false;
	let evidenceId = '';

	const asRecord = (value: unknown) => value && typeof value === 'object' ? value as RecordValue : null;
	const records = (value: unknown) => Array.isArray(value) ? value as RecordValue[] : [];
	const text = (value: unknown) => value == null || value === '' ? '—' : String(value);
	const number = (value: unknown) => Number.isFinite(Number(value)) ? Number(value) : null;
	const metrics = (row: RecordValue) => asRecord(row.metrics) ?? asRecord(asRecord(row.manifest)?.summary)?.evaluation ?? {};
	const run = () => asRecord(detail?.artifact) ?? detail;
	const primary = (row: RecordValue) => Object.entries(metrics(row)).filter(([, value]) => number(value) != null).slice(0, 2);
	const available = () => catalog.length ? catalog : artifacts;
	const filtered = () => available().filter((item) => {
		const haystack = [item.name, item.task, item.architecture, item.variant, item.dataset_id].join(' ').toLowerCase();
		return (!query || haystack.includes(query.toLowerCase())) && (!task || item.task === task);
	});
	const historyNumbers = () => history.map((row, index) => ({ x: number(row.epoch) ?? number(row.step) ?? index, y: number(row.loss) ?? number(row.val_loss) }));
	const chartPoints = () => {
		const points = historyNumbers().filter((point) => point.y != null) as { x: number; y: number }[];
		if (points.length < 2) return '';
		const minX = Math.min(...points.map((point) => point.x)); const maxX = Math.max(...points.map((point) => point.x));
		const minY = Math.min(...points.map((point) => point.y)); const maxY = Math.max(...points.map((point) => point.y));
		return points.map((point) => `${8 + ((point.x - minX) / (maxX - minX || 1)) * 284},${112 - ((point.y - minY) / (maxY - minY || 1)) * 96}`).join(' ');
	};

	async function inspect(id: string) {
		selectedId = id; detail = null; history = []; loadingDetail = true;
		try {
			const [nextDetail, nextHistory] = await Promise.all([api.artifactDetail(id), api.artifactHistory(id)]);
			detail = nextDetail; history = records(nextHistory.history ?? nextHistory.rows);
		} catch (error) { onfailure(error instanceof Error ? error.message : 'Could not load run details.'); }
		finally { loadingDetail = false; }
	}
	onMount(async () => { try { catalog = (await api.artifactCatalog()).artifacts; } catch { /* The basic registry remains usable. */ } });
</script>

<section class="panel results-selector">
	<div><p class="eyebrow">RUNS</p><h2>Every model run in one place.</h2><p>Search outcomes, inspect training behavior, and open the evidence behind each result.</p></div>
	<button on:click={onnewrun}>New run</button>
</section>

<section class="panel">
	<div class="run-filters"><label>Search runs<input placeholder="Name, architecture, dataset…" bind:value={query} /></label><label>Task<select bind:value={task}><option value="">All tasks</option>{#each [...new Set(artifacts.map((item) => String(item.task)).filter(Boolean))] as item}<option value={item}>{item}</option>{/each}</select></label><span>{filtered().length} runs</span></div>
	{#if filtered().length}<div class="table-scroll"><table><thead><tr><th>Run</th><th>Dataset</th><th>Outcome</th><th>Timing</th><th></th></tr></thead><tbody>{#each filtered() as artifact}<tr class:selected={selectedId === artifact.artifact_id}><td><strong>{text(artifact.name)}</strong><small>{text(artifact.architecture)} · {text(artifact.variant)} · {text(artifact.task)}</small></td><td><small>{text(artifact.dataset_id)}</small></td><td>{#each primary(artifact) as metric}<span class="metric-inline">{metric[0].replaceAll('_', ' ')} <strong>{Number(metric[1]).toFixed(4)}</strong></span>{:else}<small>No evaluation summary</small>{/each}</td><td>{artifact.runtime_seconds == null ? '—' : `${text(artifact.runtime_seconds)} s`}</td><td><button class="secondary small" on:click={() => inspect(String(artifact.artifact_id))}>Inspect</button></td></tr>{/each}</tbody></table></div>{:else}<EmptyState title="No matching runs" text="Try a different filter or create a new run." action="New run" onaction={onnewrun} />{/if}
</section>

{#if selectedId}<section class="panel run-inspector"><div class="panel-head"><div><p class="eyebrow">RUN INSPECTOR</p><h2>{text(run()?.name ?? filtered().find((item) => item.artifact_id === selectedId)?.name)}</h2><p>{text(run()?.architecture)} · {text(run()?.task)} · immutable artifact evidence</p></div><button class="secondary small" on:click={() => { selectedId = ''; detail = null; }}>Close</button></div>
	{#if loadingDetail}<p class="empty">Loading run details…</p>{:else if detail}<div class="metrics compact-metrics">{#each Object.entries(asRecord(run()?.metrics) ?? {}).filter(([, value]) => number(value) != null).slice(0, 4) as metric}<div><span>{metric[0].replaceAll('_', ' ')}</span><strong>{Number(metric[1]).toFixed(4)}</strong></div>{/each}</div>
		<div class="two-col"><section><h3>Training loss</h3>{#if chartPoints()}<svg class="history-chart" viewBox="0 0 300 120" role="img" aria-label="Training loss history"><line x1="8" y1="112" x2="292" y2="112" /><polyline points={chartPoints()} /></svg>{:else}<p class="empty">No history was recorded for this artifact.</p>{/if}</section><section><h3>Model & protocol</h3><dl class="detail-list"><div><dt>Architecture</dt><dd>{text(run()?.architecture)}</dd></div><div><dt>Dataset</dt><dd>{text(run()?.dataset_id)}</dd></div><div><dt>Split</dt><dd>{text(asRecord(run()?.protocol)?.split)}</dd></div><div><dt>Runtime</dt><dd>{run()?.runtime_seconds == null ? '—' : `${text(run()?.runtime_seconds)} s`}</dd></div></dl></section></div>
		{#if history.length}<div class="table-scroll"><table><thead><tr>{#each Object.keys(history[0]).slice(0, 8) as field}<th>{field.replaceAll('_', ' ')}</th>{/each}</tr></thead><tbody>{#each history.slice(0, 50) as row}<tr>{#each Object.keys(history[0]).slice(0, 8) as field}<td>{text(row[field])}</td>{/each}</tr>{/each}</tbody></table></div>{/if}
		<div class="row-actions"><button on:click={() => evidenceId = selectedId}>View outputs & evidence</button></div>
	{/if}</section>{/if}

{#if evidenceId}<ArtifactEvidence artifactId={evidenceId} onclose={() => evidenceId = ''} {onfailure} />{/if}
