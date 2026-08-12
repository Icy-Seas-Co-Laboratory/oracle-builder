<script lang="ts">
	import { onMount } from 'svelte';
	import { api, type RecordValue } from '$lib/api';
	import ArtifactEvidence from '$lib/ArtifactEvidence.svelte';
	import EmptyState from '$lib/EmptyState.svelte';

	export let experiments: RecordValue[] = [];
	export let jobs: RecordValue[] = [];
	export let onfailure: (message: string) => void;
	export let onchanged: (message: string) => void | Promise<void>;
	export let oncreateexperiment: () => void;

	let selectedExperimentId = '';
	let results: RecordValue | null = null;
	let comparisons: RecordValue[] = [];
	let selectedArtifactIds: string[] = [];
	let comparisonName = '';
	let comparisonDescription = '';
	let loading = false;
	let saving = false;
	let observedJobs = jobs;
	let evidenceArtifactId = '';

	$: if (jobs !== observedJobs) {
		observedJobs = jobs;
		if (selectedExperimentId) void loadResults(false);
	}

	const display = (value: unknown) => value === null || value === undefined || value === '' ? '—' : String(value);
	const asRecords = (value: unknown) => Array.isArray(value) ? value as RecordValue[] : [];
	const asRecord = (value: unknown) => value && typeof value === 'object' ? value as RecordValue : null;
	const candidates = () => asRecords(results?.candidates);
	const artifactOf = (candidate: RecordValue) => asRecord(candidate.artifact);
	const selectedArtifacts = () => candidates().map(artifactOf).filter((artifact): artifact is RecordValue => Boolean(artifact && selectedArtifactIds.includes(String(artifact.artifact_id))));
	const summary = () => asRecord(results?.summary);
	const experiment = () => asRecord(results?.experiment);
	const metricValue = (artifact: RecordValue, metric: string) => Number(asRecord(artifact.metrics)?.[metric]);
	const metricPercent = (artifact: RecordValue, metric: string) => Math.max(0, Math.min(100, metricValue(artifact, metric) * 100));
	const formatMetric = (value: number) => Number.isFinite(value) ? value.toFixed(4) : '—';
	const artifactCount = (comparison: RecordValue) => {
		const ids = asRecord(comparison.selection)?.artifact_ids;
		return Array.isArray(ids) ? ids.length : 0;
	};

	function selectionCheck() {
		const artifacts = selectedArtifacts();
		const reasons: string[] = [];
		if (artifacts.length < 2) reasons.push('Select at least two evaluated artifacts.');
		for (const [field, label] of [['task', 'task'], ['dataset_fingerprint_sha256', 'dataset revision'], ['split', 'evaluation split']] as const) {
			const values = new Set(artifacts.map((artifact) => asRecord(artifact.protocol)?.[field]));
			if (values.has(null) || values.has(undefined)) reasons.push(`A selected artifact does not record its ${label}.`);
			else if (values.size > 1) reasons.push(`Selected artifacts use different ${label} values.`);
		}
		for (const [field, label] of [['metric_schema', 'metric schema'], ['decision_rule', 'decision rule'], ['segmentation_target', 'segmentation target']] as const) {
			const values = new Set(artifacts.map((artifact) => asRecord(artifact.protocol)?.[field]).filter((value) => value != null));
			if (values.size > 1) reasons.push(`Selected artifacts use different ${label} values.`);
		}
		const common = artifacts.length ? [...artifacts.map((artifact) => new Set(Object.keys(asRecord(artifact.metrics) ?? {}))).reduce((left, right) => new Set([...left].filter((value) => right.has(value))))] : [];
		if (artifacts.length && !common.length) reasons.push('Selected artifacts have no metrics in common.');
		return { compatible: !reasons.length, reasons, commonMetrics: common };
	}

	function displayedMetrics() {
		const artifacts = selectedArtifacts().length ? selectedArtifacts() : candidates().map(artifactOf).filter((value): value is RecordValue => Boolean(value));
		if (!artifacts.length) return [];
		const task = String(artifacts[0].task);
		const preferred = task === 'segmentation' ? ['mean_dice', 'mean_iou', 'mean_precision', 'mean_recall', 'mean_pixel_accuracy'] : ['accuracy', 'balanced_accuracy', 'macro_f1', 'macro_precision', 'macro_recall', 'macro_average_precision'];
		return preferred.filter((metric) => artifacts.some((artifact) => Number.isFinite(metricValue(artifact, metric))));
	}

	async function loadResults(resetSelection = true) {
		if (!selectedExperimentId) { results = null; return; }
		loading = true;
		if (resetSelection) selectedArtifactIds = [];
		try { results = await api.experimentResults(selectedExperimentId); }
		catch (error) { onfailure(error instanceof Error ? error.message : 'Could not load experiment results.'); }
		finally { loading = false; }
	}

	async function initialize() {
		try {
			comparisons = (await api.comparisons()).comparisons;
			if (experiments.length) { selectedExperimentId = String(experiments[0].experiment_id); await loadResults(); }
		} catch (error) { onfailure(error instanceof Error ? error.message : 'Could not load results workspace.'); }
	}

	function toggleArtifact(artifactId: string) {
		selectedArtifactIds = selectedArtifactIds.includes(artifactId) ? selectedArtifactIds.filter((value) => value !== artifactId) : [...selectedArtifactIds, artifactId];
	}

	async function saveComparison() {
		saving = true;
		try {
			const comparison = await api.createComparison({ name: comparisonName, description: comparisonDescription, artifact_ids: selectedArtifactIds });
			comparisons = (await api.comparisons()).comparisons;
			comparisonName = ''; comparisonDescription = ''; selectedArtifactIds = [];
			await onchanged(`Saved comparison ${display(comparison.name)}.`);
		} catch (error) { onfailure(error instanceof Error ? error.message : 'Could not save comparison.'); }
		finally { saving = false; }
	}

	onMount(initialize);
</script>

<section class="panel results-selector"><div><p class="eyebrow">EXPERIMENT RESULTS</p><h2>Compare evidence produced under the same protocol.</h2><p>Metrics are read from sealed Oracle Builder artifacts, not recalculated in the browser.</p></div><label>Experiment<select bind:value={selectedExperimentId} on:change={() => loadResults()} disabled={!experiments.length}>{#each experiments as item}<option value={item.experiment_id}>{display(item.name)}</option>{/each}</select></label></section>

{#if !experiments.length}
	<section class="panel"><EmptyState title="No experiments available" text="Create and run an experiment before comparing results." action="Create experiment" onaction={oncreateexperiment} /></section>
{:else if loading}
	<section class="panel"><p class="empty">Loading experiment evidence…</p></section>
{:else if results}
	<section class="metrics results-summary"><div><span>Total candidates</span><strong>{display(summary()?.total)}</strong><small>planned specifications</small></div><div><span>Indexed results</span><strong>{display(summary()?.indexed)}</strong><small>validated artifacts</small></div><div><span>Active</span><strong>{display(summary()?.active)}</strong><small>compute or validation</small></div><div><span>Needs attention</span><strong>{display(summary()?.failed)}</strong><small>failed or invalid</small></div></section>

	<section class="panel"><div class="panel-head"><div><h2>{display(experiment()?.name)}</h2><p>{display(experiment()?.description)}</p></div><span class="status {display(experiment()?.status)}">{display(experiment()?.status)}</span></div>{#if candidates().length}<div class="table-scroll"><table><thead><tr><th></th><th>Candidate</th><th>Recipe / seed</th><th>Status</th><th>Runtime</th><th>Primary evidence</th><th></th></tr></thead><tbody>{#each candidates() as candidate}<tr><td>{#if artifactOf(candidate)?.metrics && Object.keys(asRecord(artifactOf(candidate)?.metrics) ?? {}).length}<input class="selection-box" type="checkbox" checked={selectedArtifactIds.includes(String(artifactOf(candidate)?.artifact_id))} aria-label={`Select ${display(candidate.name)} for comparison`} on:change={() => toggleArtifact(String(artifactOf(candidate)?.artifact_id))} />{/if}</td><td><strong>{display(candidate.name)}</strong><small>{display(artifactOf(candidate)?.architecture)} · {display(artifactOf(candidate)?.variant)}</small></td><td>{display(candidate.recipe_name)}<small>seed {display(candidate.seed)}</small></td><td><span class="status {display(candidate.status)}">{display(candidate.status)}</span></td><td>{candidate.runtime_seconds == null ? '—' : `${display(candidate.runtime_seconds)} s`}<small>{display(asRecord(candidate.job)?.worker_id)}</small></td><td>{#if artifactOf(candidate)}{#each Object.entries(asRecord(artifactOf(candidate)?.primary_metrics) ?? {}).slice(0, 2) as metric}<span class="metric-inline">{metric[0].replaceAll('_', ' ')} <strong>{formatMetric(Number(metric[1]))}</strong></span>{/each}{:else}<small>No indexed artifact</small>{/if}</td><td>{#if artifactOf(candidate)}<button class="secondary small" on:click={() => evidenceArtifactId = String(artifactOf(candidate)?.artifact_id)}>Inspect evidence</button>{/if}</td></tr>{/each}</tbody></table></div>{:else}<EmptyState title="No candidates" text="This experiment has no run specifications." />{/if}</section>

	{#if candidates().some((candidate) => artifactOf(candidate))}
		<section class="panel"><div class="panel-head"><div><h2>Metric comparison</h2><p>{selectedArtifactIds.length ? 'Showing selected candidates.' : 'Showing every evaluated candidate.'}</p></div></div>{#if displayedMetrics().length}<div class="metric-groups">{#each displayedMetrics() as metric}<div class="metric-group"><strong>{metric.replaceAll('_', ' ')}</strong>{#each (selectedArtifacts().length ? selectedArtifacts() : candidates().map(artifactOf).filter((value): value is RecordValue => Boolean(value))) as artifact}{#if Number.isFinite(metricValue(artifact, metric))}<div class="metric-row"><span>{display(artifact.name)}</span><div><i style={`width:${metricPercent(artifact, metric)}%`}></i></div><b>{formatMetric(metricValue(artifact, metric))}</b></div>{/if}{/each}</div>{/each}</div>{:else}<EmptyState title="No standard evaluation metrics" text="The indexed artifacts do not contain a recognized held-out evaluation summary." />{/if}</section>
	{/if}

	<section class="panel"><div class="panel-head"><div><h2>Save a comparison</h2><p>Persist the selected artifact IDs and their common evaluation protocol.</p></div><span class="count">{selectedArtifactIds.length}</span></div>{#if selectionCheck().reasons.length}<div class="validation-message invalid"><strong>Selection is not comparable yet</strong><ul>{#each selectionCheck().reasons as reason}<li>{reason}</li>{/each}</ul></div>{:else}<div class="validation-message valid"><strong>Compatible evidence</strong><span>Task, dataset fingerprint, and evaluation split match. Common metrics: {selectionCheck().commonMetrics.join(', ')}.</span></div>{/if}<form class="comparison-form" on:submit|preventDefault={saveComparison}><label>Name<input bind:value={comparisonName} required /></label><label>Description<textarea bind:value={comparisonDescription} rows="2"></textarea></label><button disabled={!selectionCheck().compatible || saving}>{saving ? 'Saving…' : 'Save comparison'}</button></form></section>

	<section class="panel"><div class="panel-head"><div><h2>Saved comparisons</h2><p>Durable selections backed by immutable artifacts.</p></div><span class="count">{comparisons.length}</span></div>{#if comparisons.length}<div class="cards">{#each comparisons as comparison}<article class="comparison-card"><div><strong>{display(comparison.name)}</strong><p>{display(comparison.description)}</p><small>{artifactCount(comparison)} artifacts · {display(asRecord(comparison.protocol)?.split)} split</small></div></article>{/each}</div>{:else}<EmptyState title="No saved comparisons" text="Select compatible evaluated artifacts to record the first comparison." />{/if}</section>
{/if}

{#if evidenceArtifactId}<ArtifactEvidence artifactId={evidenceArtifactId} onclose={() => evidenceArtifactId = ''} {onfailure} />{/if}
