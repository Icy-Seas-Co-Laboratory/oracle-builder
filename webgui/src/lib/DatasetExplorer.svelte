<script lang="ts">
	import { api, type RecordValue } from '$lib/api';
	import EmptyState from '$lib/EmptyState.svelte';

	export let datasets: RecordValue[] = [];
	export let selectedDatasetId = '';
	export let oncreate: () => void;
	export let onrun: (datasetId: string) => void;

	let detail: RecordValue | null = null;
	let previews: RecordValue[] = [];
	let loading = false;
	let previewError = '';
	let activeId = '';
	let split = '';
	let page = 0;

	const record = (value: unknown) => value && typeof value === 'object' ? value as RecordValue : null;
	const records = (value: unknown) => Array.isArray(value) ? value as RecordValue[] : [];
	const display = (value: unknown) => value === null || value === undefined || value === '' ? '—' : String(value);
	const selected = () => datasets.find((dataset) => String(dataset.dataset_id) === selectedDatasetId);
	const summary = () => record(detail?.info) ?? record(detail?.dataset) ?? detail;
	const counts = () => record(detail?.counts) ?? record(summary()?.counts) ?? record(summary()?.statistics) ?? {};
	const splits = () => record(summary()?.splits) ?? record(summary()?.split_counts) ?? {};
	const previewUrl = (sample: RecordValue) => String(sample.preview_url ?? sample.image_url ?? sample.url ?? '');
	const previewLabel = (sample: RecordValue) => display(sample.label ?? sample.class_name ?? sample.target ?? sample.uuid);

	async function load(id: string) {
		if (!id) return;
		loading = true; previewError = ''; activeId = id;
		try {
			const [nextDetail, previewResult] = await Promise.all([api.datasetDetail(id), api.datasetPreviews(id, { limit: 24, offset: page * 24, split })]);
			if (activeId !== id) return;
			detail = nextDetail;
			previews = records(previewResult.samples ?? previewResult.items ?? previewResult.previews);
		} catch (error) {
			if (activeId === id) { detail = null; previews = []; previewError = error instanceof Error ? error.message : 'Preview service is unavailable.'; }
		} finally { if (activeId === id) loading = false; }
	}

	$: if (!selectedDatasetId && datasets.length) selectedDatasetId = String(datasets[0].dataset_id);
	$: if (selectedDatasetId && selectedDatasetId !== activeId) { page = 0; void load(selectedDatasetId); }
	$: if (selectedDatasetId && activeId === selectedDatasetId) { void split; }
	function choose(id: string) { if (id !== selectedDatasetId) { selectedDatasetId = id; detail = null; previews = []; } }
	function setSplit(value: string) { split = value; page = 0; activeId = ''; }
	function nextPage(delta: number) { page = Math.max(0, page + delta); activeId = ''; }
</script>

<section class="entity-intro">
	<div><p class="eyebrow">DATASETS</p><h2>See the data before you train.</h2><p>Inspect a frozen revision, its composition, and representative samples. Every run keeps a link back to this exact dataset.</p></div>
	<button on:click={oncreate}>Add dataset</button>
</section>

{#if datasets.length}
	<div class="entity-layout">
		<aside class="entity-list" aria-label="Datasets">
			<div class="list-heading"><strong>Your datasets</strong><span class="count">{datasets.length}</span></div>
			{#each datasets as dataset}
				<button class:active={String(dataset.dataset_id) === selectedDatasetId} on:click={() => choose(String(dataset.dataset_id))}>
					<strong>{display(dataset.name ?? dataset.dataset_id)}</strong>
					<small>{display(dataset.task)} · {display(dataset.status ?? dataset.lifecycle)}</small>
				</button>
			{/each}
		</aside>
		<div class="entity-content">
			{#if selected()}
				<section class="panel dataset-heading">
					<div><p class="eyebrow">FROZEN REVISION</p><h2>{display(selected()?.name ?? selected()?.dataset_id)}</h2><p>{display(selected()?.description ?? summary()?.description ?? 'A versioned dataset ready for inspection.')}</p></div>
					<div class="heading-actions"><span class="status {display(selected()?.status ?? selected()?.lifecycle)}">{display(selected()?.status ?? selected()?.lifecycle)}</span><button on:click={() => onrun(selectedDatasetId)}>Create run</button></div>
				</section>

				<section class="dataset-facts">
					<div><span>Examples</span><strong>{display(counts().items ?? counts().samples ?? selected()?.item_count ?? selected()?.count)}</strong></div>
					<div><span>Task</span><strong>{display(summary()?.task ?? selected()?.task)}</strong></div>
					<div><span>Revision</span><strong title={display(summary()?.fingerprint ?? selected()?.fingerprint ?? selected()?.dataset_id)}>{display(summary()?.revision ?? selected()?.revision ?? selected()?.dataset_id)}</strong></div>
					<div><span>Schema</span><strong>{display(summary()?.schema_version ?? selected()?.schema_version ?? 'Recorded')}</strong></div>
				</section>

				<section class="panel">
					<div class="panel-head"><div><h2>Sample browser</h2><p>Representative records are rendered as bounded previews; raw data stays in the dataset store.</p></div><label class="compact-label">Split<select value={split} on:change={(event) => setSplit(event.currentTarget.value)}><option value="">All splits</option>{#each Object.keys(splits()) as name}<option value={name}>{name}</option>{/each}</select></label></div>
					{#if loading}<p class="empty">Loading dataset details and previews…</p>
					{:else if previewError}<div class="preview-unavailable"><strong>Previews are not available yet</strong><p>{previewError}</p><small>The dataset record is still usable. This view will populate once the preview endpoint is available.</small></div>
					{:else if previews.length}<div class="sample-grid">{#each previews as sample}<article class="sample-card">{#if previewUrl(sample)}<img src={previewUrl(sample)} alt={`Dataset sample: ${previewLabel(sample)}`} loading="lazy" />{:else}<div class="sample-placeholder">No preview</div>{/if}<div><strong>{previewLabel(sample)}</strong><small>{display(sample.split)} · {display(sample.uuid ?? sample.id)}</small></div></article>{/each}</div><div class="pager"><button class="secondary small" disabled={page === 0} on:click={() => nextPage(-1)}>Previous</button><span>Page {page + 1}</span><button class="secondary small" disabled={previews.length < 24} on:click={() => nextPage(1)}>Next</button></div>
					{:else}<EmptyState title="No preview samples" text="This dataset has no previewable records for the current filter." />{/if}
				</section>

				<section class="panel"><div class="panel-head"><div><h2>Composition</h2><p>Counts recorded for this dataset revision.</p></div></div>{#if records(detail?.labels).length}<div class="composition-grid">{#each records(detail?.labels) as label}<div><span>{display(label.name)}</span><strong>{display(label.item_count)}</strong></div>{/each}</div>{:else}<p class="empty">This dataset does not expose class counts.</p>{/if}<details><summary>Dataset identity & provenance</summary><dl class="identity-grid"><div><dt>Dataset ID</dt><dd><code>{display(selected()?.dataset_id)}</code></dd></div><div><dt>Fingerprint</dt><dd><code>{display(summary()?.fingerprint_sha256 ?? selected()?.fingerprint_sha256)}</code></dd></div><div><dt>Source</dt><dd>{display(summary()?.path ?? selected()?.path)}</dd></div></dl></details></section>
			{/if}
		</div>
	</div>
{:else}<section class="panel"><EmptyState title="Add your first dataset" text="Register a dataset to inspect its contents and use the exact revision in a training run." action="Add dataset" onaction={oncreate} /></section>{/if}
