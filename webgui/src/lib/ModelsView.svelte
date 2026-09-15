<script lang="ts">
	import { onMount } from 'svelte';
	import { api, type RecordValue } from '$lib/api';
	import EmptyState from '$lib/EmptyState.svelte';
	import ArtifactEvidence from '$lib/ArtifactEvidence.svelte';

	export let artifacts: RecordValue[] = [];
	export let onimport: () => void;
	export let onfailure: (message: string) => void;

	let selectedId = '';
	let evidenceId = '';
	let catalog: RecordValue[] = [];
	let detail: RecordValue | null = null;
	let loadingDetail = false;
	let detailError = '';
	const record = (value: unknown) => value && typeof value === 'object' ? value as RecordValue : null;
	const display = (value: unknown) => value === null || value === undefined || value === '' ? '—' : String(value);
	const available = () => catalog.length ? catalog : artifacts;
	const selected = () => available().find((artifact) => String(artifact.artifact_id) === selectedId);
	const selectedDetail = () => record(detail?.artifact) ?? selected();
	const metrics = () => Object.entries(record(selectedDetail()?.metrics) ?? {}).filter(([, value]) => Number.isFinite(Number(value))).slice(0, 6);
	const config = (): RecordValue => record(detail?.config) ?? record(selectedDetail()?.resolved_config) ?? record(selectedDetail()?.config) ?? record(selectedDetail()?.metadata) ?? {};
	const contract = (): RecordValue => record(detail?.model_contract) ?? record(record(detail?.manifest)?.model) ?? {};
	async function inspect(id: string) {
		selectedId = id; detail = null; detailError = ''; loadingDetail = true;
		try { detail = await api.artifactDetail(id); }
		catch (error) { detailError = error instanceof Error ? error.message : 'Could not retrieve model metadata.'; onfailure(detailError); }
		finally { loadingDetail = false; }
	}
	$: if (!selectedId && available().length) selectedId = String(available()[0].artifact_id);
	$: if (selectedId && !detail && !loadingDetail && !detailError) void inspect(selectedId);
	onMount(async () => { try { catalog = (await api.artifactCatalog()).artifacts; } catch { /* The registry fallback is enough to inspect evidence. */ } });
</script>

<section class="entity-intro">
	<div><p class="eyebrow">MODELS</p><h2>Understand the model behind each result.</h2><p>Model artifacts are immutable. Inspect their architecture and resolved configuration, then clone a configuration when you need a new candidate.</p></div>
	<button on:click={onimport}>Import model</button>
</section>

{#if available().length}
	<div class="entity-layout">
		<aside class="entity-list" aria-label="Models">
			<div class="list-heading"><strong>Model artifacts</strong><span class="count">{available().length}</span></div>
			{#each available() as artifact}
				<button class:active={String(artifact.artifact_id) === selectedId} on:click={() => inspect(String(artifact.artifact_id))}>
					<strong>{display(artifact.name ?? artifact.artifact_id)}</strong>
					<small>{display(artifact.architecture)} · {display(artifact.task)}</small>
				</button>
			{/each}
		</aside>
		<div class="entity-content">
			{#if selected()}
				<section class="panel dataset-heading"><div><p class="eyebrow">SEALED MODEL ARTIFACT</p><h2>{display(selectedDetail()?.name ?? selectedDetail()?.artifact_id)}</h2><p>{display(selectedDetail()?.architecture)} · {display(selectedDetail()?.task)} · trained against a recorded dataset revision.</p></div><div class="heading-actions"><span class="status {display(selectedDetail()?.status ?? selectedDetail()?.lifecycle)}">{display(selectedDetail()?.status ?? selectedDetail()?.lifecycle)}</span><button on:click={() => evidenceId = selectedId}>View outputs</button></div></section>
				{#if loadingDetail}<section class="panel"><p class="empty" aria-live="polite">Loading model metadata…</p></section>{:else if detailError}<section class="panel"><p class="inline-error" role="alert">{detailError}</p><button class="secondary small" on:click={() => inspect(selectedId)}>Try again</button></section>{:else}
				<section class="dataset-facts"><div><span>Architecture</span><strong>{display(selectedDetail()?.architecture)}</strong></div><div><span>Parameters</span><strong>{display(contract().parameter_count ?? selectedDetail()?.parameter_count ?? selectedDetail()?.parameters)}</strong></div><div><span>Task</span><strong>{display(selectedDetail()?.task)}</strong></div><div><span>Dataset revision</span><strong title={display(selectedDetail()?.dataset_fingerprint_sha256)}>{display(selectedDetail()?.dataset_fingerprint_sha256 ?? selectedDetail()?.dataset_id)}</strong></div></section>
				<section class="panel"><div class="panel-head"><div><h2>Recorded performance</h2><p>Metrics are read from the sealed evaluation summary.</p></div></div>{#if metrics().length}<div class="metric-chip-grid">{#each metrics() as [name, value]}<div><span>{name.replaceAll('_', ' ')}</span><strong>{Number(value).toFixed(4)}</strong></div>{/each}</div>{:else}<p class="empty">This artifact does not expose summary metrics yet.</p>{/if}</section>
				<section class="panel"><div class="panel-head"><div><h2>Architecture & configuration</h2><p>The resolved configuration is a readable model card today; graphical editing belongs in a cloned draft, never in this artifact.</p></div></div><div class="architecture-card"><div><span>Input</span><strong>{display(contract().input_shape ?? config().input_shape ?? config().image_size ?? selectedDetail()?.input_shape)}</strong></div><div><span>Output</span><strong>{display(contract().output_shape ?? config().num_classes ?? config().output_shape ?? selectedDetail()?.output_shape)}</strong></div><div><span>Framework</span><strong>{display(contract().framework ?? config().framework ?? selectedDetail()?.framework ?? 'Keras / TensorFlow')}</strong></div></div><details open><summary>Resolved configuration</summary><pre>{JSON.stringify(config(), null, 2)}</pre></details></section>
				{/if}
			{/if}
		</div>
	</div>
{:else}<section class="panel"><EmptyState title="No model artifacts yet" text="A model appears here after a run is indexed, or after you import a compatible model artifact." action="Import model" onaction={onimport} /></section>{/if}

{#if evidenceId}<ArtifactEvidence artifactId={evidenceId} onclose={() => evidenceId = ''} {onfailure} />{/if}
