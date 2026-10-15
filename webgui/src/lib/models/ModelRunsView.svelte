<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { api, type RecordValue } from '$lib/api';
	import PairedMetricPlot from '$lib/PairedMetricPlot.svelte';
	import { modelCatalogClient, type CatalogQuery } from './catalog-client';

	export let oncompare: (artifactIds: string[]) => void = () => {};
	export let onfailure: (message: string) => void = () => {};

	type Field = { key: string; label?: string; type?: string; operators?: string[] };
	const defaults = ['name', 'status', 'architecture', 'variant', 'classifier_type', 'training_set', 'macro_f1', 'accuracy', 'epochs', 'artifact_size_bytes', 'created_at'];
	const asRecord = (value: unknown) => value && typeof value === 'object' ? value as RecordValue : {};
	const readable = (key: string) => key.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
	const value = (row: RecordValue, key: string): unknown => {
		if (key in row) return row[key];
		for (const group of ['metrics', 'runtime', 'model', 'training', 'metadata']) {
			const nested = asRecord(row[group]); if (key in nested) return nested[key];
		}
		return undefined;
	};
	const text = (item: unknown, key = '') => {
		if (item === null || item === undefined || item === '') return '—';
		if (key.endsWith('_bytes') && typeof item === 'number') { const units = ['B', 'KiB', 'MiB', 'GiB']; let value = item, unit = 0; while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit += 1; } return `${value.toLocaleString(undefined, { maximumFractionDigits: 1 })} ${units[unit]}`; }
		if (key.endsWith('_at') && typeof item === 'string') { const date = new Date(item); return Number.isNaN(date.valueOf()) ? item : date.toLocaleString(); }
		if (key.endsWith('_seconds') && typeof item === 'number') return `${(item / 60).toLocaleString(undefined, { maximumFractionDigits: 1 })} min`;
		if (Array.isArray(item)) return item.map((value) => typeof value === 'object' ? String(asRecord(value).name ?? asRecord(value).label ?? '') : String(value)).filter(Boolean).join(', ') || '—';
		return typeof item === 'number' ? Number(item).toLocaleString(undefined, { maximumFractionDigits: 4 }) : String(item);
	};

	let fields: Field[] = [];
	let rows: RecordValue[] = [];
	let visible = [...defaults];
	let search = '';
	let filterField = 'architecture';
	let filterValue = '';
	let filters: Record<string, unknown> = {};
	let sort: CatalogQuery['sort'] = { field: 'updated_at', direction: 'desc' };
	let selected = new Set<string>();
	let loading = true;
	let total = 0;
	let detail: RecordValue | null = null;
	let architecture: RecordValue | null = null;
	let history: RecordValue[] = [];
	let historyError = '';
	let historyLoading = false;
	let selectedHistoryMetric = '';
	let insight: { kind: 'metric' | 'field' | 'module' | 'model'; key: string } | null = null;
	let insightElement: HTMLElement | null = null;
	async function selectInsight(kind: 'metric' | 'field' | 'module' | 'model', key: string) {
		insight = { kind, key };
		await tick();
		insightElement?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
	}
	let detailLoading = false;
	let detailRequest = 0;
	let tagInput = '';
	let showingColumns = false;
	let reindexing = false;
	let reindexNotice = '';
	let inferenceOpen = false;
	let inferenceDatasets: RecordValue[] = [];
	let workerPoolsForInference: RecordValue[] = [];
	let inferenceDatasetId = '';
	let inferencePoolId = '';
	let inferenceName = '';
	let inferenceSplit = 'all';
	let inferenceRun: RecordValue | null = null;
	let inferenceBusy = false;
	let inferenceError = '';
	let posthocOpen = false;
	let posthocName = '';
	let posthocDatasetId = '';
	let posthocPoolId = '';
	let posthocValidationPercent = 20;
	let posthocTestPercent = 10;
	let posthocSplitStrategy = 'auto';
	let posthocRun: RecordValue | null = null;
	let posthocBusy = false;
	let posthocInputsLoading = false;
	let posthocError = '';
	$: frozenDatasetOptions = inferenceDatasets.filter((dataset) => String(dataset.lifecycle ?? dataset.status ?? '').toLowerCase() === 'frozen');
	$: inferencePoolOptions = workerPoolsForInference.filter((pool) => Array.isArray(pool.allowed_actions) && pool.allowed_actions.map(String).includes('infer'));
	$: posthocDatasetOptions = frozenDatasetOptions.filter((dataset) => String(dataset.dataset_type ?? '').toLowerCase() === 'classification');
	$: posthocPoolOptions = workerPoolsForInference.filter((pool) => Array.isArray(pool.allowed_actions) && pool.allowed_actions.map(String).includes('train') && pool.enabled !== false);

	function id(row: RecordValue) { return String(row.artifact_id ?? row.id); }
	function queryFilters() {
		const out = { ...filters } as Record<string, unknown>;
		if (search.trim()) out.search = search.trim();
		return out;
	}
	async function load() {
		loading = true;
		try {
			const result = await modelCatalogClient.artifactCatalogQuery({ filters: queryFilters(), sort, columns: visible, offset: 0, limit: 250 });
			rows = result.artifacts; total = result.total ?? rows.length;
		} catch (error) {
			// Existing installations can still browse artifacts before the catalog index migration runs.
			try { rows = (await api.artifactCatalog()).artifacts; total = rows.length; }
			catch { onfailure(error instanceof Error ? error.message : 'Could not load model runs.'); }
		} finally { loading = false; }
	}
	async function loadSchema() {
		try {
			const schema = await modelCatalogClient.artifactFilterSchema();
			fields = schema.fields.map((field) => ({ key: String(field.key ?? field.name), label: String(field.label ?? field.key ?? field.name), type: String(field.type ?? 'text') })).filter((field) => field.key && field.key !== 'undefined');
			if (schema.default_columns?.length) visible = schema.default_columns;
		} catch { fields = defaults.map((key) => ({ key, label: readable(key) })); }
	}
	function setSort(key: string) { sort = { field: key, direction: sort?.field === key && sort.direction === 'asc' ? 'desc' : 'asc' }; void load(); }
	function toggleColumn(key: string) { visible = visible.includes(key) ? visible.filter((item) => item !== key) : [...visible, key]; void load(); }
	function applyFilter() { filters = filterValue.trim() ? { ...filters, [filterField]: filterValue.trim() } : (() => { const next = { ...filters }; delete next[filterField]; return next; })(); void load(); }
	function removeFilter(key: string) { const next = { ...filters }; delete next[key]; filters = next; filterValue = ''; void load(); }
	function toggle(row: RecordValue) { const next = new Set(selected); next.has(id(row)) ? next.delete(id(row)) : next.add(id(row)); selected = next; }
	async function inspect(row: RecordValue) {
		const request = ++detailRequest;
		detailLoading = true; historyLoading = true; detail = row; architecture = null; history = []; historyError = ''; selectedHistoryMetric = ''; insight = null;
		try { const [model, graph] = await Promise.all([modelCatalogClient.artifactDetail(id(row)), modelCatalogClient.artifactArchitectureView(id(row))]); if (request !== detailRequest) return; detail = model; architecture = graph; }
		catch (error) { if (request === detailRequest) onfailure(error instanceof Error ? error.message : 'Could not load model details.'); }
		finally { if (request === detailRequest) detailLoading = false; }
		if (request !== detailRequest) return;
		try {
			const result = await api.artifactHistory(id(row), 2000);
			if (request === detailRequest) history = Array.isArray(result.rows) ? result.rows as RecordValue[] : [];
		} catch (error) { if (request === detailRequest) historyError = error instanceof Error ? error.message : 'Training history is unavailable.'; }
		finally { if (request === detailRequest) historyLoading = false; }
	}
	function closeDetail() { detailRequest += 1; detail = null; architecture = null; history = []; selectedHistoryMetric = ''; insight = null; historyLoading = false; posthocOpen = false; }
	async function addTags() {
		const tags = tagInput.split(',').map((tag) => tag.trim()).filter(Boolean);
		if (!tags.length || !selected.size) return;
		try { await modelCatalogClient.assignArtifactTags([...selected], tags); tagInput = ''; await load(); }
		catch (error) { onfailure(error instanceof Error ? error.message : 'Could not assign tags.'); }
	}
	async function refreshFacts() {
		reindexing = true; reindexNotice = '';
		try {
			const result = await modelCatalogClient.reindexArtifactCatalog(selected.size ? [...selected] : undefined);
			reindexNotice = `${result.refreshed.length} catalog ${result.refreshed.length === 1 ? 'record' : 'records'} refreshed from sealed evidence. Artifact files were not changed.`;
			await load();
		} catch (error) { onfailure(error instanceof Error ? error.message : 'Could not refresh catalog facts.'); }
		finally { reindexing = false; }
	}
	function inferenceId(run: RecordValue | null): string {
		const nested = asRecord(run?.inference_run ?? run?.run);
		return String(run?.inference_run_id ?? run?.run_id ?? run?.id ?? nested.inference_run_id ?? nested.run_id ?? nested.id ?? '');
	}
	function inferenceStatus(run: RecordValue | null): string { return String(run?.status ?? run?.state ?? asRecord(run?.inference_run ?? run?.run).status ?? 'submitted'); }
	async function openInference() {
		inferenceOpen = true; inferenceError = ''; inferenceRun = null;
		inferenceName = `Inference: ${text(detailArtifact().name ?? detailArtifact().artifact_id ?? 'model')}`;
		try {
			const [datasets, pools] = await Promise.all([api.datasets(), api.workerPools()]);
			inferenceDatasets = datasets.datasets; workerPoolsForInference = pools.pools;
			await tick();
			if (!inferenceDatasetId && frozenDatasetOptions.length) inferenceDatasetId = String(frozenDatasetOptions[0].dataset_id ?? frozenDatasetOptions[0].id);
			if (!inferencePoolId && inferencePoolOptions.length) inferencePoolId = String(inferencePoolOptions[0].pool_id ?? inferencePoolOptions[0].id);
		} catch (error) { inferenceError = error instanceof Error ? error.message : 'Could not load inference inputs.'; }
	}
	async function submitInference() {
		const modelArtifactId = String(detailArtifact().artifact_id ?? detail?.artifact_id ?? '');
		if (!modelArtifactId || !inferenceDatasetId || !inferencePoolId) { inferenceError = 'Choose a frozen dataset and an inference-capable worker pool.'; return; }
		inferenceBusy = true; inferenceError = '';
		try {
			const created = await api.createInferenceRun({ name: inferenceName, model_artifact_id: modelArtifactId, dataset_id: inferenceDatasetId, worker_pool_id: inferencePoolId, split: inferenceSplit });
			const runId = inferenceId(created); if (!runId) throw new Error('The Orchestrator did not return an inference run ID.');
			await api.startInferenceRun(runId); inferenceRun = await api.inferenceRun(runId);
		} catch (error) { inferenceError = error instanceof Error ? error.message : 'Could not submit inference.'; }
		finally { inferenceBusy = false; }
	}
	async function refreshInference() { const runId = inferenceId(inferenceRun); if (!runId) return; inferenceBusy = true; try { inferenceRun = await api.inferenceRun(runId); } catch (error) { inferenceError = error instanceof Error ? error.message : 'Could not refresh inference status.'; } finally { inferenceBusy = false; } }
	async function cancelInference() { const runId = inferenceId(inferenceRun); if (!runId) return; inferenceBusy = true; try { await api.cancelInferenceRun(runId); await refreshInference(); } catch (error) { inferenceError = error instanceof Error ? error.message : 'Could not cancel inference.'; } finally { inferenceBusy = false; } }
	async function openPosthoc() {
		posthocOpen = true; posthocError = ''; posthocRun = null;
		posthocName = `${text(detailArtifact().name ?? 'Model')} + post-hoc`;
		posthocInputsLoading = true;
		try {
			const [datasets, pools] = await Promise.all([api.datasets(), api.workerPools()]);
			inferenceDatasets = datasets.datasets; workerPoolsForInference = pools.pools;
			await tick();
			const sourceDatasetId = String(detailArtifact().dataset_id ?? '');
			posthocDatasetId = posthocDatasetOptions.some((dataset) => String(dataset.dataset_id) === sourceDatasetId)
				? sourceDatasetId : String(posthocDatasetOptions[0]?.dataset_id ?? '');
			posthocPoolId = String(posthocPoolOptions[0]?.pool_id ?? '');
		} catch (error) { posthocError = error instanceof Error ? error.message : 'Could not load fitting inputs.'; }
		finally { posthocInputsLoading = false; }
	}
	async function submitPosthoc() {
		const modelArtifactId = String(detailArtifact().artifact_id ?? detail?.artifact_id ?? '');
		if (!modelArtifactId || !posthocDatasetId || !posthocPoolId) { posthocError = 'Choose a frozen classification dataset and a training worker pool.'; return; }
		if (posthocValidationPercent <= 0 || posthocTestPercent <= 0 || posthocValidationPercent + posthocTestPercent >= 100) { posthocError = 'Validation and test percentages must be positive and total less than 100%.'; return; }
		posthocBusy = true; posthocError = '';
		try {
			const created = await api.createPosthocRun({ name: posthocName, model_artifact_id: modelArtifactId, dataset_id: posthocDatasetId, worker_pool_id: posthocPoolId, validation_split: posthocValidationPercent / 100, test_split: posthocTestPercent / 100, split_strategy: posthocSplitStrategy });
			const runId = String(created.posthoc_run.posthoc_run_id ?? '');
			if (!runId) throw new Error('The Orchestrator did not return a post-hoc run ID.');
			await api.startPosthocRun(runId);
			posthocRun = (await api.posthocRun(runId)).posthoc_run;
		} catch (error) { posthocError = error instanceof Error ? error.message : 'Could not start post-hoc fitting.'; }
		finally { posthocBusy = false; }
	}
	async function refreshPosthoc() {
		const runId = String(posthocRun?.posthoc_run_id ?? ''); if (!runId) return;
		posthocBusy = true;
		try { posthocRun = (await api.posthocRun(runId)).posthoc_run; if (posthocRun?.status === 'indexed') await load(); }
		catch (error) { posthocError = error instanceof Error ? error.message : 'Could not refresh fit status.'; }
		finally { posthocBusy = false; }
	}
	async function cancelPosthoc() {
		const runId = String(posthocRun?.posthoc_run_id ?? ''); if (!runId) return;
		posthocBusy = true;
		try { posthocRun = (await api.cancelPosthocRun(runId)).posthoc_run; }
		catch (error) { posthocError = error instanceof Error ? error.message : 'Could not cancel fitting.'; }
		finally { posthocBusy = false; }
	}
	function architectureModules(): unknown[] {
		const graph = asRecord(architecture?.architecture ?? architecture);
		return Array.isArray(graph.modules) ? graph.modules : ['Input', 'Encoder', 'Embedding', 'Classifier'];
	}
	function detailArtifact(): RecordValue { return asRecord(detail?.artifact); }
	function derivedLineage(): RecordValue { return asRecord(asRecord(detail?.manifest).lineage); }
	function sourceNeuralMetrics(): [string, unknown][] { return Object.entries(asRecord(detail?.source_neural_evaluation)).filter(([, item]) => typeof item === 'number'); }
	function detailFacts(): RecordValue { return asRecord(asRecord(detail?.facts).facts); }
	function detailValue(key: string): unknown { return value(detailFacts(), key) ?? value(detailArtifact(), key) ?? value(asRecord(detail), key); }
	function fieldSource(row: RecordValue, key: string): RecordValue { return asRecord(asRecord(row.field_sources)[key]); }
	function sourceTitle(row: RecordValue, key: string): string {
		const source = fieldSource(row, key);
		return source.status ? `${text(value(row, key), key)} · ${String(source.status).replaceAll('_', ' ')} from ${String(source.source).replaceAll('_', ' ')}` : text(value(row, key), key);
	}
	function detailSources(): [string, RecordValue][] { return Object.entries(asRecord(detailFacts().field_sources)).map(([key, source]) => [key, asRecord(source)]); }
	function detailRuntime(): unknown { return value(asRecord(detail), 'training_seconds') ?? value(asRecord(detailArtifact().runtime), 'training_time_seconds') ?? value(asRecord(detail), 'runtime_seconds'); }
	function modelInspection(): RecordValue { return asRecord(detail?.model_inspection); }
	function resolvedConfig(): RecordValue { return asRecord(detail?.config); }
	function modelConfig(): RecordValue { return asRecord(resolvedConfig().model); }
	function evaluationMetrics(): RecordValue { const facts = asRecord(detailFacts().metrics); return Object.keys(facts).length ? facts : asRecord(detailArtifact().metrics); }
	function metricEntries(): [string, unknown][] {
		const entries = new Map<string, unknown>(Object.entries(evaluationMetrics()));
		if (derivedLineage().source_model_artifact_id) return [...entries];
		for (const key of chartKeys()) {
			if (entries.has(key)) continue;
			const row = [...history].reverse().find((item) => typeof (item[`val_${key}`] ?? item[key]) === 'number' && Number.isFinite((item[`val_${key}`] ?? item[key]) as number));
			if (row) entries.set(key, row[`val_${key}`] ?? row[key]);
		}
		return [...entries];
	}
	function chartKeys(): string[] {
		const keys = new Set(history.flatMap((row) => Object.keys(row).filter((key) => !['epoch', 'step', 'batch'].includes(key) && history.some((item) => typeof item[key] === 'number' && Number.isFinite(item[key] as number))).map((key) => key.replace(/^val_/, ''))));
		const priority = (key: string) => { const index = ['loss', 'accuracy', 'macro_f1'].indexOf(key); return index < 0 ? 100 : index; };
		return [...keys].sort((left, right) => priority(left) - priority(right) || left.localeCompare(right));
	}
	function canonicalMetric(key: string): string {
		const name = key.replace(/^val_/, '');
		const match = name.match(/^(f1|precision|recall)_(micro|macro|weighted)$/);
		return match ? `${match[2]}_${match[1]}` : name;
	}
	function matchingChart(key: string): string | null { return derivedLineage().source_model_artifact_id ? null : chartKeys().find((item) => canonicalMetric(item) === canonicalMetric(key)) ?? null; }
	function moduleRecord(module: unknown): RecordValue { return typeof module === 'string' ? { id: module.toLowerCase(), label: module } : asRecord(module); }
	function selectedModule(): RecordValue { return moduleRecord(architectureModules().find((module) => String(moduleRecord(module).id) === insight?.key)); }
	function stemLayer(): RecordValue { const layers = Array.isArray(modelInspection().layers) ? modelInspection().layers as RecordValue[] : []; return layers.find((layer) => String(layer.name).includes('stem_conv')) ?? layers.find((layer) => String(layer.class_name) === 'Conv2D') ?? {}; }
	function moduleDetails(module: RecordValue): [string, unknown][] {
		const config = asRecord(module.config);
		if (module.id === 'stem') {
			const stem = asRecord(resolvedConfig().stem);
			const layer = stemLayer();
			return [['Kernel size', layer.kernel_size ?? stem.kernel_size ?? modelConfig().stem_kernel_size], ['Stride', layer.strides ?? stem.stride ?? modelConfig().stem_stride], ['Filters', layer.filters ?? stem.width ?? modelConfig().stem_filters ?? modelConfig().base_filters], ['Pooling', stem.pool ?? modelConfig().stem_pool]];
		}
		return Object.entries(config).filter(([, item]) => item !== null && item !== undefined).slice(0, 12);
	}
	function modelFacts(): [string, unknown][] {
		const encoder = asRecord(resolvedConfig().encoder);
		return ([['Architecture', detailArtifact().architecture ?? encoder.family ?? asRecord(resolvedConfig().run).model], ['Variant', detailArtifact().variant ?? encoder.variant ?? modelConfig().variant], ['Layers', modelInspection().layer_count ?? (Array.isArray(modelInspection().layers) ? (modelInspection().layers as unknown[]).length : null)], ['Model parameters', modelInspection().parameter_count], ['Trainable parameters', modelInspection().trainable_parameter_count], ['Non-trainable parameters', modelInspection().non_trainable_parameter_count], ['Optimizer state parameters', modelInspection().optimizer_parameter_count], ['Framework', modelInspection().framework]] as [string, unknown][]).filter(([, item]) => item !== null && item !== undefined);
	}
	function fieldContext(key: string): string {
		if (key === 'training_set') return 'Frozen dataset used to train this model. Its identity is recorded in the sealed configuration.';
		if (key === 'stem_size') return 'The stem is the first convolution stage. Open its architecture chip for kernel, stride, and filter details.';
		if (key === 'macro_f1') return 'Macro F1 gives every class equal weight when averaging class F1 scores.';
		if (key === 'loss') return 'The optimization objective. A lower value is usually better for the same task and protocol.';
		if (key === 'accuracy') return 'Fraction of correctly classified examples on the recorded evaluation split.';
		if (key === 'balanced_accuracy') return 'Average recall across classes. Each class contributes equally even when class sizes differ.';
		if (key === 'macro_precision') return 'Precision averaged equally across classes.';
		if (key === 'macro_recall') return 'Recall averaged equally across classes.';
		if (key === 'macro_average_precision') return 'Average precision from prediction scores, averaged equally across classes.';
		if (key === 'mean_dice') return 'Average overlap between predicted and reference masks, measured with the Dice coefficient.';
		if (key === 'mean_iou') return 'Average intersection over union between predicted and reference masks.';
		return 'This value comes from the artifact catalog. Its provenance is shown below when available.';
	}
	$: shownRows = rows.filter((row) => !search.trim() || JSON.stringify(row).toLowerCase().includes(search.toLowerCase()));
	onMount(() => {
		void Promise.all([loadSchema(), load()]);
		const timer = window.setInterval(() => {
			if (posthocOpen && posthocRun && !posthocBusy && !['indexed', 'failed', 'cancelled'].includes(String(posthocRun.status))) void refreshPosthoc();
		}, 3000);
		return () => window.clearInterval(timer);
	});
</script>

<section class="runs-hero">
	<div><p class="eyebrow">MODEL RUNS</p><h2>A clear record of every trained or imported model.</h2><p>Search results, compare candidates, and inspect the sealed configuration behind each score.</p></div>
	<div class="run-count"><strong>{total}</strong><span>indexed models</span></div>
</section>

<section class="runs-toolbar" aria-label="Model run controls">
	<label class="search"><span>Search models</span><input bind:value={search} on:keydown={(event) => event.key === 'Enter' && load()} placeholder="Name, architecture, dataset…" /></label>
	<label><span>Filter by</span><select bind:value={filterField}>{#each fields as field}<option value={field.key}>{field.label ?? readable(field.key)}</option>{/each}</select></label>
	<label><span>Value</span><input bind:value={filterValue} on:keydown={(event) => event.key === 'Enter' && applyFilter()} placeholder="Any value" /></label>
	<button class="secondary" on:click={applyFilter}>Apply filter</button>
	<div class="column-picker"><button class="secondary" aria-expanded={showingColumns} on:click={() => showingColumns = !showingColumns}>Columns ({visible.length})</button>{#if showingColumns}<div class="column-menu">{#each fields as field}<label><input type="checkbox" checked={visible.includes(field.key)} on:change={() => toggleColumn(field.key)} /> {field.label ?? readable(field.key)}</label>{/each}</div>{/if}</div>
	<button class="quiet-button" disabled={reindexing || loading} on:click={refreshFacts}>{reindexing ? 'Refreshing evidence…' : selected.size ? `Refresh facts (${selected.size})` : 'Refresh catalog facts'}</button>
</section>

{#if reindexNotice}<p class="catalog-note" role="status">{reindexNotice}</p>{/if}

{#if Object.keys(filters).length}<div class="filter-chips">{#each Object.entries(filters) as [key, item]}<button class="filter-chip" on:click={() => removeFilter(key)} title="Remove filter">{readable(key)}: {text(item)} ×</button>{/each}</div>{/if}

{#if selected.size}<section class="selection-bar"><strong>{selected.size} selected</strong><label><span>Tags (comma separated)</span><input bind:value={tagInput} placeholder="baseline, reviewed" /></label><button class="secondary" on:click={addTags}>Add tags</button><button disabled={selected.size < 2} on:click={() => oncompare([...selected])}>Compare selected</button><button class="quiet-button" on:click={() => selected = new Set()}>Clear</button></section>{/if}

	<section class="runs-table-wrap">
	{#if loading}<p class="empty">Loading indexed model runs…</p>{:else if !shownRows.length}<p class="empty">No models match these filters.</p>{:else}<table><thead><tr><th><input class="selection-box" type="checkbox" aria-label="Select all visible models" checked={shownRows.length > 0 && shownRows.every((row) => selected.has(id(row)))} on:change={() => selected = new Set(shownRows.every((row) => selected.has(id(row))) ? [] : shownRows.map(id))} /></th>{#each visible as column}<th><button class="sort-button" on:click={() => setSort(column)}>{readable(column)} {sort?.field === column ? (sort.direction === 'asc' ? '↑' : '↓') : ''}</button></th>{/each}<th>Inspect</th></tr></thead><tbody>{#each shownRows as row}<tr class:selected={selected.has(id(row))}><td><input class="selection-box" type="checkbox" aria-label={`Select ${text(value(row, 'name') ?? id(row))}`} checked={selected.has(id(row))} on:change={() => toggle(row)} /></td>{#each visible as column}<td title={sourceTitle(row, column)}>{#if column === 'status' || column === 'lifecycle'}<span class="status {String(value(row, column))}">{text(value(row, column), column)}</span>{:else}{text(value(row, column), column)}{/if}</td>{/each}<td><button class="secondary small" on:click={() => inspect(row)}>Details</button></td></tr>{/each}</tbody></table>{/if}
</section>

{#if detail}<div class="modal-backdrop" role="presentation" on:click={(event) => event.target === event.currentTarget && closeDetail()}><div class="model-modal" role="dialog" aria-modal="true" aria-label="Model details" tabindex="-1"><header><div><p class="eyebrow">SEALED MODEL</p><h2>{text(detailArtifact().name ?? detail.name ?? detailArtifact().artifact_id ?? detail.artifact_id)}</h2><p>{text(detailArtifact().architecture ?? detail.architecture)} · {text(detailArtifact().task ?? detail.task)}</p></div><div class="modal-actions">{#if detailArtifact().artifact_id ?? detail.artifact_id}<a class="secondary small" href={api.artifactDownloadUrl(String(detailArtifact().artifact_id ?? detail.artifact_id))}>Download archive</a>{#if String(detailArtifact().task ?? detail.task) === 'classification'}<button class="secondary small" on:click={openPosthoc}>Fit post-hoc head</button>{/if}<button class="secondary small" on:click={openInference}>Run inference</button>{/if}<button class="icon-button" aria-label="Close" on:click={closeDetail}>×</button></div></header>{#if detailLoading}<p class="empty">Gathering architecture and provenance…</p>{:else}<div class="detail-grid">
				<article><h3>{detail.posthoc && derivedLineage().source_model_artifact_id ? 'Post-hoc performance' : 'Performance'}</h3><p class="inspect-hint">{detail.posthoc && derivedLineage().source_model_artifact_id ? 'Held-out test scores for the new head on its selected dataset.' : 'Select a value to inspect its source and training history.'}</p>{#each metricEntries() as [key, metric]}<button class="key-value evidence-button" class:active={insight?.kind === 'metric' && insight.key === key} on:click={() => selectInsight('metric', key)}><span>{readable(key)}</span><strong>{text(metric)}</strong></button>{:else}<p class="empty">No evaluation metrics were recorded for this artifact.</p>{/each}</article>
				<article><h3>Training & data</h3><div class="key-value"><span>{derivedLineage().source_model_artifact_id ? 'Post-hoc dataset' : 'Dataset'}</span><strong>{text(detailValue('training_set') ?? detailArtifact().dataset_id)}</strong></div>{#if derivedLineage().source_model_artifact_id}<div class="key-value"><span>Source neural dataset</span><strong>{text(derivedLineage().source_training_dataset_id)}</strong></div><div class="key-value"><span>Source model</span><strong>{text(derivedLineage().source_model_artifact_id)}</strong></div>{/if}<div class="key-value"><span>Runtime</span><strong>{text(detailRuntime())}</strong></div><div class="key-value"><span>Tags</span><strong>{text(detailArtifact().tags ?? detail.tags)}</strong></div></article>
			</div>
			<article class="module-map"><h3>Architecture modules</h3><p class="inspect-hint">Select the model or a stage to see recorded dimensions and settings.</p><div class="module-track"><button class:active={insight?.kind === 'model'} on:click={() => selectInsight('model', 'model')}>Model overview</button>{#each architectureModules() as module}<button class:active={insight?.kind === 'module' && insight.key === moduleRecord(module).id} on:click={() => selectInsight('module', String(moduleRecord(module).id))}>{text(moduleRecord(module).label ?? moduleRecord(module).name ?? moduleRecord(module).type)}</button>{/each}</div></article>
			{#if detailSources().length}<article class="field-evidence"><h3>Catalog field evidence</h3><p>Recorded values come from the artifact; inferred values are catalog fallbacks. Select a field for context.</p><div>{#each detailSources() as [key, source]}<button class:active={insight?.kind === 'field' && insight.key === key} on:click={() => selectInsight('field', key)}><strong>{readable(key)}</strong><small class:inferred={source.status === 'inferred'}>{text(detailValue(key))} · {text(source.status).replaceAll('_', ' ')} from {text(source.source).replaceAll('_', ' ')}</small></button>{/each}</div></article>{/if}
			{#if insight}<article class="insight-panel" bind:this={insightElement} aria-live="polite"><div class="insight-head"><div><p class="eyebrow">INSPECTION DETAIL</p><h3>{insight.kind === 'model' ? 'Model overview' : insight.kind === 'module' ? text(selectedModule().label ?? insight.key) : readable(insight.key)}</h3></div><button class="secondary small" on:click={() => insight = null}>Close detail</button></div>
				{#if insight.kind === 'model'}<div class="insight-facts">{#each modelFacts() as [label, item]}<div><span>{label}</span><strong>{text(item)}</strong></div>{/each}</div>{#if !modelInspection().parameter_count}<p class="inspect-hint">Detailed parameter counts were not recorded for this artifact.</p>{/if}
				{:else if insight.kind === 'module'}<p class="inspect-hint">Configuration and layer details recorded for this architecture stage.</p><div class="insight-facts">{#each moduleDetails(selectedModule()) as [label, item]}<div><span>{readable(label)}</span><strong>{typeof item === 'object' && item !== null ? JSON.stringify(item) : text(item)}</strong></div>{:else}<p>No module-specific values were recorded.</p>{/each}</div>{#if Object.keys(asRecord(selectedModule().config)).length}<details><summary>Full module configuration</summary><pre>{JSON.stringify(selectedModule().config, null, 2)}</pre></details>{/if}
				{:else}<p>{fieldContext(insight.key)}</p>{#if insight.kind === 'metric'}<p class="inspect-hint">{insight.key in evaluationMetrics() ? 'Held-out evaluation result' : 'Latest recorded epoch value'}: <strong>{text(metricEntries().find(([key]) => key === insight?.key)?.[1])}</strong>{insight.key in evaluationMetrics() ? ` · ${text(asRecord(detailArtifact().protocol).split)} split.` : ''}</p>{:else}<p class="inspect-hint">Value: <strong>{text(detailValue(insight.key))}</strong> · {text(fieldSource(detailFacts(), insight.key).status)} from {text(fieldSource(detailFacts(), insight.key).source)}.</p>{/if}{#if matchingChart(insight.key)}<div class="insight-chart"><PairedMetricPlot points={history} metric={matchingChart(insight.key) ?? ''} title={readable(insight.key)} mode="epoch" /></div><p class="inspect-hint">Training and validation are recorded per epoch; held-out evaluation is shown separately when available.</p>{:else}<p class="inspect-hint">No epoch series was recorded for this field.</p>{/if}{/if}</article>{/if}
			<article class="training-charts"><div><h3>{detail.history_provenance ? 'Source neural training history' : 'Training history'}</h3><p class="inspect-hint">{detail.history_provenance ? `These epoch metrics belong to source model ${text(derivedLineage().source_model_artifact_id)} on dataset ${text(derivedLineage().source_training_dataset_id)}. The post-hoc head was fitted separately.` : 'The same epoch metrics used by the live dashboard, preserved in this artifact.'}</p></div>{#if historyLoading}<p class="empty">Loading recorded epoch metrics…</p>{:else if historyError}<p class="empty">{historyError}</p>{:else if chartKeys().length}<label class="history-metric-picker"><span>Plot a metric</span><select bind:value={selectedHistoryMetric}><option value="">Choose any recorded metric</option>{#each chartKeys() as key}<option value={key}>{readable(key)}</option>{/each}</select></label>{#if selectedHistoryMetric}<div class="training-chart"><h4>{readable(selectedHistoryMetric)}</h4><PairedMetricPlot points={history} metric={selectedHistoryMetric} title={readable(selectedHistoryMetric)} mode="epoch" /></div>{/if}<div class="training-chart-grid">{#each ['loss', 'accuracy', 'macro_f1'].filter((key) => chartKeys().includes(key)) as key}<div class="training-chart"><h4>{readable(key)}</h4><PairedMetricPlot points={history} metric={key} title={readable(key)} mode="epoch" /></div>{/each}</div>{#if chartKeys().some((key) => !['loss', 'accuracy', 'macro_f1'].includes(key))}<details><summary>Additional training metrics</summary><div class="training-chart-grid">{#each chartKeys().filter((key) => !['loss', 'accuracy', 'macro_f1'].includes(key)) as key}<div class="training-chart"><h4>{readable(key)}</h4><PairedMetricPlot points={history} metric={key} title={readable(key)} mode="epoch" /></div>{/each}</div></details>{/if}{:else}<p class="empty">No epoch history was recorded for this artifact.</p>{/if}</article>
			{#if sourceNeuralMetrics().length}<article class="insight-panel"><h3>Original neural evaluation</h3><p class="inspect-hint">Preserved from the source model and its original dataset. These scores are separate from the new post-hoc test scores above.</p><div class="insight-facts">{#each sourceNeuralMetrics() as [key, item]}<div><span>{readable(key)}</span><strong>{text(item)}</strong></div>{/each}</div></article>{/if}
			{#if detail.posthoc}<article class="insight-panel"><h3>Post-hoc classifier</h3><p class="inspect-hint">Selected on validation embeddings and scored on the held-out test split. The neural model was not retrained.</p><div class="insight-facts"><div><span>Estimator</span><strong>{text(asRecord(detail.posthoc).estimator)}</strong></div><div><span>Representation</span><strong>{text(asRecord(detail.posthoc).representation)}</strong></div><div><span>Selected C</span><strong>{text(asRecord(detail.posthoc).selected_C)}</strong></div>{#each Object.entries(asRecord(asRecord(detail.posthoc).validation_metrics)) as [key, item]}<div><span>Validation {readable(key)}</span><strong>{text(item)}</strong></div>{/each}{#each Object.entries(asRecord(asRecord(detail.posthoc).metrics)) as [key, item]}<div><span>Test {readable(key)}</span><strong>{text(item)}</strong></div>{/each}</div></article>{/if}
			<details><summary>Complete sealed metadata</summary><pre>{JSON.stringify(detail, null, 2)}</pre></details>{/if}
	{#if posthocOpen}<section class="inference-panel"><div class="panel-head"><div><p class="eyebrow">POST-HOC CLASSIFIER</p><h3>Fit a head to this model</h3><p>Reuse the sealed neural model. A worker extracts embeddings from the chosen frozen dataset, selects logistic regression strength on validation data, and publishes a separate model product. Its original training history is retained; new test scores belong to the fitted head.</p></div><button class="icon-button" aria-label="Close post-hoc form" on:click={() => posthocOpen = false}>×</button></div>{#if posthocError}<p class="inline-error" role="alert">{posthocError}</p>{/if}{#if posthocInputsLoading}<p class="inspect-hint" role="status">Loading frozen datasets and training pools…</p>{:else if !posthocDatasetOptions.length || !posthocPoolOptions.length}<p class="inspect-hint" role="status">{!posthocDatasetOptions.length ? "No frozen classification datasets are registered." : "No training worker pool is available."}</p>{/if}{#if !posthocRun}<div class="inference-form"><label><span>New model name</span><input bind:value={posthocName} /></label><label><span>Frozen classification dataset</span><select bind:value={posthocDatasetId}><option value="">Choose a dataset</option>{#each posthocDatasetOptions as dataset}<option value={String(dataset.dataset_id ?? dataset.id)}>{text(dataset.name ?? dataset.dataset_id ?? dataset.id)}</option>{/each}</select></label><label><span>Training worker pool</span><select bind:value={posthocPoolId}><option value="">Choose a pool</option>{#each posthocPoolOptions as pool}<option value={String(pool.pool_id ?? pool.id)}>{text(pool.name ?? pool.pool_id ?? pool.id)}</option>{/each}</select></label><label><span>Split strategy</span><select bind:value={posthocSplitStrategy}><option value="auto">Auto: respect source partitions</option><option value="class_stratified">Class stratified</option><option value="source_partitions">Source partitions only</option></select></label><label><span>Validation %</span><input type="number" min="1" max="98" bind:value={posthocValidationPercent} /></label><label><span>Test %</span><input type="number" min="1" max="98" bind:value={posthocTestPercent} /></label></div><button disabled={posthocBusy || posthocInputsLoading || !posthocDatasetOptions.length || !posthocPoolOptions.length} on:click={submitPosthoc}>{posthocBusy ? 'Submitting…' : 'Fit post-hoc head'}</button>{:else}<div class="inference-result"><div><span>Status</span><strong class="status {String(posthocRun.status)}">{text(posthocRun.status)}</strong></div><div><span>Run</span><strong>{text(posthocRun.posthoc_run_id)}</strong></div>{#if posthocRun.output_artifact_id}<div><span>New model</span><strong>{text(posthocRun.output_artifact_id)}</strong></div>{/if}<div class="modal-actions"><button class="secondary small" disabled={posthocBusy} on:click={refreshPosthoc}>Refresh</button>{#if posthocRun.output_artifact_id && posthocRun.status === 'indexed'}<button class="secondary small" on:click={() => { posthocOpen = false; inspect({ artifact_id: posthocRun?.output_artifact_id }); }}>Inspect new model</button>{/if}{#if !['indexed', 'completed', 'failed', 'cancelled'].includes(String(posthocRun.status))}<button class="secondary small" disabled={posthocBusy} on:click={cancelPosthoc}>Cancel</button>{/if}</div></div>{/if}</section>{/if}
	{#if inferenceOpen}<section class="inference-panel"><div class="panel-head"><div><p class="eyebrow">ORCHESTRATED BATCH INFERENCE</p><h3>Run this sealed model</h3><p>The Orchestrator validates and leases this request to an admitted worker; no browser connects to a worker.</p></div><button class="icon-button" aria-label="Close inference form" on:click={() => inferenceOpen = false}>×</button></div>{#if inferenceError}<p class="inline-error" role="alert">{inferenceError}</p>{/if}{#if !inferenceRun}<div class="inference-form"><label><span>Name</span><input bind:value={inferenceName} /></label><label><span>Frozen dataset</span><select bind:value={inferenceDatasetId}><option value="">Choose frozen dataset</option>{#each frozenDatasetOptions as dataset}<option value={String(dataset.dataset_id ?? dataset.id)}>{text(dataset.name ?? dataset.dataset_id ?? dataset.id)}</option>{/each}</select></label><label><span>Inference worker pool</span><select bind:value={inferencePoolId}><option value="">Choose inference pool</option>{#each inferencePoolOptions as pool}<option value={String(pool.pool_id ?? pool.id)}>{text(pool.name ?? pool.pool_id ?? pool.id)}</option>{/each}</select></label><label><span>Dataset split</span><select bind:value={inferenceSplit}><option value="all">All records</option><option value="train">Train</option><option value="validation">Validation</option><option value="test">Test</option></select></label></div><button disabled={inferenceBusy} on:click={submitInference}>{inferenceBusy ? 'Submitting…' : 'Queue inference'}</button>{:else}<div class="inference-result"><div><span>Status</span><strong class="status {inferenceStatus(inferenceRun)}">{inferenceStatus(inferenceRun)}</strong></div><div><span>Run</span><strong>{inferenceId(inferenceRun)}</strong></div><div class="modal-actions"><button class="secondary small" disabled={inferenceBusy} on:click={refreshInference}>Refresh</button>{#if !['completed', 'published', 'failed', 'cancelled', 'canceled'].includes(inferenceStatus(inferenceRun).toLowerCase())}<button class="secondary small" disabled={inferenceBusy} on:click={cancelInference}>Cancel</button>{/if}<a class="secondary small" href={api.inferenceDownloadUrl(inferenceId(inferenceRun))}>Download results</a></div></div>{/if}</section>{/if}</div></div>{/if}

<style>
	.inference-panel{margin-top:1rem;padding:1rem;border:1px solid #bdebdc;border-radius:9px;background:#f4fbf8}.inference-panel .panel-head{display:flex;justify-content:space-between;gap:1rem}.inference-panel h3{margin:.1rem 0}.inference-panel p{margin:.25rem 0;color:var(--muted);font-size:.8rem}.inference-form{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.7rem;margin:.8rem 0}.inference-form label{display:grid;gap:.25rem}.inference-form span,.inference-result span{font-size:.67rem;color:var(--muted);font-weight:700;text-transform:uppercase}.inference-result{display:flex;flex-wrap:wrap;align-items:end;gap:1rem;margin-top:.8rem}.inference-result>div{display:grid;gap:.2rem}.inference-result .modal-actions{display:flex;align-items:center;margin-left:auto}.inference-result a{text-decoration:none}@media(max-width:650px){.inference-form{grid-template-columns:1fr}.inference-result .modal-actions{margin-left:0}}
	.runs-hero { display:flex; justify-content:space-between; gap:2rem; margin:0 0 1.2rem; padding:1.8rem; color:#dceced; background:linear-gradient(120deg,#102a35,#1d4c55); border-radius:14px; }.runs-hero h2{max-width:650px;margin:.1rem 0 .45rem;color:white;font-size:clamp(1.45rem,3vw,2.25rem)}.runs-hero p{max-width:680px;margin:0;color:#b7d1d2}.run-count{display:grid;place-content:center;min-width:130px;text-align:center}.run-count strong{font-size:2rem}.run-count span{font-size:.72rem;color:#9fc6c4}.runs-toolbar,.selection-bar{display:flex;flex-wrap:wrap;align-items:end;gap:.65rem;margin-bottom:.75rem;padding:1rem;background:#fff;border:1px solid var(--line);border-radius:10px}.runs-toolbar label,.selection-bar label{gap:.25rem;min-width:130px}.runs-toolbar label span,.selection-bar label span{font-size:.67rem;color:var(--muted);font-weight:700;text-transform:uppercase}.runs-toolbar .search{flex:1;min-width:220px}.column-picker{position:relative}.column-menu{position:absolute;z-index:4;right:0;top:calc(100% + .4rem);display:grid;gap:.35rem;min-width:220px;max-height:300px;overflow:auto;padding:.7rem;background:white;border:1px solid var(--line);border-radius:8px;box-shadow:var(--shadow)}.column-menu label{display:flex;align-items:center;gap:.45rem;font-size:.78rem;font-weight:500}.column-menu input{width:auto;min-height:0}.filter-chips{display:flex;flex-wrap:wrap;gap:.4rem;margin-bottom:.75rem}.filter-chip{padding:.35rem .55rem;color:#17675d;background:var(--teal-soft);font-size:.7rem}.catalog-note{margin:-.25rem 0 .75rem;padding:.55rem .7rem;color:#17675d;background:#edfdf8;border:1px solid #bdebdc;border-radius:7px;font-size:.75rem}.selection-bar{position:sticky;top:.5rem;z-index:3;align-items:center}.selection-bar label{flex:1;min-width:180px}.selection-bar input{min-height:2.15rem}.runs-table-wrap{overflow:auto;background:white;border:1px solid var(--line);border-radius:10px;box-shadow:var(--shadow)}.runs-table-wrap table{min-width:850px}.sort-button{padding:0;color:inherit;background:transparent;font-size:inherit;letter-spacing:inherit;text-transform:inherit}.sort-button:hover{color:var(--teal);background:transparent}.runs-table-wrap tr.selected{background:#edf8f5}.modal-backdrop{position:fixed;z-index:20;inset:0;display:grid;place-items:center;padding:1rem;background:rgba(8,29,35,.56)}.model-modal{width:min(850px,100%);max-height:calc(100vh - 2rem);overflow:auto;padding:1.35rem;background:white;border-radius:14px;box-shadow:0 20px 65px rgba(0,0,0,.25)}.model-modal header,.modal-actions{display:flex;align-items:flex-start;justify-content:space-between;gap:1rem}.modal-actions{align-items:center}.modal-actions a{text-decoration:none}.model-modal header h2{margin:.1rem 0}.detail-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.8rem;margin:1rem 0}.detail-grid article,.module-map,.field-evidence{padding:1rem;border:1px solid var(--line);border-radius:9px;background:#f9fbfa}.key-value{display:flex;justify-content:space-between;gap:1rem;padding:.42rem 0;border-bottom:1px solid #e4ebe9;font-size:.78rem}.key-value span{color:var(--muted)}.module-track{display:flex;flex-wrap:wrap;align-items:center;gap:.45rem}.field-evidence{margin-top:.8rem}.field-evidence h3{margin:0 0 .25rem}.field-evidence p{margin:.25rem 0 .7rem;color:var(--muted);font-size:.74rem}.field-evidence>div{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:.45rem}.field-evidence strong{font-size:.68rem}.field-evidence small{color:#17675d;font-size:.65rem}.field-evidence small.inferred{color:#9a5a12}.model-modal details{margin-top:1rem}.model-modal pre{max-height:260px;overflow:auto;padding:.75rem;background:#102a35;color:#dceced;border-radius:7px;font-size:.7rem}@media(max-width:650px){.runs-hero{padding:1.25rem}.run-count{display:none}.detail-grid{grid-template-columns:1fr}.runs-toolbar{align-items:stretch}.runs-toolbar>*{width:100%}}
	.model-modal{width:min(1080px,100%)}
	.history-metric-picker{display:flex;align-items:center;gap:.7rem;margin:.6rem 0 1rem;font-size:.76rem;font-weight:700}.history-metric-picker select{max-width:310px}
	.detail-grid article:first-child{max-height:370px;overflow:auto}
	.field-evidence>div{max-height:250px;overflow:auto;align-content:start}
	.inspect-hint{margin:.3rem 0 .7rem;color:var(--muted);font-size:.75rem;line-height:1.45}
	.evidence-button{width:100%;align-items:center;text-align:left;background:transparent;color:var(--ink);border:0;border-bottom:1px solid #e4ebe9;border-radius:0;font-weight:inherit}
	.evidence-button:hover,.evidence-button.active{background:#edf8f5;color:#17675d}
	.evidence-button strong{color:var(--ink)}
	.module-track button{padding:.4rem .65rem;color:#17675d;background:#e7f5f1;border:1px solid transparent;border-radius:99px;font-size:.72rem;font-weight:700}
	.module-track button:hover,.module-track button.active{background:#d2eee4;border-color:#76bba8}
	.field-evidence>div>button{display:grid;gap:.12rem;padding:.5rem;text-align:left;background:#fff;color:var(--ink);border:1px solid #e4ebe9;border-radius:6px}
	.field-evidence>div>button:hover,.field-evidence>div>button.active{background:#edf8f5;border-color:#76bba8}
	.insight-panel,.training-charts{margin-top:.8rem;padding:1rem;border:1px solid var(--line);border-radius:9px;background:#f9fbfa}
	.insight-head{display:flex;justify-content:space-between;align-items:start;gap:1rem}
	.insight-head h3,.training-charts h3{margin:.1rem 0 .35rem}
	.insight-panel>p{font-size:.78rem;line-height:1.5}
	.insight-facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:.5rem;margin:.75rem 0}
	.insight-facts>div{display:grid;gap:.2rem;padding:.65rem;background:white;border:1px solid #e4ebe9;border-radius:7px;min-width:0}
	.insight-facts span{font-size:.67rem;font-weight:700;color:var(--muted);text-transform:uppercase}
	.insight-facts strong{overflow-wrap:anywhere;font-size:.82rem}
	.insight-chart{max-width:680px;margin:.8rem auto}
	.training-chart-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.7rem;margin-top:.8rem}
	.training-chart{min-width:0;padding:.8rem;background:white;border:1px solid #e4ebe9;border-radius:8px}
	.training-chart h4{margin:0 0 .4rem;font-size:.8rem}
	@media(max-width:650px){.training-chart-grid{grid-template-columns:1fr}.insight-head{flex-wrap:wrap}}
</style>
