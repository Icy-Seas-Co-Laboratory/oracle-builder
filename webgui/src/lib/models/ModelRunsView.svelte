<script lang="ts">
	import { onMount } from 'svelte';
	import { api, type RecordValue } from '$lib/api';
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
	let detailLoading = false;
	let tagInput = '';
	let showingColumns = false;
	let reindexing = false;
	let reindexNotice = '';

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
		detailLoading = true; detail = row; architecture = null;
		try { const [model, graph] = await Promise.all([modelCatalogClient.artifactDetail(id(row)), modelCatalogClient.artifactArchitectureView(id(row))]); detail = model; architecture = graph; }
		catch (error) { onfailure(error instanceof Error ? error.message : 'Could not load model details.'); }
		finally { detailLoading = false; }
	}
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
	function architectureModules(): unknown[] {
		const graph = asRecord(architecture?.architecture ?? architecture);
		return Array.isArray(graph.modules) ? graph.modules : ['Input', 'Encoder', 'Embedding', 'Classifier'];
	}
	function detailArtifact(): RecordValue { return asRecord(detail?.artifact); }
	function detailFacts(): RecordValue { return asRecord(asRecord(detail?.facts).facts); }
	function detailValue(key: string): unknown { return value(detailFacts(), key) ?? value(detailArtifact(), key) ?? value(asRecord(detail), key); }
	function fieldSource(row: RecordValue, key: string): RecordValue { return asRecord(asRecord(row.field_sources)[key]); }
	function sourceTitle(row: RecordValue, key: string): string {
		const source = fieldSource(row, key);
		return source.status ? `${text(value(row, key), key)} · ${String(source.status).replaceAll('_', ' ')} from ${String(source.source).replaceAll('_', ' ')}` : text(value(row, key), key);
	}
	function detailSources(): [string, RecordValue][] { return Object.entries(asRecord(detailFacts().field_sources)).map(([key, source]) => [key, asRecord(source)]); }
	function detailRuntime(): unknown { return value(asRecord(detail), 'training_seconds') ?? value(asRecord(detailArtifact().runtime), 'training_time_seconds') ?? value(asRecord(detail), 'runtime_seconds'); }
	$: shownRows = rows.filter((row) => !search.trim() || JSON.stringify(row).toLowerCase().includes(search.toLowerCase()));
	onMount(async () => { await Promise.all([loadSchema(), load()]); });
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

{#if detail}<div class="modal-backdrop" role="presentation" on:click={(event) => event.target === event.currentTarget && (detail = null)}><div class="model-modal" role="dialog" aria-modal="true" aria-label="Model details" tabindex="-1"><header><div><p class="eyebrow">SEALED MODEL</p><h2>{text(detailArtifact().name ?? detail.name ?? detailArtifact().artifact_id ?? detail.artifact_id)}</h2><p>{text(detailArtifact().architecture ?? detail.architecture)} · {text(detailArtifact().task ?? detail.task)}</p></div><button class="icon-button" aria-label="Close" on:click={() => detail = null}>×</button></header>{#if detailLoading}<p class="empty">Gathering architecture and provenance…</p>{:else}<div class="detail-grid"><article><h3>Performance</h3>{#each Object.entries(asRecord(detailFacts().metrics)) as [key, metric]}<div class="key-value"><span>{readable(key)}</span><strong>{text(metric)}</strong></div>{:else}<p class="empty">No evaluation metrics were recorded for this artifact.</p>{/each}</article><article><h3>Training & data</h3><div class="key-value"><span>Dataset</span><strong>{text(detailValue('training_set') ?? detailArtifact().dataset_id)}</strong></div><div class="key-value"><span>Runtime</span><strong>{text(detailRuntime())}</strong></div><div class="key-value"><span>Tags</span><strong>{text(detailArtifact().tags ?? detail.tags)}</strong></div></article></div><article class="module-map"><h3>Architecture modules</h3><div class="module-track">{#each architectureModules() as module}<span>{typeof module === 'string' ? module : text(asRecord(module).label ?? asRecord(module).name ?? asRecord(module).type)}</span>{/each}</div></article>{#if detailSources().length}<article class="field-evidence"><h3>Catalog field evidence</h3><p>Values marked <em>inferred</em> are catalog fallbacks, not a replacement for sealed evaluation evidence.</p><div>{#each detailSources() as [key, source]}<span><strong>{readable(key)}</strong><small class:inferred={source.status === 'inferred'}>{text(source.status).replaceAll('_', ' ')} · {text(source.source).replaceAll('_', ' ')}</small></span>{/each}</div></article>{/if}<details><summary>Complete sealed metadata</summary><pre>{JSON.stringify(detail, null, 2)}</pre></details>{/if}</div></div>{/if}

<style>
	.runs-hero { display:flex; justify-content:space-between; gap:2rem; margin:0 0 1.2rem; padding:1.8rem; color:#dceced; background:linear-gradient(120deg,#102a35,#1d4c55); border-radius:14px; }.runs-hero h2{max-width:650px;margin:.1rem 0 .45rem;color:white;font-size:clamp(1.45rem,3vw,2.25rem)}.runs-hero p{max-width:680px;margin:0;color:#b7d1d2}.run-count{display:grid;place-content:center;min-width:130px;text-align:center}.run-count strong{font-size:2rem}.run-count span{font-size:.72rem;color:#9fc6c4}.runs-toolbar,.selection-bar{display:flex;flex-wrap:wrap;align-items:end;gap:.65rem;margin-bottom:.75rem;padding:1rem;background:#fff;border:1px solid var(--line);border-radius:10px}.runs-toolbar label,.selection-bar label{gap:.25rem;min-width:130px}.runs-toolbar label span,.selection-bar label span{font-size:.67rem;color:var(--muted);font-weight:700;text-transform:uppercase}.runs-toolbar .search{flex:1;min-width:220px}.column-picker{position:relative}.column-menu{position:absolute;z-index:4;right:0;top:calc(100% + .4rem);display:grid;gap:.35rem;min-width:220px;max-height:300px;overflow:auto;padding:.7rem;background:white;border:1px solid var(--line);border-radius:8px;box-shadow:var(--shadow)}.column-menu label{display:flex;align-items:center;gap:.45rem;font-size:.78rem;font-weight:500}.column-menu input{width:auto;min-height:0}.filter-chips{display:flex;flex-wrap:wrap;gap:.4rem;margin-bottom:.75rem}.filter-chip{padding:.35rem .55rem;color:#17675d;background:var(--teal-soft);font-size:.7rem}.catalog-note{margin:-.25rem 0 .75rem;padding:.55rem .7rem;color:#17675d;background:#edfdf8;border:1px solid #bdebdc;border-radius:7px;font-size:.75rem}.selection-bar{position:sticky;top:.5rem;z-index:3;align-items:center}.selection-bar label{flex:1;min-width:180px}.selection-bar input{min-height:2.15rem}.runs-table-wrap{overflow:auto;background:white;border:1px solid var(--line);border-radius:10px;box-shadow:var(--shadow)}.runs-table-wrap table{min-width:850px}.sort-button{padding:0;color:inherit;background:transparent;font-size:inherit;letter-spacing:inherit;text-transform:inherit}.sort-button:hover{color:var(--teal);background:transparent}.runs-table-wrap tr.selected{background:#edf8f5}.modal-backdrop{position:fixed;z-index:20;inset:0;display:grid;place-items:center;padding:1rem;background:rgba(8,29,35,.56)}.model-modal{width:min(850px,100%);max-height:calc(100vh - 2rem);overflow:auto;padding:1.35rem;background:white;border-radius:14px;box-shadow:0 20px 65px rgba(0,0,0,.25)}.model-modal header{display:flex;justify-content:space-between;gap:1rem}.model-modal header h2{margin:.1rem 0}.detail-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.8rem;margin:1rem 0}.detail-grid article,.module-map,.field-evidence{padding:1rem;border:1px solid var(--line);border-radius:9px;background:#f9fbfa}.key-value{display:flex;justify-content:space-between;gap:1rem;padding:.42rem 0;border-bottom:1px solid #e4ebe9;font-size:.78rem}.key-value span{color:var(--muted)}.module-track{display:flex;flex-wrap:wrap;align-items:center;gap:.45rem}.module-track span{padding:.4rem .6rem;color:#17675d;background:#e7f5f1;border-radius:99px;font-size:.72rem;font-weight:700}.module-track span+span:before{content:'→';margin-right:.6rem;color:#789}.field-evidence{margin-top:.8rem}.field-evidence h3{margin:0 0 .25rem}.field-evidence p{margin:.25rem 0 .7rem;color:var(--muted);font-size:.74rem}.field-evidence>div{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:.45rem}.field-evidence span{display:grid;gap:.12rem;padding:.45rem;background:#fff;border:1px solid #e4ebe9;border-radius:6px}.field-evidence strong{font-size:.68rem}.field-evidence small{color:#17675d;font-size:.65rem}.field-evidence small.inferred{color:#9a5a12}.model-modal details{margin-top:1rem}.model-modal pre{max-height:260px;overflow:auto;padding:.75rem;background:#102a35;color:#dceced;border-radius:7px;font-size:.7rem}@media(max-width:650px){.runs-hero{padding:1.25rem}.run-count{display:none}.detail-grid{grid-template-columns:1fr}.runs-toolbar{align-items:stretch}.runs-toolbar>*{width:100%}}
</style>
