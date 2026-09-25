<script lang="ts">
	import { onMount } from 'svelte';
	import { type RecordValue } from '$lib/api';
	import { modelCatalogClient } from './catalog-client';

	export let artifactIds: string[] = [];
	export let onback: () => void = () => {};
	export let onfailure: (message: string) => void = () => {};
	const object = (value: unknown) => value && typeof value === 'object' ? value as RecordValue : {};
	const array = (value: unknown) => Array.isArray(value) ? value as RecordValue[] : [];
	const text = (value: unknown) => value === null || value === undefined || value === '' ? '—' : String(value);
	const scalar = (row: RecordValue, key: string): unknown => row[key] ?? object(row.artifact)[key] ?? object(row.facts)[key] ?? object(row.metrics)[key] ?? object(object(row.artifact).metrics)[key] ?? object(row.runtime)[key];
	let members: RecordValue[] = [];
	let architectures: Record<string, RecordValue> = {};
	let groups: RecordValue[] = [];
	let selectedGroup = '';
	let loading = true;
	let saving = false;
	let name = '';
	let note = '';
	let baseline = '';

	async function loadGroup(groupId: string) {
		if (!groupId) return;
		loading = true;
		try {
			const group = await modelCatalogClient.comparisonGroup(groupId);
			members = array(group.members).map((member) => ({ ...object(member.artifact), ...member }));
			baseline = String(group.baseline_artifact_id ?? members[0]?.artifact_id ?? '');
			await loadArchitecture();
		} catch (error) { onfailure(error instanceof Error ? error.message : 'Could not load comparison group.'); }
		finally { loading = false; }
	}
	async function loadDirect() {
		loading = true;
		try {
			members = await Promise.all(artifactIds.map(async (id) => {
				const detail = await modelCatalogClient.artifactDetail(id);
				return { ...object(detail.artifact), ...object(detail.facts), ...detail };
			}));
			baseline = artifactIds[0] ?? '';
			await loadArchitecture();
		} catch (error) { onfailure(error instanceof Error ? error.message : 'Could not load selected models.'); }
		finally { loading = false; }
	}
	async function loadArchitecture() {
		architectures = Object.fromEntries(await Promise.all(members.map(async (member) => {
			const id = String(member.artifact_id ?? member.id);
			try { return [id, await modelCatalogClient.artifactArchitectureView(id)]; } catch { return [id, {}]; }
		})));
	}
	function metrics() {
		const keys = members.flatMap((member) => Object.keys(object(member.metrics)).concat(Object.keys(object(object(member.artifact).metrics))));
		return [...new Set(keys)].slice(0, 14);
	}
	function sharedMetrics() { return metrics().filter((key) => members.every((member) => scalar(member, key) !== null && scalar(member, key) !== undefined && scalar(member, key) !== '')); }
	function datasetRelationship() {
		const datasets = new Set(members.map((member) => String(scalar(member, 'dataset_id') ?? '')).filter(Boolean));
		return datasets.size === 1 ? 'Same recorded dataset' : datasets.size > 1 ? 'Different recorded datasets' : 'Dataset not recorded';
	}
	function protocolRelationship() {
		const protocols = new Set(members.map((member) => String(scalar(member, 'evaluation_protocol') ?? scalar(member, 'evaluation_split') ?? '')).filter(Boolean));
		return protocols.size === 1 && protocols.size > 0 ? 'Shared recorded protocol' : protocols.size > 1 ? 'Different recorded protocols' : 'Protocol not recorded';
	}
	function baselineValue(key: string) { return scalar(members.find((member) => String(member.artifact_id ?? member.id) === baseline) ?? {}, key); }
	function delta(member: RecordValue, key: string) { const current = Number(scalar(member, key)); const base = Number(baselineValue(key)); return Number.isFinite(current) && Number.isFinite(base) ? current - base : null; }
	function modules(member: RecordValue) { const result = object(architectures[String(member.artifact_id ?? member.id)]); const graph = object(result.architecture ?? result); return Array.isArray(graph.modules) ? graph.modules : []; }
	async function save() {
		if (members.length < 2) return;
		saving = true;
		try {
			const ids = members.map((member) => String(member.artifact_id ?? member.id));
			await modelCatalogClient.createComparisonGroup({ name: name || 'Model comparison', description: note, baseline_artifact_id: baseline || ids[0], members: ids.map((artifact_id) => ({ artifact_id, relationship_label: artifact_id === baseline ? 'baseline' : 'comparison' })) });
			groups = (await modelCatalogClient.comparisonGroups()).comparison_groups;
		} catch (error) { onfailure(error instanceof Error ? error.message : 'Could not save comparison.'); }
		finally { saving = false; }
	}
	onMount(async () => {
		try { groups = (await modelCatalogClient.comparisonGroups()).comparison_groups; } catch { /* comparison creation still works without history */ }
		if (artifactIds.length) await loadDirect(); else if (groups.length) { selectedGroup = String(groups[0].comparison_group_id); await loadGroup(selectedGroup); } else loading = false;
	});
</script>

<section class="compare-hero"><button class="back" on:click={onback}>← Model runs</button><p class="eyebrow">MODEL COMPARISON</p><h2>Make the differences between runs legible.</h2><p>Metrics are shown next to their training data, runtime, and model modules so a score never loses its context.</p></section>

{#if groups.length}<section class="group-picker"><label>Saved comparison<select bind:value={selectedGroup} on:change={() => loadGroup(selectedGroup)}>{#each groups as group}<option value={String(group.comparison_group_id)}>{text(group.name)}</option>{/each}</select></label></section>{/if}

{#if loading}<section class="comparison-surface"><p class="empty">Loading comparison evidence…</p></section>{:else if members.length < 2}<section class="comparison-surface"><h2>Select two model runs to compare</h2><p>Return to Model Runs, select candidates, then choose Compare selected.</p><button on:click={onback}>Browse model runs</button></section>{:else}
	<section class="comparison-surface overview"><div><span>Models</span><strong>{members.length}</strong></div><div><span>Baseline</span><strong>{text(scalar(members.find((member) => String(member.artifact_id ?? member.id) === baseline) ?? {}, 'name'))}</strong></div><div><span>Shared metrics</span><strong>{sharedMetrics().length}</strong></div><div><span>Dataset relationship</span><strong>{datasetRelationship()}</strong></div></section>
	<section class="comparison-surface compatibility"><header><div><p class="eyebrow">COMPARISON COMPATIBILITY</p><h2>{sharedMetrics().length ? 'Numeric evidence is available' : 'Provenance-only comparison'}</h2><p>{sharedMetrics().length ? `Shared metrics: ${sharedMetrics().join(', ').replaceAll('_', ' ')}.` : 'These runs have no metrics recorded for every selected model, so a numeric ranking would be misleading.'}</p></div></header><div class="compatibility-items"><span class:ready={datasetRelationship().startsWith('Same')}>{datasetRelationship()}</span><span class:ready={protocolRelationship().startsWith('Shared')}>{protocolRelationship()}</span><span class:ready={sharedMetrics().length > 0}>{sharedMetrics().length ? `${sharedMetrics().length} shared metric${sharedMetrics().length === 1 ? '' : 's'}` : 'No shared metrics'}</span></div>{#if !sharedMetrics().length}<p class="compatibility-next"><strong>Next:</strong> use the architecture and protocol evidence below, choose runs evaluated with the same metric, or inspect each model’s missing evidence in Model Runs.</p>{/if}</section>
	{#if sharedMetrics().length}<section class="comparison-surface"><header><div><h2>Performance and runtime</h2><p>Delta is relative to the selected baseline. Only metrics recorded for every model are shown.</p></div><label>Baseline<select bind:value={baseline}>{#each members as member}<option value={String(member.artifact_id ?? member.id)}>{text(scalar(member, 'name') ?? member.artifact_id)}</option>{/each}</select></label></header><div class="scroll"><table><thead><tr><th>Model</th>{#each sharedMetrics() as metric}<th>{metric.replaceAll('_', ' ')}</th>{/each}<th>Runtime</th></tr></thead><tbody>{#each members as member}<tr class:baseline={String(member.artifact_id ?? member.id) === baseline}><td><strong>{text(scalar(member, 'name') ?? member.artifact_id)}</strong><small>{text(scalar(member, 'architecture'))} · {text(scalar(member, 'variant'))}</small></td>{#each sharedMetrics() as metric}<td><strong>{text(scalar(member, metric))}</strong>{#if delta(member, metric) !== null}<small class:positive={delta(member, metric)! > 0} class:negative={delta(member, metric)! < 0}>{delta(member, metric)! > 0 ? '+' : ''}{delta(member, metric)!.toFixed(4)}</small>{/if}</td>{/each}<td>{text(scalar(member, 'runtime_seconds') ?? scalar(member, 'training_seconds') ?? scalar(member, 'training_time_seconds'))}</td></tr>{/each}</tbody></table></div></section>{/if}
	<section class="comparison-surface"><header><div><h2>Architecture and protocol</h2><p>Modules are read from each sealed V2 architecture contract.</p></div></header><div class="model-columns">{#each members as member}<article><h3>{text(scalar(member, 'name') ?? member.artifact_id)}</h3><dl><div><dt>Architecture</dt><dd>{text(scalar(member, 'architecture'))}</dd></div><div><dt>Dataset</dt><dd>{text(scalar(member, 'dataset_id'))}</dd></div><div><dt>Classifier</dt><dd>{text(scalar(member, 'classifier_type'))}</dd></div><div><dt>Stem</dt><dd>{text(scalar(member, 'stem_size'))}</dd></div></dl><div class="module-chain">{#each modules(member) as module}<span>{text(object(module).label ?? object(module).name ?? object(module).type ?? module)}</span>{/each}{#if !modules(member).length}<span>Architecture details unavailable</span>{/if}</div></article>{/each}</div></section>
	<section class="comparison-surface save"><div><h2>Save this comparison</h2><p>Saved groups keep your analytical relationship without modifying any artifact.</p></div><label>Name<input bind:value={name} placeholder="e.g. EfficientNetV2 classifier study" /></label><label>Context<textarea rows="2" bind:value={note} placeholder="What decision does this compare?"></textarea></label><button disabled={saving} on:click={save}>{saving ? 'Saving…' : 'Save comparison group'}</button></section>
{/if}

<style>
	.compare-hero{padding:1.55rem 0 1.1rem}.compare-hero h2{margin:.1rem 0 .4rem;font-size:clamp(1.55rem,3.2vw,2.35rem)}.compare-hero p{max-width:700px}.back{margin-bottom:1rem;padding:0;color:#17675d;background:none}.back:hover{background:none;text-decoration:underline}.group-picker{display:flex;justify-content:flex-end;margin-bottom:.75rem}.group-picker label{width:min(360px,100%)}.comparison-surface{margin-bottom:1rem;padding:1.15rem;background:white;border:1px solid var(--line);border-radius:11px;box-shadow:var(--shadow)}.comparison-surface header{display:flex;justify-content:space-between;gap:1rem;margin-bottom:1rem}.comparison-surface header p{margin:0;font-size:.82rem}.comparison-surface header label{min-width:190px;font-size:.7rem}.comparison-surface header select{min-height:2.1rem}.overview{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;padding:1px;background:var(--line);overflow:hidden}.overview div{padding:1rem;background:#fbfcfc}.overview span{display:block;color:var(--muted);font-size:.68rem;text-transform:uppercase}.overview strong{display:block;margin-top:.35rem;color:#1d464c;font-size:1.05rem}.compatibility{border-color:#beded8;background:#fbfefd}.compatibility h2{margin:.15rem 0 .25rem}.compatibility-items{display:flex;flex-wrap:wrap;gap:.45rem}.compatibility-items span{padding:.38rem .55rem;border-radius:99px;background:#f8e9cd;color:#77561f;font-size:.72rem;font-weight:700}.compatibility-items span.ready{background:#e3f3ef;color:#17675d}.compatibility-next{margin:.75rem 0 0;font-size:.8rem}.scroll{overflow:auto}.scroll table{min-width:740px}.scroll tr.baseline{background:#eaf7f3}.scroll small{display:block;margin-top:.22rem;color:#778d91;font-size:.68rem}.scroll small.positive{color:#157164}.scroll small.negative{color:#a13d44}.model-columns{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:.8rem}.model-columns article{padding:1rem;background:#f8faf9;border:1px solid var(--line);border-radius:9px}.model-columns h3{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.model-columns dl{margin:.8rem 0}.model-columns dl div{display:flex;justify-content:space-between;gap:.8rem;padding:.35rem 0;border-bottom:1px solid #e1e9e6;font-size:.73rem}.model-columns dt{color:var(--muted)}.model-columns dd{margin:0;max-width:55%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:700}.module-chain{display:flex;flex-wrap:wrap;gap:.35rem}.module-chain span{padding:.32rem .5rem;color:#25665f;background:#e3f3ef;border-radius:99px;font-size:.66rem;font-weight:700}.save{display:grid;grid-template-columns:1.1fr 1fr 1fr auto;gap:.75rem;align-items:end}.save p{margin:0;font-size:.8rem}.save label{font-size:.7rem}.save input,.save textarea{min-height:2.25rem}@media(max-width:720px){.overview{grid-template-columns:repeat(2,1fr)}.save{grid-template-columns:1fr}.comparison-surface header{display:grid}.comparison-surface header label{min-width:0}}
</style>
