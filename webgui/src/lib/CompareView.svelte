<script lang="ts">
	import { onMount } from 'svelte';
	import { api, type RecordValue } from '$lib/api';
	import EmptyState from '$lib/EmptyState.svelte';

	export let onfailure: (message: string) => void;
	const asRecord = (value: unknown) => value && typeof value === 'object' ? value as RecordValue : null;
	const records = (value: unknown) => Array.isArray(value) ? value as RecordValue[] : [];
	const text = (value: unknown) => value == null || value === '' ? '—' : String(value);
	const numeric = (value: unknown) => Number.isFinite(Number(value)) ? Number(value) : null;
	let artifacts: RecordValue[] = [];
	let groups: RecordValue[] = [];
	let selected = new Set<string>();
	let groupName = '';
	let relationshipTag = 'related';
	let note = '';
	let baselineId = '';
	let activeGroup: RecordValue | null = null;
	let loading = true;
	let saving = false;

	async function load() {
		loading = true;
		try {
			const [catalog, saved] = await Promise.all([api.artifactCatalog(), api.comparisonGroups()]);
			artifacts = catalog.artifacts; groups = saved.comparison_groups;
		} catch (error) {
			try { artifacts = (await api.artifacts()).artifacts; groups = (await api.comparisons()).comparisons; }
			catch (fallback) { onfailure(fallback instanceof Error ? fallback.message : 'Could not load comparisons.'); }
		} finally { loading = false; }
	}
	function toggle(id: string) { selected = new Set(selected); selected.has(id) ? selected.delete(id) : selected.add(id); if (!baselineId) baselineId = id; }
	function metric(row: RecordValue, key: string) { return numeric(asRecord(row.metrics)?.[key]); }
	function keys(rows: RecordValue[]) { return [...new Set(rows.flatMap((row) => Object.keys(asRecord(row.metrics) ?? {})))].filter((key) => rows.some((row) => numeric(asRecord(row.metrics)?.[key]) != null)).slice(0, 8); }
	async function create() {
		if (selected.size < 2) return;
		saving = true;
		try {
			const artifactIds = [...selected];
			const created = await api.createComparisonGroup({ name: groupName || 'Related runs', description: note, relationship_label: relationshipTag, baseline_artifact_id: baselineId || artifactIds[0], members: artifactIds.map((artifact_id) => ({ artifact_id, relationship_label: artifact_id === baselineId ? 'baseline' : relationshipTag })) });
			activeGroup = created; groupName = ''; note = ''; selected = new Set(); baselineId = '';
			groups = (await api.comparisonGroups()).comparison_groups;
		} catch (error) { onfailure(error instanceof Error ? error.message : 'Could not save this comparison group.'); }
		finally { saving = false; }
	}
	async function openGroup(id: string) { try { activeGroup = await api.comparisonGroup(id); } catch (error) { onfailure(error instanceof Error ? error.message : 'Could not open comparison.'); } }
	const rows = () => records(activeGroup?.members ?? asRecord(activeGroup?.selection)?.artifacts).map((member) => asRecord(member.artifact) ? { ...asRecord(member.artifact)!, ...member, runtime_seconds: asRecord(member.timing)?.runtime_seconds ?? member.runtime_seconds } : member);
	onMount(load);
</script>

<section class="panel results-selector"><div><p class="eyebrow">COMPARE</p><h2>Group related runs, then let the evidence speak.</h2><p>Relationships are user-defined. Compatibility is explained, never used to hide useful context.</p></div></section>

{#if loading}<section class="panel"><p class="empty">Loading model runs…</p></section>{:else}<section class="panel"><div class="panel-head"><div><h2>Create a related-run group</h2><p>Select any two or more artifacts. Pick a baseline to calculate deltas.</p></div><span class="count">{selected.size}</span></div>
	{#if artifacts.length}<div class="table-scroll"><table><thead><tr><th></th><th>Run</th><th>Task</th><th>Primary metrics</th><th>Baseline</th></tr></thead><tbody>{#each artifacts as artifact}<tr><td><input class="selection-box" type="checkbox" checked={selected.has(String(artifact.artifact_id))} aria-label={`Select ${text(artifact.name)}`} on:change={() => toggle(String(artifact.artifact_id))} /></td><td><strong>{text(artifact.name)}</strong><small>{text(artifact.architecture)} · {text(artifact.dataset_id)}</small></td><td>{text(artifact.task)}</td><td>{#each Object.entries(asRecord(artifact.metrics) ?? {}).filter(([, value]) => numeric(value) != null).slice(0, 2) as item}<span class="metric-inline">{item[0].replaceAll('_', ' ')} <strong>{Number(item[1]).toFixed(4)}</strong></span>{/each}</td><td>{#if selected.has(String(artifact.artifact_id))}<input type="radio" name="baseline" checked={baselineId === artifact.artifact_id} aria-label={`Use ${text(artifact.name)} as baseline`} on:change={() => baselineId = String(artifact.artifact_id)} />{/if}</td></tr>{/each}</tbody></table></div>
		<form class="comparison-form" on:submit|preventDefault={create}><label>Group name<input bind:value={groupName} placeholder="Architecture trial" /></label><label>Relationship<select bind:value={relationshipTag}><option value="related">Related runs</option><option value="baseline">Baseline comparison</option><option value="ablation">Ablation</option><option value="dataset-revision">Dataset revision</option></select></label><label>Note<textarea rows="2" bind:value={note} placeholder="What changed, and why are these related?"></textarea></label><button disabled={selected.size < 2 || saving}>{saving ? 'Saving…' : 'Save comparison group'}</button></form>
	{:else}<EmptyState title="No artifacts to compare" text="Index or complete a model run first." />}{/if}</section>

	<section class="panel"><div class="panel-head"><div><h2>Saved comparison groups</h2><p>Manual relationships remain separate from immutable artifacts.</p></div><span class="count">{groups.length}</span></div>{#if groups.length}<div class="cards">{#each groups as group}<article class="comparison-card"><div><strong>{text(group.name)}</strong><p>{text(group.description)}</p><small>{text(group.relationship_label)} · {text(group.member_count)} runs</small></div><button class="secondary small" on:click={() => openGroup(String(group.comparison_group_id))}>Open</button></article>{/each}</div>{:else}<p class="empty">No groups yet. Start by tagging two related runs.</p>{/if}</section>
{/if}

{#if activeGroup}<section class="panel"><div class="panel-head"><div><p class="eyebrow">COMPARISON</p><h2>{text(activeGroup.name)}</h2><p>{text(activeGroup.description)}</p></div><button class="secondary small" on:click={() => activeGroup = null}>Close</button></div>{#if records(asRecord(activeGroup.compatibility)?.reasons).length}<div class="validation-message invalid"><strong>Protocol differences to keep in mind</strong><ul>{#each records(asRecord(activeGroup.compatibility)?.reasons) as reason}<li>{text(reason)}</li>{/each}</ul></div>{:else}<div class="validation-message valid"><strong>Comparable evidence</strong><span>Selected runs share the recorded evaluation protocol.</span></div>{/if}
	{#if rows().length}<div class="table-scroll"><table><thead><tr><th>Run</th>{#each keys(rows()) as key}<th>{key.replaceAll('_', ' ')}</th>{/each}<th>Runtime</th></tr></thead><tbody>{#each rows() as row}<tr><td><strong>{text(row.name)}</strong><small>{text(row.architecture)} · {text(row.task)}</small></td>{#each keys(rows()) as key}<td>{#if metric(row, key) != null}{metric(row, key)?.toFixed(4)}{:else}—{/if}</td>{/each}<td>{row.runtime_seconds == null ? '—' : `${text(row.runtime_seconds)} s`}</td></tr>{/each}</tbody></table></div>{/if}</section>{/if}
