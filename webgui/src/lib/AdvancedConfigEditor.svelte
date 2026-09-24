<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import type { RecordValue } from '$lib/api';

	export let config: RecordValue = {};
	export let title = 'Advanced configuration';
	export let description = 'These values map directly to the sealed configuration.';
	export let sections: string[] = [];
	export let readOnlyPaths: string[] = [];
	export let excludePaths: string[] = [];

	type Field = { path: string; value: unknown };
	const dispatch = createEventDispatcher<{ change: { config: RecordValue } }>();
	const isRecord = (value: unknown): value is RecordValue => Boolean(value) && typeof value === 'object' && !Array.isArray(value);
	const copy = <T,>(value: T): T => JSON.parse(JSON.stringify(value));
	const label = (path: string) => path.split('.').map((part) => part.replaceAll('_', ' ')).join(' · ');

	function leaves(value: unknown, path: string): Field[] {
		if (Array.isArray(value) || value === null || typeof value !== 'object') return [{ path, value }];
		return Object.entries(value as RecordValue).flatMap(([key, item]) => leaves(item, `${path}.${key}`));
	}
	function setPath(target: RecordValue, path: string, next: unknown) {
		const keys = path.split('.'); let current = target;
		for (const key of keys.slice(0, -1)) current = isRecord(current[key]) ? current[key] as RecordValue : (current[key] = {});
		current[keys.at(-1)!] = next;
	}
	function decode(raw: string, original: unknown): unknown {
		if (typeof original === 'boolean') return raw === 'true';
		if (typeof original === 'number') return Number(raw);
		if (Array.isArray(original) || isRecord(original) || original === null) return JSON.parse(raw);
		return raw;
	}
	function update(path: string, raw: string, original: unknown) {
		try {
			const next = copy(config); setPath(next, path, decode(raw, original)); dispatch('change', { config: next });
		} catch { /* Keep the last valid config while JSON is incomplete. */ }
	}
	function editable(path: string) { return !readOnlyPaths.includes(path); }
	$: groups = (sections.length ? sections : Object.keys(config))
		.map((section) => ({ section, fields: leaves(config[section], section).filter((field) => !excludePaths.includes(field.path)) }))
		.filter((group) => group.fields.length);
</script>

<section class="advanced-config panel">
	<header><div><p class="eyebrow">ADVANCED ONLY</p><h2>{title}</h2><p>{description}</p></div></header>
	{#each groups as group}
		<details>
			<summary>{group.section.replaceAll('_', ' ')}</summary>
			<div class="advanced-fields">
				{#each group.fields as field}
					<label class:readonly={!editable(field.path)}>
						<span>{label(field.path)}</span>
						{#if typeof field.value === 'boolean'}
							<select disabled={!editable(field.path)} value={String(field.value)} on:change={(event) => update(field.path, event.currentTarget.value, field.value)}><option value="true">true</option><option value="false">false</option></select>
						{:else if Array.isArray(field.value) || isRecord(field.value) || field.value === null}
							<textarea rows="2" disabled={!editable(field.path)} value={JSON.stringify(field.value)} on:change={(event) => update(field.path, event.currentTarget.value, field.value)}></textarea>
						{:else}
							<input disabled={!editable(field.path)} type={typeof field.value === 'number' ? 'number' : 'text'} step={typeof field.value === 'number' ? 'any' : undefined} value={String(field.value ?? '')} on:change={(event) => update(field.path, event.currentTarget.value, field.value)} />
						{/if}
					</label>
				{/each}
			</div>
		</details>
	{/each}
</section>

<style>
	.advanced-config header { margin-bottom:.8rem; }.advanced-config h2 { margin:.12rem 0; font-size:1.05rem; }.advanced-config header p:last-child { margin:0; color:var(--muted); font-size:.78rem; }
	.advanced-config details { border-top:1px solid var(--line); }.advanced-config summary { padding:.7rem 0; cursor:pointer; color:#214d50; font-size:.8rem; font-weight:700; text-transform:capitalize; }.advanced-fields { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:.65rem; padding:0 0 .85rem; }.advanced-fields label { gap:.25rem; min-width:0; }.advanced-fields label > span { overflow:hidden; color:var(--muted); font-size:.67rem; text-overflow:ellipsis; white-space:nowrap; }.advanced-fields input,.advanced-fields select,.advanced-fields textarea { min-height:2.1rem; padding:.42rem .5rem; font:inherit; }.advanced-fields textarea { resize:vertical; font-family:'SFMono-Regular',Consolas,monospace; font-size:.7rem; }.advanced-fields .readonly { opacity:.58; }
</style>
