<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import type { RecordValue } from '$lib/api';
	import ConfigField from '$lib/config/ConfigField.svelte';
	import { configWithValue, fieldApplies, isRecord } from '$lib/config/paths';
	import type { ConfigFieldDefinition } from '$lib/config/types';

	export let config: RecordValue = {};
	export let title = 'Advanced configuration';
	export let description = 'These values map directly to the sealed configuration.';
	export let sections: string[] = [];
	export let readOnlyPaths: string[] = [];
	export let excludePaths: string[] = [];
	export let fieldDefinitions: RecordValue[] = [];

	type Field = { path: string; value: unknown };
	type Definition = ConfigFieldDefinition;
	const dispatch = createEventDispatcher<{ change: { config: RecordValue } }>();
	const label = (path: string) => path.split('.').map((part) => part.replaceAll('_', ' ')).join(' · ');

	function leaves(value: unknown, path: string): Field[] {
		if (Array.isArray(value) || value === null || typeof value !== 'object') return [{ path, value }];
		return Object.entries(value as RecordValue).flatMap(([key, item]) => leaves(item, `${path}.${key}`));
	}
	function update(path: string, nextValue: unknown) { dispatch('change', { config: configWithValue(config, path, nextValue) }); }
	let showExpert = false;
	$: definitions = new Map((fieldDefinitions as Definition[]).map((field) => [field.path, field]));
	function definition(path: string) { return definitions.get(path); }
	function exposure(path: string) {
		// Preserve a usable editor during an API outage; once the catalog loads,
		// unknown imported keys are deliberately treated as expert-only.
		return definition(path)?.exposure ?? (fieldDefinitions.length ? 'expert' : 'advanced');
	}
	function editable(path: string) { return !readOnlyPaths.includes(path) && definition(path)?.editable !== false; }
	function visible(path: string) {
		const tier = exposure(path);
		const field = definition(path);
		return !excludePaths.includes(path) && tier !== 'internal' && fieldApplies(field ?? { path }, config) && (tier === 'standard' || tier === 'advanced' || (tier === 'expert' && showExpert));
	}
	$: groups = (sections.length ? sections : Object.keys(config))
		.map((section) => ({ section, fields: leaves(config[section], section).filter((field) => visible(field.path)) }))
		.filter((group) => group.fields.length);
	$: hasExpert = (sections.length ? sections : Object.keys(config)).some((section) => leaves(config[section], section).some((field) => exposure(field.path) === 'expert' && !excludePaths.includes(field.path)));
</script>

{#if groups.length || hasExpert}
	<section class="advanced-config panel">
		<header><div><p class="eyebrow">ADVANCED OPTIONS</p><h2>{title}</h2><p>{description}</p></div>{#if hasExpert}<label class="expert-toggle"><input type="checkbox" bind:checked={showExpert} /> Show expert controls</label>{/if}</header>
		{#each groups as group}
			<details>
				<summary>{group.section.replaceAll('_', ' ')}</summary>
				<div class="advanced-fields">
					{#each group.fields as field}
						<ConfigField definition={definition(field.path) ?? { path: field.path, label: label(field.path) }} value={field.value} {config} disabled={!editable(field.path)} on:change={(event) => update(field.path, event.detail.value)} />
					{/each}
				</div>
			</details>
		{/each}
	</section>
{/if}

<style>
	.advanced-config header { align-items:start; display:flex; gap:.8rem; justify-content:space-between; margin-bottom:.8rem; }.advanced-config h2 { margin:.12rem 0; font-size:1.05rem; }.advanced-config header p:last-child { margin:0; color:var(--muted); font-size:.78rem; }.expert-toggle { align-items:center; color:var(--muted); display:flex; flex:none; font-size:.72rem; gap:.35rem; white-space:nowrap; }.expert-toggle input { min-height:auto; }
	.advanced-config details { border-top:1px solid var(--line); }.advanced-config summary { padding:.7rem 0; cursor:pointer; color:#214d50; font-size:.8rem; font-weight:700; text-transform:capitalize; }.advanced-fields { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:.65rem; padding:0 0 .85rem; }
</style>
