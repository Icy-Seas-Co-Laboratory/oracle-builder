<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import type { RecordValue } from '$lib/api';
	import ConfigField from '$lib/config/ConfigField.svelte';
	import { configWithValue, fieldApplies } from '$lib/config/paths';
	import type { ConfigFieldDefinition } from '$lib/config/types';

	export let config: RecordValue = {};
	export let sections: string[] = [];
	export let excludePaths: string[] = [];
	export let fieldDefinitions: RecordValue[] = [];
	export let description = 'Less frequently changed controls for this stage.';

	type Field = { path: string; value: unknown };
	const dispatch = createEventDispatcher<{ change: { config: RecordValue } }>();
	const label = (path: string) => path.split('.').map((part) => part.replaceAll('_', ' ')).join(' · ');
	function leaves(value: unknown, path: string): Field[] {
		if (Array.isArray(value) || value === null || typeof value !== 'object') return [{ path, value }];
		return Object.entries(value as RecordValue).flatMap(([key, item]) => leaves(item, `${path}.${key}`));
	}
	function update(path: string, nextValue: unknown) { dispatch('change', { config: configWithValue(config, path, nextValue) }); }
	let showExpert = false;
	$: definitions = new Map((fieldDefinitions as ConfigFieldDefinition[]).map((field) => [field.path, field]));
	$: fields = sections.flatMap((section) => leaves(config[section], section)).filter((item) => {
		const definition = definitions.get(item.path);
		const exposure = definition?.exposure ?? (fieldDefinitions.length ? 'expert' : 'advanced');
		return !excludePaths.includes(item.path) && exposure !== 'internal' && (exposure !== 'expert' || showExpert) && fieldApplies(definition ?? { path: item.path }, config);
	});
	$: hasExpert = sections.flatMap((section) => leaves(config[section], section)).some((item) => definitions.get(item.path)?.exposure === 'expert' && !excludePaths.includes(item.path));
</script>

{#if fields.length || hasExpert}
	<details class="stage-advanced">
		<summary><span>Advanced options</span><small>{description}</small></summary>
		<div class="stage-advanced-body">
			{#if hasExpert}<label class="expert-toggle"><input type="checkbox" bind:checked={showExpert} /> Show expert controls</label>{/if}
			{#if fields.length}<div class="advanced-fields">
				{#each fields as item}
					<ConfigField definition={definitions.get(item.path) ?? { path: item.path, label: label(item.path) }} value={item.value} {config} disabled={definitions.get(item.path)?.editable === false} on:change={(event) => update(item.path, event.detail.value)} />
				{/each}
			</div>{:else}<p class="empty">Enable expert controls to edit the remaining options for this stage.</p>{/if}
		</div>
	</details>
{/if}

<style>
	.stage-advanced { border-top:1px solid var(--line); margin-top:1rem; }
	.stage-advanced summary { align-items:center; cursor:pointer; display:flex; gap:.55rem; justify-content:space-between; padding:.75rem 0; }
	.stage-advanced summary span { color:#214d50; font-size:.8rem; font-weight:700; }
	.stage-advanced summary small { color:var(--muted); font-size:.7rem; text-align:right; }
	.stage-advanced-body { padding:0 0 .25rem; }
	.advanced-fields { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:.65rem; padding-bottom:.85rem; }
	.expert-toggle { align-items:center; color:var(--muted); display:flex; font-size:.72rem; gap:.35rem; margin:0 0 .7rem; }.expert-toggle input { min-height:auto; }
</style>
