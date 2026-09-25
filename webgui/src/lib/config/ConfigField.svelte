<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import type { ConfigFieldDefinition } from './types';
	import { getPath, humanize, isRecord } from './paths';

	export let definition: ConfigFieldDefinition;
	export let value: unknown;
	export let disabled = false;
	export let config: Record<string, unknown> = {};

	const dispatch = createEventDispatcher<{ change: { value: unknown } }>();
	$: choiceContext = definition.allowed_values_path ? getPath(config, definition.allowed_values_path) : undefined;
	$: choices = (definition.allowed_values_by && typeof choiceContext === 'string' ? definition.allowed_values_by[choiceContext] : undefined) ?? definition.choices ?? [];
	$: choiceOptions = definition.choice_options?.filter((option) => choices.some((choice) => choice === option.value)) ?? choices.map((choice) => ({ value: choice, label: humanize(choice) }));
	$: controlType = definition.type ?? (typeof value === 'boolean' ? 'boolean' : Array.isArray(value) ? 'list' : typeof value);
	$: isBoolean = controlType === 'boolean';
	$: isBooleanAuto = choices.length === 3 && choices.includes(true) && choices.includes(false) && choices.includes('auto');
	$: isAutoNumber = controlType === 'integer_or_auto';
	$: isChoice = choices.length > 0 && !isBooleanAuto;
	$: isMultiChoice = definition.control === 'multi_select';
	$: isSegmentedChoice = isChoice && definition.control === 'segmented';
	$: isNumeric = controlType === 'integer' || controlType === 'number' || typeof value === 'number';
	$: isShape = controlType === 'shape';
	$: isCollection = controlType === 'list' || controlType === 'shape' || Array.isArray(value);
	$: isObject = isRecord(value) && !isCollection;

	function emit(raw: string) {
		if (isBoolean) dispatch('change', { value: raw === 'true' });
		else if (isBooleanAuto) dispatch('change', { value: raw === 'auto' ? 'auto' : raw === 'true' });
		else if (isChoice) dispatch('change', { value: choices.find((choice) => String(choice) === raw) ?? raw });
		else if (isNumeric) dispatch('change', { value: raw === '' ? null : Number(raw) });
		else if (isCollection) {
			try {
				const parsed = JSON.parse(raw);
				if (Array.isArray(parsed)) dispatch('change', { value: parsed });
			} catch { /* Preserve the last valid collection while editing. */ }
		} else if (isObject) {
			try { dispatch('change', { value: JSON.parse(raw) }); } catch { /* idem */ }
		} else dispatch('change', { value: raw });
	}
	function updateShape(index: number, raw: string) {
		const current = Array.isArray(value) ? [...value] : [];
		current[index] = Number(raw);
		dispatch('change', { value: current });
	}
	function toggleMultiChoice(choice: unknown) {
		if (disabled) return;
		const current = Array.isArray(value) ? value : [];
		const required = definition.required_choices ?? [];
		if (current.includes(choice)) {
			if (required.includes(choice)) return;
			dispatch('change', { value: current.filter((item) => item !== choice) });
		} else dispatch('change', { value: [...current, choice] });
	}
</script>

<label class:readonly={disabled} class="config-field">
	<span class="field-label" title={definition.help}>{definition.label ?? humanize(definition.path.split('.').at(-1) ?? definition.path)}</span>
	{#if isBooleanAuto}
		<span class="segmented" role="group" aria-label={definition.label ?? definition.path}>
			{#each [true, false, 'auto'] as choice}
				<button type="button" class:active={value === choice} disabled={disabled} on:click={() => dispatch('change', { value: choice })}>{choice === true ? 'On' : choice === false ? 'Off' : 'Auto'}</button>
			{/each}
		</span>
	{:else if isBoolean}
		<span class="boolean-toggle"><input type="checkbox" checked={Boolean(value)} disabled={disabled} on:change={(event) => dispatch('change', { value: event.currentTarget.checked })} /> <span>{value ? 'On' : 'Off'}</span></span>
	{:else if isAutoNumber}
		<span class="auto-number"><button type="button" class:active={value === 'auto'} disabled={disabled} on:click={() => dispatch('change', { value: 'auto' })}>Auto</button><input type="number" min={definition.minimum} max={definition.maximum} step={definition.step ?? '1'} disabled={disabled || value === 'auto'} value={value === 'auto' ? '' : value ?? ''} placeholder="Manual" on:change={(event) => emit(event.currentTarget.value)} /></span>
	{:else if isShape}
		<span class="shape-input" aria-label={definition.label ?? definition.path}>{#each (Array.isArray(value) ? value : []) as dimension, index}<input type="number" min="1" step="1" disabled={disabled} value={dimension} aria-label={`${definition.label ?? definition.path} dimension ${index + 1}`} on:change={(event) => updateShape(index, event.currentTarget.value)} />{/each}</span>
	{:else if isMultiChoice}
		<span class="choice-chips" aria-label={definition.label ?? definition.path}>{#each choiceOptions as choice}<button type="button" class:selected={Array.isArray(value) && value.includes(choice.value)} class:required={definition.required_choices?.includes(choice.value)} disabled={disabled} aria-pressed={Array.isArray(value) && value.includes(choice.value)} on:click={() => toggleMultiChoice(choice.value)}>{choice.label}</button>{/each}</span>
	{:else if isSegmentedChoice}
		<span class="segmented" role="group" aria-label={definition.label ?? definition.path}>{#each choiceOptions as choice}<button type="button" class:active={value === choice.value} disabled={disabled} on:click={() => dispatch('change', { value: choice.value })}>{choice.label}</button>{/each}</span>
	{:else if isChoice}
		<select disabled={disabled} value={String(value ?? '')} on:change={(event) => emit(event.currentTarget.value)}>
			{#each choiceOptions as choice}<option value={String(choice.value)}>{choice.label}</option>{/each}
		</select>
	{:else if isNumeric}
		<input type="number" min={definition.minimum} max={definition.maximum} step={definition.step ?? (controlType === 'integer' ? '1' : 'any')} disabled={disabled} value={value ?? ''} on:change={(event) => emit(event.currentTarget.value)} />
	{:else if isCollection || isObject}
		<textarea rows="2" disabled={disabled} value={JSON.stringify(value ?? (isCollection ? [] : {}))} aria-label={definition.label ?? definition.path} on:change={(event) => emit(event.currentTarget.value)}></textarea>
	{:else}
		<input type="text" disabled={disabled} value={String(value ?? '')} on:change={(event) => emit(event.currentTarget.value)} />
	{/if}
	{#if definition.help}<small>{definition.help}</small>{/if}
</label>

<style>
	.config-field { display:grid; gap:.25rem; min-width:0; }
	.field-label { color:var(--muted); font-size:.67rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
	.config-field input,.config-field select,.config-field textarea { min-height:2.1rem; padding:.42rem .5rem; font:inherit; }
	.config-field textarea { resize:vertical; font-family:'SFMono-Regular',Consolas,monospace; font-size:.7rem; }
	.config-field small { color:var(--muted); font-size:.66rem; line-height:1.25; }
	.readonly { opacity:.58; }
	.boolean-toggle { align-items:center; display:flex; gap:.4rem; min-height:2.1rem; font-size:.76rem; }
	.boolean-toggle input { min-height:auto; }
	.segmented { display:flex; border:1px solid var(--line); border-radius:.38rem; overflow:hidden; min-height:2.1rem; }
	.segmented button { background:transparent; border:0; border-right:1px solid var(--line); border-radius:0; color:var(--muted); flex:1; font-size:.72rem; padding:.2rem .35rem; }
	.segmented button:last-child { border-right:0; }
	.segmented button.active { background:#214d50; color:white; }
	.auto-number { display:flex; gap:.35rem; }.auto-number button { min-width:3.6rem; }.auto-number button.active { background:#214d50; color:white; }.auto-number input { min-width:0; width:100%; }
	.shape-input { display:flex; gap:.35rem; }.shape-input input { min-width:0; width:100%; }
	.choice-chips { display:flex; flex-wrap:wrap; gap:.35rem; min-height:2.1rem; }.choice-chips button { background:#f8faf9; border:1px solid var(--line); color:var(--muted); font-size:.7rem; padding:.3rem .48rem; }.choice-chips button.selected { background:#e2f3ef; border-color:#70bdb1; color:#125f58; }.choice-chips button.required { cursor:default; }.choice-chips button:disabled { opacity:.58; }
</style>
