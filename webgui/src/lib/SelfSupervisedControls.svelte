<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import ConfigField from '$lib/config/ConfigField.svelte';
	import { configWithValue, fieldApplies, getPath, setPath } from '$lib/config/paths';
	import type { RecordValue } from '$lib/api';
	import type { ConfigFieldDefinition } from '$lib/config/types';
	export let config: RecordValue = {};
	export let fieldDefinitions: RecordValue[] = [];
	const dispatch = createEventDispatcher<{ change: { config: RecordValue } }>();
	$: definitions = fieldDefinitions as ConfigFieldDefinition[];
	$: task = String(getPath(config, 'run.task', 'classification'));
	$: enabled = getPath(config, 'self_supervised.enabled', false) === true;
	const definition = (path: string): ConfigFieldDefinition => definitions.find((field) => field.path === path) ?? { path };
	const value = (path: string) => getPath(config, path, definition(path).default);
	const applicable = (path: string) => fieldApplies(definition(path), config);
	function update(path: string, nextValue: unknown) {
		const next = configWithValue(config, path, nextValue);
		if (path === 'self_supervised.enabled' && nextValue === true) {
			const method = String(getPath(next, 'self_supervised.method', 'byol'));
			if (task === 'segmentation') setPath(next, 'self_supervised.method', 'grayscale_reconstruction');
			else if (method === 'grayscale_reconstruction') setPath(next, 'self_supervised.method', 'byol');
		}
		dispatch('change', { config: next });
	}
	const objective = ['self_supervised.method', 'self_supervised.learning_rate', 'self_supervised.ssl_optimizer', 'self_supervised.teacher_momentum', 'self_supervised.temperature', 'self_supervised.projection_dim', 'self_supervised.projection_hidden_dim', 'self_supervised.minimum_global_batch_size', 'self_supervised.reconstruction_foreground_weight'];
	const optimizer = ['self_supervised.weight_decay', 'self_supervised.beta_1', 'self_supervised.beta_2', 'self_supervised.optimizer_epsilon'];
	$: regularization = definitions.filter((field) => /^self_supervised\.(vicreg_|encoder_|collapse_)/.test(field.path) && fieldApplies(field, config));
	$: augmentation = definitions.filter((field) => field.path.startsWith('self_supervised.augmentation.') && fieldApplies(field, config));
	$: health = definitions.filter((field) => field.path.startsWith('self_supervised.embedding_health.') && fieldApplies(field, config));
</script>

<section class="panel ssl-controls">
	<header><p class="eyebrow">SELF-SUPERVISED PRETRAINING</p><h2>Learn image representations</h2><p>{task === 'embedding' ? 'This workflow trains an embedding with SSL. It has no supervised classifier fitting phase.' : 'Optional SSL pretraining runs before supervised fitting, with its own optimizer and epoch budget.'} Set independent phase budgets in Queue.</p></header>
	<ConfigField definition={definition('self_supervised.enabled')} value={enabled} {config} on:change={(event) => update('self_supervised.enabled', event.detail.value)} />
	{#if enabled}
		<div class="ssl-grid">{#each objective.filter(applicable) as path}<ConfigField definition={definition(path)} value={value(path)} {config} on:change={(event) => update(path, event.detail.value)} />{/each}</div>
		<details><summary>Optimizer details</summary><div class="ssl-grid">{#each optimizer.filter(applicable) as path}<ConfigField definition={definition(path)} value={value(path)} {config} on:change={(event) => update(path, event.detail.value)} />{/each}</div></details>
		{#if regularization.length}<details><summary>Representation regularization and collapse thresholds</summary><div class="ssl-grid">{#each regularization as field}<ConfigField definition={field} value={value(field.path)} {config} on:change={(event) => update(field.path, event.detail.value)} />{/each}</div></details>{/if}
		{#if augmentation.length}<details><summary>SSL views and augmentation</summary><p>These controls create the views used by SSL. Supervised image augmentation is configured separately.</p><div class="ssl-grid">{#each augmentation as field}<ConfigField definition={field} value={value(field.path)} {config} on:change={(event) => update(field.path, event.detail.value)} />{/each}</div></details>{/if}
		{#if task === 'embedding' && health.length}<details><summary>Embedding health checks</summary><p>Inspect spread and similarity in the exported representation after embedding training.</p><div class="ssl-grid">{#each health as field}<ConfigField definition={field} value={value(field.path)} {config} on:change={(event) => update(field.path, event.detail.value)} />{/each}</div></details>{/if}
	{/if}
</section>

<style>.ssl-controls header{margin-bottom:1rem}.ssl-controls header p,.ssl-controls details p{color:var(--muted);font-size:.8rem}.ssl-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:.75rem;margin:.75rem 0}.ssl-controls details{border-top:1px solid var(--line);margin-top:.75rem}.ssl-controls summary{cursor:pointer;padding:.75rem 0;font-size:.82rem;font-weight:700}</style>
