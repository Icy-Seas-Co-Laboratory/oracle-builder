<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import type { RecordValue } from '$lib/api';
	import ConfigField from '$lib/config/ConfigField.svelte';
	import { configWithValue, fieldApplies, getPath } from '$lib/config/paths';
	import type { ConfigFieldDefinition } from '$lib/config/types';
	import { trainingHelp } from '$lib/config/training-help';

	export let config: RecordValue = {};
	export let fieldDefinitions: RecordValue[] = [];

	type Item = { path: string; value: unknown; definition: ConfigFieldDefinition };
	type Group = { regular: Item[]; expert: Item[] };
	type Category = 'augmentation' | 'optimization' | 'execution' | 'checkpoints' | 'outputs' | 'protocol';
	const corePaths = ['training.optimizer', 'training.learning_rate', 'training.loss', 'training.metrics'];
	const clearerLabels: Record<string, string> = {
		'recovery.enabled': 'Recovery snapshots',
		'recovery.save_every_epochs': 'Recovery interval (epochs)',
		'augmentation.enabled': 'Image augmentation',
		'self_supervised.augmentation.enabled': 'Self-supervised augmentation',
		'evidence.enabled': 'Evidence artifacts',
		'evaluation.benchmark.enabled': 'Inference benchmark',
		'evaluation.uncertainty.enabled': 'Uncertainty estimates',
		'output.intermediate_artifacts.enabled': 'Intermediate artifacts'
	};
	const dispatch = createEventDispatcher<{ change: { config: RecordValue } }>();
	const humanize = (path: string) => path.split('.').at(-1)?.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase()) ?? path;
	function leaves(value: unknown, path: string): { path: string; value: unknown }[] {
		if (Array.isArray(value) || value === null || typeof value !== 'object') return [{ path, value }];
		return Object.entries(value as RecordValue).flatMap(([key, item]) => leaves(item, `${path}.${key}`));
	}
	function category(path: string): Category | null {
		if (path.startsWith('augmentation.') || path.startsWith('metadata.augmentation.') || path.startsWith('self_supervised.augmentation.')) return 'augmentation';
		if (path === 'output.save_checkpoints' || path === 'callbacks.checkpoint_monitor' || path.startsWith('recovery.')) return 'checkpoints';
		if (path.startsWith('distribution.') || ['training.display', 'training.step_cursor_policy', 'training.work_unit_steps'].includes(path)) return 'execution';
		if (path.startsWith('training.') || path.startsWith('callbacks.')) return 'optimization';
		if (path.startsWith('output.') || path.startsWith('evaluation.') || path.startsWith('evidence.') || path.startsWith('inference.')) return 'outputs';
		if (path.startsWith('run.')) return 'protocol';
		return null;
	}
	function inThisStage(path: string): boolean {
		if (path === 'run.task' || path === 'training.epochs' || path === 'training.segmentation_target' || path === 'training.class_weights.mode' || path === 'training.class_weights.alpha') return false;
		if (path.startsWith('self_supervised.') && !path.startsWith('self_supervised.augmentation.')) return false;
		return category(path) !== null;
	}
	function update(path: string, nextValue: unknown) { dispatch('change', { config: configWithValue(config, path, nextValue) }); }
	$: definitions = new Map((fieldDefinitions as ConfigFieldDefinition[]).map((field) => [field.path, { ...field, label: clearerLabels[field.path] ?? field.label, help: field.help || trainingHelp[field.path] }]));
	$: stageItems = ['run', 'training', 'callbacks', 'recovery', 'augmentation', 'distribution', 'output', 'evidence', 'inference', 'evaluation']
		.flatMap((section) => leaves(config[section], section))
		.concat(leaves((config.metadata as RecordValue | undefined)?.augmentation, 'metadata.augmentation'))
		.concat(leaves((config.self_supervised as RecordValue | undefined)?.augmentation, 'self_supervised.augmentation'))
		.filter((item) => item.value !== undefined && inThisStage(item.path))
		.map((item) => ({ ...item, definition: definitions.get(item.path) ?? { path: item.path, label: humanize(item.path), help: trainingHelp[item.path] } }))
		.filter((item) => {
			const exposure = item.definition.exposure ?? (fieldDefinitions.length ? 'expert' : 'advanced');
			if (item.path === 'training.class_weights.beta' && getPath(config, 'training.class_weights.mode') !== 'effective_number') return false;
			if (item.path === 'training.class_weights.values' && getPath(config, 'training.class_weights.mode') !== 'explicit') return false;
			return exposure !== 'internal' && fieldApplies(item.definition, config);
		});
	$: grouped = (() => {
		const result: Record<Category, Group> = {
			augmentation: { regular: [], expert: [] },
			optimization: { regular: [], expert: [] },
			execution: { regular: [], expert: [] },
			checkpoints: { regular: [], expert: [] },
			outputs: { regular: [], expert: [] },
			protocol: { regular: [], expert: [] }
		};
		for (const item of stageItems) {
			if (corePaths.includes(item.path)) continue;
			const name = category(item.path);
			if (name) result[name][item.definition.exposure === 'expert' ? 'expert' : 'regular'].push(item);
		}
		result.augmentation.regular.sort((a, b) => a.path === 'augmentation.enabled' ? -1 : b.path === 'augmentation.enabled' ? 1 : a.path.localeCompare(b.path));
		result.checkpoints.regular.sort((a, b) => {
			const order = ['output.save_checkpoints', 'callbacks.checkpoint_monitor', 'recovery.enabled', 'recovery.save_every_epochs'];
			return order.indexOf(a.path) - order.indexOf(b.path);
		});
		return result;
	})();
	$: task = String(getPath(config, 'run.task', 'classification'));
	$: augmentationEnabled = Boolean(getPath(config, 'augmentation.enabled', false));
	$: checkpointsEnabled = Boolean(getPath(config, 'output.save_checkpoints', false));
</script>

<div class="training-controls">
	<div class="training-core">
		<div class="subsection-heading"><h3>Core training</h3><p>Optimizer, learning rate, objective, and reported metrics.</p></div>
		<div class="field-grid">
			{#each corePaths as path}
				<ConfigField helpDisplay="tooltip" definition={definitions.get(path) ?? { path, label: humanize(path), help: trainingHelp[path] }} value={getPath(config, path)} {config} on:change={(event) => update(path, event.detail.value)} />
			{/each}
		</div>
	</div>

	<details class="training-menu">
		<summary><span><strong>Augmentation</strong><small>Image transforms, metadata noise, and self-supervised views</small></span><em>{augmentationEnabled ? 'Image on' : 'Image off'}</em></summary>
		<div class="menu-body">
			<div class="subsection-heading"><h3>Training images</h3><p>These transforms apply only when augmentation is enabled.</p></div>
			<div class="field-grid">{#each grouped.augmentation.regular.filter((item) => item.path.startsWith('augmentation.')) as item}<ConfigField helpDisplay="tooltip" definition={item.definition} value={item.value} {config} on:change={(event) => update(item.path, event.detail.value)} />{/each}</div>
			{#if grouped.augmentation.regular.some((item) => item.path.startsWith('metadata.augmentation.'))}<div class="subsection-heading nested"><h3>Metadata noise</h3><p>Applied to eligible scalar fields during classification training.</p></div><div class="field-grid">{#each grouped.augmentation.regular.filter((item) => item.path.startsWith('metadata.augmentation.')) as item}<ConfigField helpDisplay="tooltip" definition={item.definition} value={item.value} {config} on:change={(event) => update(item.path, event.detail.value)} />{/each}</div>{/if}
			{#if (task === 'embedding' || task === 'classification') && Boolean(getPath(config, 'self_supervised.enabled', false)) && grouped.augmentation.regular.some((item) => item.path.startsWith('self_supervised.augmentation.'))}<div class="subsection-heading nested"><h3>Self-supervised views</h3><p>Transforms used to create paired representation-learning views.</p></div><div class="field-grid">{#each grouped.augmentation.regular.filter((item) => item.path.startsWith('self_supervised.augmentation.')) as item}<ConfigField helpDisplay="tooltip" definition={item.definition} value={item.value} {config} on:change={(event) => update(item.path, event.detail.value)} />{/each}</div>{/if}
			{#if grouped.augmentation.expert.length}<details class="expert-menu"><summary>Expert augmentation settings</summary><div class="field-grid">{#each grouped.augmentation.expert as item}<ConfigField helpDisplay="tooltip" definition={item.definition} value={item.value} {config} on:change={(event) => update(item.path, event.detail.value)} />{/each}</div></details>{/if}
		</div>
	</details>

	{#if grouped.optimization.regular.length || grouped.optimization.expert.length}<details class="training-menu"><summary><span><strong>Optimization & scheduling</strong><small>Class weighting, weight decay, early stopping, and learning-rate changes</small></span></summary><div class="menu-body"><div class="field-grid">{#each grouped.optimization.regular as item}<ConfigField helpDisplay="tooltip" definition={item.definition} value={item.value} {config} on:change={(event) => update(item.path, event.detail.value)} />{/each}</div>{#if grouped.optimization.expert.length}<details class="expert-menu"><summary>Expert optimization settings</summary><div class="field-grid">{#each grouped.optimization.expert as item}<ConfigField helpDisplay="tooltip" definition={item.definition} value={item.value} {config} on:change={(event) => update(item.path, event.detail.value)} />{/each}</div></details>{/if}</div></details>{/if}

	{#if grouped.execution.regular.length || grouped.execution.expert.length}<details class="training-menu"><summary><span><strong>Compute & execution</strong><small>Device strategy, worker behavior, and execution boundaries</small></span></summary><div class="menu-body"><div class="field-grid">{#each grouped.execution.regular as item}<ConfigField helpDisplay="tooltip" definition={item.definition} value={item.value} {config} on:change={(event) => update(item.path, event.detail.value)} />{/each}</div>{#if grouped.execution.expert.length}<details class="expert-menu"><summary>Expert compute settings</summary><div class="field-grid">{#each grouped.execution.expert as item}<ConfigField helpDisplay="tooltip" definition={item.definition} value={item.value} {config} on:change={(event) => update(item.path, event.detail.value)} />{/each}</div></details>{/if}</div></details>{/if}

	<details class="training-menu advanced-menu"><summary><span><strong>Advanced</strong><small>Checkpoint saving, recovery, outputs, and evaluation</small></span></summary><div class="menu-body">
		<details class="nested-menu"><summary><span><strong>Checkpoints & recovery</strong><small>Saved model checkpoints and resumable training snapshots are separate controls</small></span><em>{checkpointsEnabled ? 'Saving' : 'Not saving'}</em></summary><div class="menu-body"><div class="field-grid">{#each grouped.checkpoints.regular as item}<ConfigField helpDisplay="tooltip" definition={item.definition} value={item.value} {config} on:change={(event) => update(item.path, event.detail.value)} />{/each}</div>{#if grouped.checkpoints.expert.length}<details class="expert-menu"><summary>Expert checkpoint settings</summary><div class="field-grid">{#each grouped.checkpoints.expert as item}<ConfigField helpDisplay="tooltip" definition={item.definition} value={item.value} {config} on:change={(event) => update(item.path, event.detail.value)} />{/each}</div></details>{/if}</div></details>
		{#if grouped.outputs.regular.length || grouped.outputs.expert.length}<details class="nested-menu"><summary><span><strong>Artifacts & evaluation</strong><small>Prediction outputs, evidence, inference, and benchmarks</small></span></summary><div class="menu-body"><div class="field-grid">{#each grouped.outputs.regular as item}<ConfigField helpDisplay="tooltip" definition={item.definition} value={item.value} {config} on:change={(event) => update(item.path, event.detail.value)} />{/each}</div>{#if grouped.outputs.expert.length}<details class="expert-menu"><summary>Expert evaluation & output settings</summary><div class="field-grid">{#each grouped.outputs.expert as item}<ConfigField helpDisplay="tooltip" definition={item.definition} value={item.value} {config} on:change={(event) => update(item.path, event.detail.value)} />{/each}</div></details>{/if}</div></details>{/if}
		{#if grouped.protocol.regular.length || grouped.protocol.expert.length}<details class="nested-menu"><summary><span><strong>Run protocol</strong><small>Seed and other reusable run metadata</small></span></summary><div class="menu-body"><div class="field-grid">{#each grouped.protocol.regular as item}<ConfigField helpDisplay="tooltip" definition={item.definition} value={item.value} {config} on:change={(event) => update(item.path, event.detail.value)} />{/each}</div>{#if grouped.protocol.expert.length}<details class="expert-menu"><summary>Expert protocol settings</summary><div class="field-grid">{#each grouped.protocol.expert as item}<ConfigField helpDisplay="tooltip" definition={item.definition} value={item.value} {config} on:change={(event) => update(item.path, event.detail.value)} />{/each}</div></details>{/if}</div></details>{/if}
	</div></details>
</div>

<style>
	.training-controls { margin-top:.75rem; }
	.training-core { padding:.25rem 0 .85rem; }
	.subsection-heading { margin:.15rem 0 .6rem; }
	.subsection-heading h3 { margin:0; font-size:.83rem; color:var(--ink); }
	.subsection-heading p { margin:.15rem 0 0; color:var(--muted); font-size:.7rem; }
	.subsection-heading.nested { border-top:1px solid var(--line); margin-top:1rem; padding-top:.8rem; }
	.field-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:.7rem; }
	.training-menu { border-top:1px solid var(--line); }
	.training-menu > summary,.nested-menu > summary { display:flex; align-items:center; justify-content:space-between; gap:.8rem; padding:.8rem .1rem; cursor:pointer; list-style:none; }
	.training-menu > summary::-webkit-details-marker,.nested-menu > summary::-webkit-details-marker { display:none; }
	.training-menu > summary::before,.nested-menu > summary::before { content:'▸'; color:var(--teal); font-size:.9rem; }
	.training-menu[open] > summary::before,.nested-menu[open] > summary::before { content:'▾'; }
	.training-menu > summary span,.nested-menu > summary span { flex:1; display:grid; gap:.12rem; }
	.training-menu > summary strong,.nested-menu > summary strong { color:#214d50; font-size:.8rem; }
	.training-menu > summary small,.nested-menu > summary small { color:var(--muted); font-size:.68rem; }
	.training-menu > summary em,.nested-menu > summary em { color:var(--muted); font-size:.66rem; font-style:normal; white-space:nowrap; }
	.menu-body { padding:.1rem .1rem 1rem 1.2rem; }
	.nested-menu { border-top:1px solid var(--line); }
	.nested-menu:first-child { border-top:0; }
	.expert-menu { margin-top:.9rem; border-top:1px dashed var(--line); }
	.expert-menu > summary { cursor:pointer; padding:.65rem 0; color:#31575c; font-size:.73rem; font-weight:700; }
	.expert-menu > .field-grid { padding-bottom:.3rem; }
	@media (max-width:650px) { .menu-body { padding-left:.2rem; } .training-menu > summary small,.nested-menu > summary small { line-height:1.25; } }
</style>
