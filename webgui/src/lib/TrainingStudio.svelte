<script lang="ts">
	import { onMount } from 'svelte';
	import { api, type RecordValue } from '$lib/api';
	import AdvancedConfigEditor from '$lib/AdvancedConfigEditor.svelte';

	export let datasets: RecordValue[] = [];
	export let recipes: RecordValue[] = [];
	export let artifacts: RecordValue[] = [];
	export let initialDraftId = '';
	export let onnotice: (message: string) => void;
	export let onfailure: (message: string) => void;
	export let onplanned: () => void;

	const records = (value: unknown): RecordValue[] => Array.isArray(value) ? value as RecordValue[] : [];
	const record = (value: unknown): RecordValue => value && typeof value === 'object' && !Array.isArray(value) ? value as RecordValue : {};
	const copy = <T,>(value: T): T => JSON.parse(JSON.stringify(value));
	const merge = (base: RecordValue, override: RecordValue): RecordValue => {
		const next = copy(base);
		for (const [key, value] of Object.entries(override)) next[key] = record(value) && record(next[key]) ? merge(record(next[key]), record(value)) : value;
		return next;
	};
	const runSections = ['data', 'preprocessing', 'training', 'callbacks', 'recovery', 'augmentation', 'distribution', 'self_supervised', 'evidence', 'inference', 'output', 'evaluation', 'tiling', 'metadata'];
	const selectSections = (source: RecordValue): RecordValue => Object.fromEntries(runSections.filter((key) => key in source).map((key) => [key, copy(source[key])])) as RecordValue;
	const display = (value: unknown) => value == null || value === '' ? '—' : String(value);
	let drafts: RecordValue[] = [];
	let draftId = '';
	let recipeId = '';
	let datasetId = '';
	let runName = '';
	let description = '';
	let initialization = 'scratch';
	let sourceArtifactId = '';
	let epochs = 30;
	let batchSize = 16;
	let learningRate = 0.001;
	let polarity = 'auto';
	let rotation = 0;
	let metadataVariance = 0;
	let gpuCount = 0;
	let creating = false;
	let advancedOverrides: RecordValue = {};
	let schemaDefaults: RecordValue = {};
	let synchronizedDraftId = '';

	$: if (!datasetId && datasets.length) datasetId = String(datasets[0].dataset_id);
	$: if (!recipeId && recipes.length) recipeId = String(recipes[0].recipe_id);
	$: if (initialDraftId) draftId = initialDraftId;
	$: chosenDraft = drafts.find((item) => String(item.draft_id) === draftId);
	$: chosenRecipe = recipes.find((item) => String(item.recipe_id) === recipeId);
	$: selectedDataset = datasets.find((item) => String(item.dataset_id) === datasetId);
	$: if (draftId && chosenDraft && synchronizedDraftId !== draftId) { advancedOverrides = selectSections(record(chosenDraft.config)); synchronizedDraftId = draftId; }
	$: if (!draftId && Object.keys(schemaDefaults).length && synchronizedDraftId !== 'recipe') { advancedOverrides = selectSections(schemaDefaults); synchronizedDraftId = 'recipe'; }

	onMount(async () => {
		try {
			const [draftResult, schema] = await Promise.all([api.modelDrafts(), api.configurationSchema()]);
			drafts = records(draftResult.drafts); schemaDefaults = record(schema.defaults);
		} catch { drafts = []; }
	});
	function normalOverrides(): RecordValue {
		return {
			training: { epochs, batch_size: batchSize, learning_rate: learningRate },
			preprocessing: polarity === 'auto' ? {} : { invert: polarity === 'true' },
			augmentation: { rotation },
			metadata: { augmentation: { gaussian_variance: metadataVariance } }
		};
	}
	function overrides(): RecordValue {
		return merge(advancedOverrides, normalOverrides());
	}
	function syncAdvanced(next: RecordValue) {
		advancedOverrides = next;
		const training = record(next.training); const preprocessing = record(next.preprocessing); const augmentation = record(next.augmentation); const metadata = record(record(next.metadata).augmentation);
		epochs = Number(training.epochs ?? epochs); batchSize = Number(training.batch_size ?? batchSize); learningRate = Number(training.learning_rate ?? learningRate);
		polarity = typeof preprocessing.invert === 'boolean' ? String(preprocessing.invert) : 'auto'; rotation = Number(augmentation.rotation ?? rotation); metadataVariance = Number(metadata.gaussian_variance ?? metadataVariance);
	}
	function persistNormal() { advancedOverrides = merge(advancedOverrides, normalOverrides()); }
	async function plan() {
		if (!datasetId || (!draftId && !recipeId) || !runName.trim()) return;
		creating = true;
		const initializationPayload = initialization === 'scratch' ? { mode: 'scratch' } : { mode: initialization, source_artifact_id: sourceArtifactId };
		try {
			if (draftId) {
				await api.planTrainingFromDraft(draftId, { name: runName.trim(), dataset_id: datasetId, description, resources: { gpu_count: gpuCount }, training_overrides: overrides(), initialization: initializationPayload });
			} else {
				await api.createTrainingExperiment({ name: runName.trim(), dataset_id: datasetId, recipe_ids: [recipeId], seeds: [123], description, resources: { gpu_count: gpuCount }, config_overrides: overrides() });
			}
			onnotice('Created a sealed training plan. Review it in the run queue before dispatch.'); onplanned(); runName = ''; description = '';
		} catch (error) { onfailure(error instanceof Error ? error.message : 'Could not create training plan.'); }
		finally { creating = false; }
	}
</script>

<section class="studio-intro"><div><p class="eyebrow">TRAINING</p><h1>Plan a reproducible training session.</h1><p>Choose data and a model, set training-only behavior, then seal the configuration for preflight and dispatch.</p></div></section>

<div class="training-layout">
	<section class="panel training-form">
		<div class="panel-head"><div><h2>1. Choose the model definition</h2><p>Drafts are preferred for V2 composition. Recipes remain supported for established workflows.</p></div></div>
		<div class="choice-tabs"><button class:active={Boolean(draftId)} class="secondary" on:click={() => draftId = drafts[0] ? String(drafts[0].draft_id) : ''}>Model draft</button><button class:active={!draftId} class="secondary" on:click={() => draftId = ''}>Recipe</button></div>
		{#if draftId}<label>Model draft<select bind:value={draftId}>{#each drafts as draft}<option value={String(draft.draft_id)}>{display(draft.name)} · revision {display(draft.revision)}</option>{/each}</select></label>{:else}<label>Training recipe<select bind:value={recipeId}>{#each recipes as recipe}<option value={String(recipe.recipe_id)}>{display(recipe.name)} · {display(recipe.model)}</option>{/each}</select></label>{/if}
		<div class="training-summary"><span>Selected</span><strong>{draftId ? display(chosenDraft?.name) : display(chosenRecipe?.name)}</strong><small>{draftId ? 'Editable V2 draft; a configuration revision is sealed now.' : `${display(chosenRecipe?.model)} recipe; run-only overrides are sealed now.`}</small></div>

		<div class="panel-head section-break"><div><h2>2. Choose a frozen training set</h2><p>Training always uses a registered frozen revision.</p></div></div>
		<label>Dataset<select bind:value={datasetId}>{#each datasets as dataset}<option value={String(dataset.dataset_id)}>{display(dataset.name)} · {display(dataset.lifecycle)}</option>{/each}</select></label>
		<div class="training-summary"><span>Dataset</span><strong>{display(selectedDataset?.name)}</strong><small>{display(selectedDataset?.dataset_type)} · fingerprint {display(selectedDataset?.fingerprint_sha256).slice(0, 12)}</small></div>

		<div class="panel-head section-break"><div><h2>3. Initialization</h2><p>Starting weights are explicit and checked for compatibility before dispatch.</p></div></div>
		<div class="editor-fields"><label>Mode<select bind:value={initialization}><option value="scratch">Train from scratch</option><option value="fine_tune">Fine-tune a sealed model</option><option value="transfer_encoder">Transfer encoder only</option><option value="resume">Resume compatible checkpoint</option></select></label>{#if initialization !== 'scratch'}<label>Source model<select bind:value={sourceArtifactId}><option value="">Choose a sealed model…</option>{#each artifacts as artifact}<option value={String(artifact.artifact_id)}>{display(artifact.name)} · {display(artifact.architecture)}</option>{/each}</select></label>{/if}</div>
	</section>

	<section class="panel training-form">
		<div class="panel-head"><div><h2>Training-only settings</h2><p>These settings belong to this execution plan, not the reusable model definition.</p></div></div>
		<label>Run name<input bind:value={runName} placeholder="e.g. resnet18-metadata-v1" /></label><label>Decision note<textarea rows="3" bind:value={description} placeholder="What are you testing and what decision will this inform?"></textarea></label>
		<div class="editor-fields"><label>Epochs<input type="number" min="1" bind:value={epochs} on:change={persistNormal} /></label><label>Batch size<input type="number" min="1" bind:value={batchSize} on:change={persistNormal} /></label><label>Learning rate<input type="number" min="0.000001" step="0.000001" bind:value={learningRate} on:change={persistNormal} /></label><label>Requested GPUs<input type="number" min="0" bind:value={gpuCount} /></label></div>
		<div class="panel-head section-break"><div><h3>Preprocessing & augmentation</h3><p>Applied only according to the sealed run contract.</p></div></div>
		<div class="editor-fields"><label>Image polarity<select bind:value={polarity} on:change={persistNormal}><option value="auto">Use recorded polarity</option><option value="false">Light foreground</option><option value="true">Dark foreground / invert</option></select></label><label>Rotation strength<input type="number" min="0" max="1" step="0.05" bind:value={rotation} on:change={persistNormal} /></label><label>Metadata Gaussian variance<input type="number" min="0" step="0.001" bind:value={metadataVariance} on:change={persistNormal} /></label></div>
		<div class="training-plan"><div><span>Will seal</span><strong>{draftId ? 'Draft revision + run overrides' : 'Recipe + run overrides'}</strong><small>Dataset fingerprint, initialization source, preprocessing, optimizer, and resources are recorded.</small></div><button disabled={creating || !runName.trim() || !datasetId || (!draftId && !recipeId) || (initialization !== 'scratch' && !sourceArtifactId)} on:click={plan}>{creating ? 'Creating…' : 'Create training plan'}</button></div>
		<AdvancedConfigEditor config={advancedOverrides} title="Advanced run options" description="These values are synchronized with the quick controls above and sealed only into this training plan." sections={runSections} on:change={(event) => syncAdvanced(event.detail.config)} />
	</section>
</div>
