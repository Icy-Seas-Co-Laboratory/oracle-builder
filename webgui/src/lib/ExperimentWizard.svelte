<script lang="ts">
	import { api, type RecordValue } from '$lib/api';
	import EmptyState from '$lib/EmptyState.svelte';

	export let datasets: RecordValue[] = [];
	export let recipes: RecordValue[] = [];
	export let selectedDatasetId = '';
	export let oncreated: (message: string) => void | Promise<void>;
	export let onfailure: (message: string) => void;
	export let ongoassets: () => void;

	type WizardStep = 1 | 2 | 3 | 4;
	let step: WizardStep = 1;
	let experimentName = '';
	let experimentDescription = '';
	let selectedRecipeIds: string[] = [];
	let repeatCount = 1;
	let baseSeed = 123;
	let gpuCount = 0;
	let creating = false;

	const display = (value: unknown) => value === null || value === undefined || value === '' ? '—' : String(value);
	const selectedDataset = () => datasets.find((dataset) => dataset.dataset_id === selectedDatasetId);
	const selectedRecipes = () => recipes.filter((recipe) => selectedRecipeIds.includes(String(recipe.recipe_id)));
	const readyForStep = (target: WizardStep) => target === 2 ? Boolean(selectedDatasetId) : target === 3 ? Boolean(experimentName.trim() && selectedRecipeIds.length) : true;

	function toggleRecipe(recipeId: string) {
		selectedRecipeIds = selectedRecipeIds.includes(recipeId) ? selectedRecipeIds.filter((value) => value !== recipeId) : [...selectedRecipeIds, recipeId];
	}

	async function createExperiment() {
		creating = true;
		try {
			const seeds = Array.from({ length: repeatCount }, (_, index) => baseSeed + index);
			await api.createTrainingExperiment({ name: experimentName, description: experimentDescription, dataset_id: selectedDatasetId, recipe_ids: selectedRecipeIds, seeds, resources: { gpu_count: gpuCount } });
			step = 1; experimentName = ''; experimentDescription = ''; selectedRecipeIds = [];
			await oncreated('Training experiment created. Its run specifications are ready for review.');
		} catch (error) { onfailure(error instanceof Error ? error.message : 'Could not create experiment.'); }
		finally { creating = false; }
	}
</script>

<section class="stepper" aria-label="Experiment workflow">{#each [['1', 'Dataset'], ['2', 'Model setup'], ['3', 'Run plan'], ['4', 'Review']] as item}<button class:current={step === Number(item[0])} class:complete={step > Number(item[0])} disabled={!readyForStep(Number(item[0]) as WizardStep)} on:click={() => step = Number(item[0]) as WizardStep}><span>{item[0]}</span>{item[1]}</button>{/each}</section>
<section class="panel wizard">
	{#if step === 1}
		<div class="wizard-heading"><p class="eyebrow">STEP 1 OF 4</p><h2>Choose the training asset</h2><p>Every run records the dataset fingerprint, revision, and lifecycle.</p></div>
		{#if datasets.length}<div class="choice-grid">{#each datasets as dataset}<button class="choice" class:selected={selectedDatasetId === dataset.dataset_id} on:click={() => selectedDatasetId = String(dataset.dataset_id)}><span class="choice-icon">▣</span><strong>{display(dataset.name)}</strong><small>{display(dataset.dataset_type)} · {display(dataset.lifecycle)}</small><code>{display(dataset.dataset_id)}</code></button>{/each}</div>{:else}<EmptyState title="A frozen dataset is required" text="Register a dataset first, then return to plan the experiment." action="Go to assets" onaction={ongoassets} />{/if}
	{:else if step === 2}
		<div class="wizard-heading"><p class="eyebrow">STEP 2 OF 4</p><h2>Describe the model study</h2><p>Choose validated recipes whose exact configurations will be snapshotted.</p></div>
		<div class="form-grid"><label>Experiment name<input bind:value={experimentName} required /></label><label>Purpose and decision<textarea bind:value={experimentDescription} rows="4"></textarea></label><div class="wide choice-grid">{#each recipes as recipe}<button class="choice" class:selected={selectedRecipeIds.includes(String(recipe.recipe_id))} on:click={() => toggleRecipe(String(recipe.recipe_id))}><span class="choice-icon">◈</span><strong>{display(recipe.name)}</strong><small>{display(recipe.task)} · {display(recipe.model)}</small><code>{display(recipe.config_path)}</code></button>{/each}</div>{#if !recipes.length}<div class="wide"><EmptyState title="Create a training recipe first" text="Recipes validate configurations before they enter experiments." action="Go to assets" onaction={ongoassets} /></div>{/if}</div>
	{:else if step === 3}
		<div class="wizard-heading"><p class="eyebrow">STEP 3 OF 4</p><h2>Set explicit run seeds</h2><p>Each candidate receives a distinct seed and generated configuration snapshot.</p></div>
		<div class="form-grid compact"><label>Starting seed<input type="number" min="0" bind:value={baseSeed} /></label><label>Seeds per recipe<input type="number" min="1" max="20" bind:value={repeatCount} /></label><label>Requested GPUs per run<input type="number" min="0" max="8" bind:value={gpuCount} /></label><div class="plan-callout"><strong>{selectedRecipeIds.length * repeatCount} run{selectedRecipeIds.length * repeatCount === 1 ? '' : 's'} will be created.</strong><span>Seeds: {Array.from({ length: repeatCount }, (_, index) => baseSeed + index).join(', ')}</span></div></div>
	{:else}
		<div class="wizard-heading"><p class="eyebrow">STEP 4 OF 4</p><h2>Review before creating the queue</h2><p>Nothing executes until a planned specification passes dispatch preflight.</p></div>
		<dl class="review"><div><dt>Dataset</dt><dd>{display(selectedDataset()?.name)}</dd></div><div><dt>Recipes</dt><dd>{selectedRecipes().length}</dd></div><div><dt>Seeds per recipe</dt><dd>{repeatCount}</dd></div><div><dt>Compute request</dt><dd>{gpuCount} GPU{gpuCount === 1 ? '' : 's'} per run</dd></div><div class="review-wide"><dt>Purpose</dt><dd>{display(experimentDescription)}</dd></div></dl>
	{/if}
	<div class="wizard-actions"><button class="secondary" disabled={step === 1} on:click={() => step = (step - 1) as WizardStep}>Back</button>{#if step < 4}<button disabled={!readyForStep((step + 1) as WizardStep)} on:click={() => step = (step + 1) as WizardStep}>Continue</button>{:else}<button disabled={creating} on:click={createExperiment}>{creating ? 'Creating…' : 'Create run specifications'}</button>{/if}</div>
</section>
