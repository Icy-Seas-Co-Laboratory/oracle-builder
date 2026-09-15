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
	let modelFamily = '';
	let setupError = '';
	let modelSetup: RecordValue | null = null;
	let setupValues: Record<string, unknown> = {};
	let loadingSetup = false;
	let modelPreview: RecordValue | null = null;
	let previewing = false;

	const display = (value: unknown) => value === null || value === undefined || value === '' ? '—' : String(value);
	const selectedDataset = () => datasets.find((dataset) => dataset.dataset_id === selectedDatasetId);
	const selectedRecipes = () => recipes.filter((recipe) => selectedRecipeIds.includes(String(recipe.recipe_id)));
	$: stepAvailable = {
		1: true,
		2: Boolean(selectedDatasetId),
		3: Boolean(selectedRecipeIds.length),
		4: Boolean(experimentName.trim() && selectedRecipeIds.length)
	};
	const asRecords = (value: unknown) => Array.isArray(value) ? value as RecordValue[] : [];
	const selectedTask = () => selectedRecipes()[0]?.task === 'segmentation' ? 'segmentation' : 'classification';
	const modelOptions = () => selectedTask() === 'segmentation' ? ['unet', 'residual_unet', 'unet_plus_plus'] : ['simple_cnn', 'resnet18', 'resnet34', 'resnet50', 'densenet121', 'densenet169', 'efficientnet_b0', 'efficientnet_b1'];
	const chosenArchitecture = () => modelFamily || String(selectedRecipes()[0]?.model ?? '');
	const setupFields = () => asRecords(modelSetup?.fields);
	const setupValue = (field: RecordValue) => setupValues[String(field.path)] ?? field.value;
	const setupDisplay = (field: RecordValue) => { const value = setupValue(field); return Array.isArray(value) ? value.map(String).join(', ') : String(value ?? ''); };
	function setNested(target: Record<string, unknown>, path: string, value: unknown) { const keys = path.split('.'); let current = target; for (const key of keys.slice(0, -1)) current = (current[key] as Record<string, unknown>) ?? (current[key] = {} as Record<string, unknown>) as Record<string, unknown>; current[keys.at(-1)!] = value; }
	function setSetupValue(field: RecordValue, event: Event) { const input = event.currentTarget as HTMLInputElement; let value: unknown = input.type === 'checkbox' ? input.checked : input.value; if (field.type === 'number') value = Number(value); if (field.type === 'list') value = String(value).split(',').map((part) => Number(part.trim())).filter(Number.isFinite); setupValues = { ...setupValues, [String(field.path)]: value }; }
	async function loadSetup() { const architecture = chosenArchitecture(); if (!architecture) return; loadingSetup = true; setupError = ''; modelPreview = null; try { modelSetup = await api.modelSetup(architecture); setupValues = Object.fromEntries(setupFields().map((field) => [String(field.path), field.value])); await updatePreview(); } catch (error) { setupError = error instanceof Error ? error.message : 'Could not load architecture defaults.'; } finally { loadingSetup = false; } }
	async function updatePreview() { if (!selectedDatasetId || !chosenArchitecture()) return; previewing = true; try { modelPreview = await api.modelPreview({ architecture: chosenArchitecture(), dataset_id: selectedDatasetId, overrides: configurationOverrides() }); } catch (error) { modelPreview = { error: error instanceof Error ? error.message : 'Could not build model preview.' }; } finally { previewing = false; } }
	function configurationOverrides() { const overrides: Record<string, unknown> = { run: { model: chosenArchitecture() } }; for (const field of setupFields()) setNested(overrides, String(field.path), setupValue(field)); return overrides; }

	function toggleRecipe(recipeId: string) {
		selectedRecipeIds = selectedRecipeIds.includes(recipeId) ? selectedRecipeIds.filter((value) => value !== recipeId) : [...selectedRecipeIds, recipeId];
		modelSetup = null;
		modelPreview = null;
		setupError = '';
		if (selectedRecipeIds.length) void loadSetup();
	}

	async function createExperiment() {
		creating = true;
		try {
			const seeds = Array.from({ length: repeatCount }, (_, index) => baseSeed + index);
			await api.createTrainingExperiment({ name: experimentName, description: experimentDescription, dataset_id: selectedDatasetId, recipe_ids: selectedRecipeIds, seeds, resources: { gpu_count: gpuCount }, config_overrides: configurationOverrides() });
			step = 1; experimentName = ''; experimentDescription = ''; selectedRecipeIds = [];
			await oncreated('Training experiment created. Each run has its own immutable configuration snapshot.');
		} catch (error) { setupError = error instanceof Error ? error.message : 'Could not create experiment.'; onfailure(setupError); }
		finally { creating = false; }
	}
</script>

<section class="stepper" aria-label="Experiment workflow">{#each [['1', 'Dataset'], ['2', 'Model setup'], ['3', 'Run plan'], ['4', 'Review']] as item}<button class:current={step === Number(item[0])} class:complete={step > Number(item[0])} disabled={!stepAvailable[Number(item[0]) as WizardStep]} on:click={() => step = Number(item[0]) as WizardStep}><span>{item[0]}</span>{item[1]}</button>{/each}</section>
<section class="panel wizard">
	{#if step === 1}
		<div class="wizard-heading"><p class="eyebrow">STEP 1 OF 4</p><h2>Choose the training asset</h2><p>Every run records the dataset fingerprint, revision, and lifecycle.</p></div>
		{#if datasets.length}<div class="choice-grid">{#each datasets as dataset}<button class="choice" class:selected={selectedDatasetId === dataset.dataset_id} on:click={() => selectedDatasetId = String(dataset.dataset_id)}><span class="choice-icon">▣</span><strong>{display(dataset.name)}</strong><small>{display(dataset.dataset_type)} · {display(dataset.lifecycle)}</small><code>{display(dataset.dataset_id)}</code></button>{/each}</div>{:else}<EmptyState title="A frozen dataset is required" text="Register a dataset first, then return to plan the experiment." action="Go to assets" onaction={ongoassets} />{/if}
	{:else if step === 2}
		<div class="wizard-heading"><p class="eyebrow">STEP 2 OF 4</p><h2>Set up this model run</h2><p>Start from a validated recipe, then apply run-specific changes. The recipe remains untouched; every run receives an immutable configuration snapshot.</p></div>
		<div class="form-grid"><div class="wide choice-grid">{#each recipes as recipe}<button class="choice" class:selected={selectedRecipeIds.includes(String(recipe.recipe_id))} on:click={() => toggleRecipe(String(recipe.recipe_id))}><span class="choice-icon">◈</span><strong>{display(recipe.name)}</strong><small>{display(recipe.task)} · {display(recipe.model)}</small><code>{display(recipe.config_path)}</code></button>{/each}</div>{#if selectedRecipeIds.length}<div class="wide run-setup"><div><h3>Architecture and parameters</h3><p>Defaults load automatically from Oracle Builder’s maintained TOML. Your edits apply only to this run.</p></div><label>Architecture<select bind:value={modelFamily} on:change={loadSetup}><option value="">{chosenArchitecture()} (recipe default)</option>{#each modelOptions() as option}<option value={option}>{option}</option>{/each}</select></label>{#if !modelSetup && !loadingSetup}<button class="secondary small" on:click={loadSetup}>Retry loading defaults</button>{/if}{#if loadingSetup}<p class="empty">Loading architecture defaults…</p>{:else if setupError}<p class="inline-error" role="alert">{setupError}. You can continue with the recipe configuration, but restart the Orchestrator to enable editable architecture defaults and previews.</p>{:else if modelSetup}<div class="dynamic-fields">{#each setupFields() as field}<label>{String(field.path).replaceAll('.', ' · ')}{#if field.type === 'boolean'}<input type="checkbox" checked={Boolean(setupValue(field))} on:change={(event) => setSetupValue(field, event)} />{:else}<input type={field.type === 'number' ? 'number' : 'text'} step={field.type === 'number' ? 'any' : undefined} value={setupDisplay(field)} on:change={(event) => setSetupValue(field, event)} />{/if}</label>{/each}</div><div class="preview-head"><div><h3>Model preview</h3><p>Built from the selected architecture and current values; it does not train or save anything.</p></div><button class="secondary small" disabled={previewing} on:click={updatePreview}>{previewing ? 'Updating…' : 'Update preview'}</button></div>{#if modelPreview?.error}<p class="inline-error">{display(modelPreview.error)}</p>{:else if modelPreview}<div class="architecture-card"><div><span>Parameters</span><strong>{Number(modelPreview.parameters).toLocaleString()}</strong></div><div><span>Layers</span><strong>{display(modelPreview.layers)}</strong></div><div><span>Output</span><strong>{display(modelPreview.output_shape)}</strong></div></div><details><summary>Model summary</summary><pre>{display(modelPreview.summary)}</pre></details>{/if}{/if}</div>{/if}{#if !recipes.length}<div class="wide"><EmptyState title="Create a training recipe first" text="Recipes validate configurations before they enter experiments." action="Go to assets" onaction={ongoassets} /></div>{/if}</div>
	{:else if step === 3}
		<div class="wizard-heading"><p class="eyebrow">STEP 3 OF 4</p><h2>Set explicit run seeds</h2><p>Each candidate receives a distinct seed and generated configuration snapshot.</p></div>
		<div class="form-grid compact"><label>Experiment name<input bind:value={experimentName} required /></label><label>Purpose and decision<textarea bind:value={experimentDescription} rows="4"></textarea></label><label>Starting seed<input type="number" min="0" bind:value={baseSeed} /></label><label>Seeds per recipe<input type="number" min="1" max="20" bind:value={repeatCount} /></label><label>Requested GPUs per run<input type="number" min="0" max="8" bind:value={gpuCount} /></label><div class="plan-callout"><strong>{selectedRecipeIds.length * repeatCount} run{selectedRecipeIds.length * repeatCount === 1 ? '' : 's'} will be created.</strong><span>Seeds: {Array.from({ length: repeatCount }, (_, index) => baseSeed + index).join(', ')}</span></div></div>
	{:else}
		<div class="wizard-heading"><p class="eyebrow">STEP 4 OF 4</p><h2>Review before creating the queue</h2><p>Nothing executes until a planned specification passes dispatch preflight.</p></div>
		<dl class="review"><div><dt>Dataset</dt><dd>{display(selectedDataset()?.name)}</dd></div><div><dt>Recipes</dt><dd>{selectedRecipes().length}</dd></div><div><dt>Seeds per recipe</dt><dd>{repeatCount}</dd></div><div><dt>Compute request</dt><dd>{gpuCount} GPU{gpuCount === 1 ? '' : 's'} per run</dd></div><div><dt>Architecture</dt><dd>{chosenArchitecture() || 'Select a recipe'}</dd></div><div class="review-wide"><dt>Purpose</dt><dd>{display(experimentDescription)}</dd></div></dl>{#if setupError}<p class="inline-error" role="alert">{setupError}</p>{/if}
	{/if}
	<div class="wizard-actions"><button class="secondary" disabled={step === 1} on:click={() => step = (step - 1) as WizardStep}>Back</button>{#if step < 4}<button disabled={!stepAvailable[(step + 1) as WizardStep]} on:click={() => step = (step + 1) as WizardStep}>Continue</button>{:else}<button disabled={creating} on:click={createExperiment}>{creating ? 'Creating…' : 'Create run specifications'}</button>{/if}</div>
</section>
