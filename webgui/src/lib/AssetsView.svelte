<script lang="ts">
	import { api, type RecordValue } from '$lib/api';
	import EmptyState from '$lib/EmptyState.svelte';
	import FileExplorer from '$lib/FileExplorer.svelte';

	export let datasets: RecordValue[] = [];
	export let artifacts: RecordValue[] = [];
	export let recipes: RecordValue[] = [];
	export let selectedDatasetId = '';
	export let onchanged: (message: string) => void | Promise<void>;
	export let onfailure: (message: string) => void;
	export let onuseDataset: (datasetId: string) => void;
	export let onmodelcreated: (message: string) => void | Promise<void>;

	type ExplorerTarget = 'dataset' | 'artifacts' | 'config' | 'model' | 'modelInfo' | null;
	let explorerTarget: ExplorerTarget = null;
	let uploadKind: 'datasets' | 'configs' | 'models' = 'datasets';
	let uploadFile: File | null = null;
	let uploading = false;
	let datasetPath = '';
	let assetPath = '';
	let recipeName = '';
	let recipeConfigPath = '';
	let modelImportName = '';
	let modelPath = '';
	let modelInfoPath = '';

	const display = (value: unknown) => value === null || value === undefined || value === '' ? '—' : String(value);
	const fail = (error: unknown, fallback: string) => onfailure(error instanceof Error ? error.message : fallback);

	async function ingestDataset() {
		try {
			const dataset = await api.ingestDataset(datasetPath);
			datasetPath = '';
			onuseDataset(String(dataset.dataset_id));
			await onchanged(`Registered dataset ${display(dataset.name)}.`);
		} catch (error) { fail(error, 'Dataset registration failed.'); }
	}

	async function createRecipe() {
		try {
			const recipe = await api.createRecipe({ name: recipeName, config_path: recipeConfigPath });
			recipeName = ''; recipeConfigPath = '';
			await onchanged(`Validated recipe ${display(recipe.name)}.`);
		} catch (error) { fail(error, 'Recipe validation failed.'); }
	}

	async function createModelImport() {
		try {
			await api.createModelImport({ name: modelImportName, model_path: modelPath, info_path: modelInfoPath, dataset_id: selectedDatasetId || null });
			modelImportName = ''; modelPath = ''; modelInfoPath = '';
			await onmodelcreated('Model import created. Review and dispatch it from the run queue.');
		} catch (error) { fail(error, 'Could not create model import.'); }
	}

	async function scanArtifacts() {
		try {
			const report = await api.scan(assetPath);
			const indexed = Array.isArray(report.artifacts) ? report.artifacts.length : 0;
			const skipped = Array.isArray(report.skipped) ? report.skipped.length : 0;
			await onchanged(`Catalog scan indexed ${indexed} artifact(s)${skipped ? ` and skipped ${skipped}` : ''}.`);
		} catch (error) { fail(error, 'Catalog scan failed.'); }
	}

	function chooseUpload(event: Event) {
		uploadFile = (event.currentTarget as HTMLInputElement).files?.[0] ?? null;
	}

	async function uploadAsset() {
		if (!uploadFile) return;
		uploading = true;
		try {
			const file = uploadFile;
			const result = await api.upload(uploadKind, file);
			const path = String(result.path);
			if (uploadKind === 'datasets') { datasetPath = path; await ingestDataset(); }
			else if (uploadKind === 'configs') { recipeConfigPath = path; await onchanged(`Staged ${file.name}. Create a recipe to validate it.`); }
			else { modelPath = path; await onchanged(`Staged ${file.name}. Add product metadata to create an import.`); }
			uploadFile = null;
		} catch (error) { fail(error, 'Upload failed.'); }
		finally { uploading = false; }
	}

	function selectFile(path: string) {
		if (explorerTarget === 'dataset') datasetPath = path;
		else if (explorerTarget === 'artifacts') assetPath = path;
		else if (explorerTarget === 'config') recipeConfigPath = path;
		else if (explorerTarget === 'model') modelPath = path;
		else if (explorerTarget === 'modelInfo') modelInfoPath = path;
		explorerTarget = null;
	}
</script>

<section class="panel upload-panel"><div><p class="eyebrow">UPLOAD</p><h2>Stage a new asset</h2><p>Uploads enter the Orchestrator-owned staging area before registration or validation.</p></div><div class="upload-controls"><label>Asset type<select bind:value={uploadKind}><option value="datasets">Frozen dataset (.sqlite)</option><option value="configs">Training config (.toml)</option><option value="models">Model file (.keras, .h5, .hdf5)</option></select></label><label class="file-input">Choose file<input type="file" accept={uploadKind === 'datasets' ? '.sqlite' : uploadKind === 'configs' ? '.toml' : '.keras,.h5,.hdf5'} on:change={chooseUpload} /><span>{uploadFile?.name ?? 'No file selected'}</span></label><button disabled={!uploadFile || uploading} on:click={uploadAsset}>{uploading ? 'Uploading…' : 'Upload asset'}</button></div></section>

<div class="two-col">
	<section class="panel"><div class="panel-head"><div><p class="eyebrow">DATASET</p><h2>Register a frozen dataset</h2><p>Registration indexes the SQLite contract without copying or rewriting it.</p></div></div><form on:submit|preventDefault={ingestDataset}><label>Frozen dataset path<div class="path-control"><input bind:value={datasetPath} required /><button class="secondary small" type="button" on:click={() => explorerTarget = 'dataset'}>Browse</button></div></label><button>Register dataset</button></form></section>
	<section class="panel"><div class="panel-head"><div><p class="eyebrow">RECIPE</p><h2>Validate a training recipe</h2><p>Name and validate a reusable TOML training configuration.</p></div></div><form on:submit|preventDefault={createRecipe}><label>Recipe name<input bind:value={recipeName} required /></label><label>Configuration path<div class="path-control"><input bind:value={recipeConfigPath} required /><button class="secondary small" type="button" on:click={() => explorerTarget = 'config'}>Browse</button></div></label><button>Create recipe</button></form></section>
</div>

<div class="two-col">
	<section class="panel"><div class="panel-head"><div><p class="eyebrow">EXTERNAL MODEL</p><h2>Import a model product</h2><p>Pair a supported model with Oracle Builder product metadata.</p></div></div><form on:submit|preventDefault={createModelImport}><label>Import name<input bind:value={modelImportName} required /></label><label>Model file<div class="path-control"><input bind:value={modelPath} required /><button class="secondary small" type="button" on:click={() => explorerTarget = 'model'}>Browse</button></div></label><label>Product metadata TOML<div class="path-control"><input bind:value={modelInfoPath} required /><button class="secondary small" type="button" on:click={() => explorerTarget = 'modelInfo'}>Browse</button></div></label><label>Dataset provenance (optional)<select bind:value={selectedDatasetId}><option value="">No dataset provenance</option>{#each datasets as dataset}<option value={dataset.dataset_id}>{display(dataset.name)}</option>{/each}</select></label><button>Create model import</button></form></section>
	<section class="panel"><div class="panel-head"><div><p class="eyebrow">CATALOG</p><h2>Index existing artifacts</h2><p>Scan an approved directory for standard Oracle Builder artifact manifests.</p></div></div><form on:submit|preventDefault={scanArtifacts}><label>Artifact directory<div class="path-control"><input bind:value={assetPath} required /><button class="secondary small" type="button" on:click={() => explorerTarget = 'artifacts'}>Browse</button></div></label><button>Validate and index directory</button></form></section>
</div>

<section class="panel"><div class="panel-head"><div><h2>Registered datasets</h2><p>Frozen inputs ready for reproducible training.</p></div><span class="count">{datasets.length}</span></div>{#if datasets.length}<div class="cards">{#each datasets as dataset}<article class="asset-card"><div class="asset-icon">▣</div><div><strong>{display(dataset.name)}</strong><p>{display(dataset.dataset_type)} · {display(dataset.lifecycle)}</p><small>{display(dataset.path)}</small></div><button class="secondary small" on:click={() => onuseDataset(String(dataset.dataset_id))}>Use in experiment</button></article>{/each}</div>{:else}<EmptyState title="No datasets registered" text="Register a frozen SQLite dataset to make it available for training." />{/if}</section>

<section class="panel"><div class="panel-head"><div><h2>Training recipes</h2><p>Validated configurations ready for experiments.</p></div><span class="count">{recipes.length}</span></div>{#if recipes.length}<table><thead><tr><th>Recipe</th><th>Task</th><th>Model</th><th>Configuration</th></tr></thead><tbody>{#each recipes as recipe}<tr><td><strong>{display(recipe.name)}</strong><small>{display(recipe.recipe_id)}</small></td><td>{display(recipe.task)}</td><td>{display(recipe.model)}</td><td><small>{display(recipe.config_path)}</small></td></tr>{/each}</tbody></table>{:else}<EmptyState title="No training recipes" text="Create a recipe from a TOML configuration before planning training." />{/if}</section>

<section class="panel"><div class="panel-head"><div><h2>Model library</h2><p>A working index of portable model runs and products.</p></div><span class="count">{artifacts.length}</span></div>{#if artifacts.length}<table><thead><tr><th>Model artifact</th><th>Task</th><th>Family</th><th>Dataset</th><th>Status</th></tr></thead><tbody>{#each artifacts as artifact}<tr><td><strong>{display(artifact.name)}</strong><small>{display(artifact.artifact_id)}</small></td><td>{display(artifact.task)}</td><td>{display(artifact.architecture)}</td><td>{display(artifact.dataset_id)}</td><td><span class="status {display(artifact.status)}">{display(artifact.status)}</span></td></tr>{/each}</tbody></table>{:else}<EmptyState title="No model artifacts indexed" text="Import a model or scan an approved artifact directory." />{/if}</section>

{#if explorerTarget}<FileExplorer title={explorerTarget === 'config' ? 'Choose a training configuration' : explorerTarget === 'model' ? 'Choose an external model file' : explorerTarget === 'modelInfo' ? 'Choose model product metadata' : explorerTarget === 'artifacts' ? 'Choose an artifact directory' : 'Choose a frozen dataset'} extensions={explorerTarget === 'config' || explorerTarget === 'modelInfo' ? ['.toml'] : explorerTarget === 'model' ? ['.keras', '.h5', '.hdf5'] : explorerTarget === 'dataset' ? ['.sqlite'] : []} allowDirectory={explorerTarget === 'artifacts'} onselect={selectFile} onclose={() => explorerTarget = null} />{/if}
