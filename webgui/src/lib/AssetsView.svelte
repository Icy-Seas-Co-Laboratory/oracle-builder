<script lang="ts">
	import { api, type RecordValue } from '$lib/api';
	import EmptyState from '$lib/EmptyState.svelte';
	import FileExplorer from '$lib/FileExplorer.svelte';

	export let datasets: RecordValue[] = [];
	export let artifacts: RecordValue[] = [];
	export let recipes: RecordValue[] = [];
	export let onchanged: (message: string) => void | Promise<void>;
	export let onfailure: (message: string) => void;
	export let onuseDataset: (datasetId: string) => void;

	type ExplorerTarget = 'dataset' | 'artifacts' | 'config' | null;
	let explorerTarget: ExplorerTarget = null;
	let uploadKind: 'datasets' | 'configs' | 'models' = 'datasets';
	let uploadFile: File | null = null;
	let uploading = false;
	let uploadProgress = { transferredBytes: 0, totalBytes: 0 };
	let registeringDataset = false;
	let datasetMessage = '';
	let datasetError = '';
	let datasetPath = '';
	let assetPath = '';
	let recipeName = '';
	let recipeConfigPath = '';

	const display = (value: unknown) => value === null || value === undefined || value === '' ? '—' : String(value);
	const bytes = (value: number) => value < 1024 * 1024 ? `${Math.ceil(value / 1024)} KB` : `${(value / (1024 * 1024)).toFixed(value < 1024 * 1024 * 1024 ? 1 : 2)} ${value < 1024 * 1024 * 1024 ? 'MB' : 'GB'}`;
	const progressPercent = () => uploadProgress.totalBytes ? Math.round(uploadProgress.transferredBytes / uploadProgress.totalBytes * 100) : 0;
	const fail = (error: unknown, fallback: string) => onfailure(error instanceof Error ? error.message : fallback);

	async function ingestDataset() {
		if (!datasetPath.trim()) return;
		registeringDataset = true; datasetMessage = ''; datasetError = '';
		try {
			const dataset = await api.ingestDataset(datasetPath);
			datasetPath = '';
			await onchanged(`Registered dataset ${display(dataset.name)}.`);
			datasetMessage = `Registered ${display(dataset.name)}. It is now available in Datasets and New run.`;
		} catch (error) {
			datasetError = error instanceof Error ? error.message : 'Dataset registration failed.';
			fail(error, 'Dataset registration failed.');
		} finally { registeringDataset = false; }
	}

	async function createRecipe() {
		try {
			const recipe = await api.createRecipe({ name: recipeName, config_path: recipeConfigPath });
			recipeName = ''; recipeConfigPath = '';
			await onchanged(`Validated recipe ${display(recipe.name)}.`);
		} catch (error) { fail(error, 'Recipe validation failed.'); }
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
		uploadProgress = { transferredBytes: 0, totalBytes: uploadFile.size };
		try {
			const file = uploadFile;
			const result = await api.resumableUpload(uploadKind, file, (progress) => uploadProgress = progress);
			const path = String(result.path);
			if (uploadKind === 'datasets') { datasetPath = path; await ingestDataset(); }
			else if (uploadKind === 'configs') { recipeConfigPath = path; await onchanged(`Staged ${file.name}. Create a recipe to validate it.`); }
		else { await onchanged(`Staged ${file.name}. External model import is retired; this file remains an explicit transfer asset only.`); }
			uploadFile = null;
		} catch (error) { fail(error, 'Upload failed.'); }
		finally { uploading = false; }
	}

	function selectFile(path: string) {
		if (explorerTarget === 'dataset') datasetPath = path;
		else if (explorerTarget === 'artifacts') assetPath = path;
		else if (explorerTarget === 'config') recipeConfigPath = path;
		explorerTarget = null;
	}
</script>

<section class="panel upload-panel"><div><p class="eyebrow">UPLOAD</p><h2>Transfer a new asset</h2><p>Large files upload in resumable 16 MB chunks to the Orchestrator-owned staging area. Retrying after an interruption continues the same selected file.</p></div><div class="upload-controls"><label>Asset type<select bind:value={uploadKind} disabled={uploading}><option value="datasets">Frozen dataset (.sqlite)</option><option value="configs">Training config (.toml)</option><option value="models">Model file (.keras, .h5, .hdf5)</option></select></label><label class="file-input">Choose file<input type="file" accept={uploadKind === 'datasets' ? '.sqlite' : uploadKind === 'configs' ? '.toml' : '.keras,.h5,.hdf5'} on:change={chooseUpload} disabled={uploading} /><span>{uploadFile?.name ?? 'No file selected'}</span></label><button disabled={!uploadFile || uploading} on:click={uploadAsset}>{uploading ? `Uploading ${progressPercent()}%` : 'Upload asset'}</button>{#if uploading}<div class="upload-progress" role="status"><progress value={uploadProgress.transferredBytes} max={uploadProgress.totalBytes}></progress><span>{bytes(uploadProgress.transferredBytes)} of {bytes(uploadProgress.totalBytes)} transferred</span></div>{/if}</div></section>

<div class="two-col">
	<section class="panel"><div class="panel-head"><div><p class="eyebrow">DATASET</p><h2>Register a frozen dataset</h2><p>Registration indexes the SQLite contract without copying or rewriting it.</p></div></div><form on:submit|preventDefault={ingestDataset}><label>Frozen dataset path<div class="path-control"><input bind:value={datasetPath} required disabled={registeringDataset} /><button class="secondary small" type="button" on:click={() => explorerTarget = 'dataset'} disabled={registeringDataset}>Browse</button></div></label><button disabled={!datasetPath.trim() || registeringDataset}>{registeringDataset ? 'Registering…' : 'Register dataset'}</button></form>{#if datasetMessage}<p class="inline-success" role="status">{datasetMessage}</p>{/if}{#if datasetError}<p class="inline-error" role="alert">{datasetError}</p>{/if}</section>
	<section class="panel"><div class="panel-head"><div><p class="eyebrow">RECIPE</p><h2>Validate a training recipe</h2><p>Name and validate a reusable TOML training configuration.</p></div></div><form on:submit|preventDefault={createRecipe}><label>Recipe name<input bind:value={recipeName} required /></label><label>Configuration path<div class="path-control"><input bind:value={recipeConfigPath} required /><button class="secondary small" type="button" on:click={() => explorerTarget = 'config'}>Browse</button></div></label><button>Create recipe</button></form></section>
</div>

<section class="panel"><div class="panel-head"><div><p class="eyebrow">CATALOG</p><h2>Index existing artifacts</h2><p>Scan an approved directory for standard Oracle Builder artifact manifests.</p></div></div><form on:submit|preventDefault={scanArtifacts}><label>Artifact directory<div class="path-control"><input bind:value={assetPath} required /><button class="secondary small" type="button" on:click={() => explorerTarget = 'artifacts'}>Browse</button></div></label><button>Validate and index directory</button></form></section>

<section class="panel"><div class="panel-head"><div><h2>Registered datasets</h2><p>Frozen inputs ready for reproducible training.</p></div><span class="count">{datasets.length}</span></div>{#if datasets.length}<div class="cards">{#each datasets as dataset}<article class="asset-card"><div class="asset-icon">▣</div><div><strong>{display(dataset.name)}</strong><p>{display(dataset.dataset_type)} · {display(dataset.lifecycle)}</p><small>{display(dataset.path)}</small></div><div class="asset-actions"><a class="secondary small" href={api.datasetDownloadUrl(String(dataset.dataset_id))}>Download</a><button class="secondary small" on:click={() => onuseDataset(String(dataset.dataset_id))}>Use in experiment</button></div></article>{/each}</div>{:else}<EmptyState title="No datasets registered" text="Register a frozen SQLite dataset to make it available for training." />{/if}</section>

<section class="panel"><div class="panel-head"><div><h2>Training recipes</h2><p>Validated configurations ready for experiments.</p></div><span class="count">{recipes.length}</span></div>{#if recipes.length}<table><thead><tr><th>Recipe</th><th>Task</th><th>Model</th><th>Configuration</th></tr></thead><tbody>{#each recipes as recipe}<tr><td><strong>{display(recipe.name)}</strong><small>{display(recipe.recipe_id)}</small></td><td>{display(recipe.task)}</td><td>{display(recipe.model)}</td><td><small>{display(recipe.config_path)}</small></td></tr>{/each}</tbody></table>{:else}<EmptyState title="No training recipes" text="Create a recipe from a TOML configuration before planning training." />{/if}</section>

<section class="panel"><div class="panel-head"><div><h2>Model library</h2><p>A working index of portable model runs and products.</p></div><span class="count">{artifacts.length}</span></div>{#if artifacts.length}<table><thead><tr><th>Model artifact</th><th>Task</th><th>Family</th><th>Dataset</th><th>Status</th><th></th></tr></thead><tbody>{#each artifacts as artifact}<tr><td><strong>{display(artifact.name)}</strong><small>{display(artifact.artifact_id)}</small></td><td>{display(artifact.task)}</td><td>{display(artifact.architecture)}</td><td>{display(artifact.dataset_id)}</td><td><span class="status {display(artifact.status)}">{display(artifact.status)}</span></td><td><a class="secondary small" href={api.artifactDownloadUrl(String(artifact.artifact_id))}>Download</a></td></tr>{/each}</tbody></table>{:else}<EmptyState title="No model artifacts indexed" text="Import a model or scan an approved artifact directory." />{/if}</section>

{#if explorerTarget}<FileExplorer title={explorerTarget === 'config' ? 'Choose a training configuration' : explorerTarget === 'artifacts' ? 'Choose an artifact directory' : 'Choose a frozen dataset'} extensions={explorerTarget === 'config' ? ['.toml'] : explorerTarget === 'dataset' ? ['.sqlite'] : []} allowDirectory={explorerTarget === 'artifacts'} onselect={selectFile} onclose={() => explorerTarget = null} />{/if}
