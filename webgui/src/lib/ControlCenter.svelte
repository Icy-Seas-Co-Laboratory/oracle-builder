<script lang="ts">
	import { api, type RecordValue } from '$lib/api';
	import AssetsView from '$lib/AssetsView.svelte';
	import ArtifactEvidence from '$lib/ArtifactEvidence.svelte';

	export let datasets: RecordValue[] = [];
	export let recipes: RecordValue[] = [];
	export let artifacts: RecordValue[] = [];
	export let specifications: RecordValue[] = [];
	export let jobs: RecordValue[] = [];
	export let experiments: RecordValue[] = [];
	export let computeEndpoints: RecordValue[] = [];
	export let selectedDatasetId = '';
	export let onchanged: (message: string) => void | Promise<void>;
	export let onfailure: (message: string) => void;

	let showRunForm = false;
	let showAssets = false;
	let runName = '';
	let runDescription = '';
	let selectedRecipeId = '';
	let gpuCount = 0;
	let creating = false;
	let dispatchingId = '';
	let selectedRunId = '';
	let events: RecordValue[] = [];
	let modelSetup: RecordValue | null = null;
	let setupValues: Record<string, unknown> = {};
	let loadingSetup = false;
	let setupError = '';
	let observedRecipeId = '';
	let dispatchError = '';
	let editingSpecificationId = '';
	let selectedArtifactId = '';
	let artifactDetail: RecordValue | null = null;
	let artifactHistory: RecordValue[] = [];
	let loadingArtifact = false;
	let evidenceArtifactId = '';
	let rows: { spec: RecordValue; job: RecordValue | undefined; artifact: RecordValue | undefined; status: string }[] = [];
	let selected: { spec: RecordValue; job: RecordValue | undefined; artifact: RecordValue | undefined; status: string } | undefined;

	const text = (value: unknown) => value == null || value === '' ? '—' : String(value);
	const active = (status: unknown) => ['dispatching', 'submitted', 'queued', 'running', 'validating'].includes(String(status));
	const failed = (status: unknown) => ['failed', 'dispatch_failed', 'artifact_invalid', 'cancelled'].includes(String(status));
	const readyEndpoint = () => computeEndpoints.find((endpoint) => endpoint.status === 'ready');
	const availableGpus = () => Array.isArray(readyEndpoint()?.workers) ? (readyEndpoint()?.workers as RecordValue[]).reduce((count, worker) => count + (Array.isArray((worker.capabilities as RecordValue | undefined)?.gpus) ? ((worker.capabilities as RecordValue).gpus as unknown[]).length : 0), 0) : 0;
	const gpuChoices = () => Array.from({ length: availableGpus() + 1 }, (_, index) => index);
	const recipe = () => recipes.find((item) => String(item.recipe_id) === selectedRecipeId);
	const records = (value: unknown) => Array.isArray(value) ? value as RecordValue[] : [];
	const asRecord = (value: unknown) => value && typeof value === 'object' ? value as RecordValue : {};
	const numeric = (value: unknown) => Number.isFinite(Number(value));
	const artifact = () => asRecord(artifactDetail?.artifact);
	const artifactMetrics = () => Object.entries(asRecord(artifact().metrics)).filter(([, value]) => numeric(value));
	const setupFields = () => records(modelSetup?.fields);
	const fieldsFor = (section: string) => setupFields().filter((field) => String(field.path).startsWith(`${section}.`));
	const setupValue = (field: RecordValue) => setupValues[String(field.path)] ?? field.value;
	const setupDisplay = (field: RecordValue) => Array.isArray(setupValue(field)) ? (setupValue(field) as unknown[]).join(', ') : String(setupValue(field) ?? '');
	const fieldHelp = (path: unknown) => ({ 'model.filters': 'Number of learned feature maps. Higher values increase model capacity and compute.', 'model.dropout': 'Fraction of activations randomly disabled during training to reduce overfitting.', 'model.width_multiplier': 'Scales the width of an efficient model. Larger values use more parameters.', 'training.epochs': 'Maximum passes through the training data. Early stopping may finish sooner.', 'training.batch_size': 'Samples processed together. Larger batches use more memory.', 'training.learning_rate': 'Initial optimization step size. Too high can make training unstable.', 'data.input_shape': 'Image dimensions and channels expected by the model.', 'data.num_classes': 'Output classes. This is normally inferred from the selected dataset.' }[String(path)] ?? 'A validated value from this architecture’s default configuration.');
	const datasetName = (id: unknown) => text(datasets.find((item) => item.dataset_id === id)?.name ?? id);
	const experimentName = (id: unknown) => text(experiments.find((item) => item.experiment_id === id)?.name ?? id);
	const jobFor = (specificationId: unknown) => jobs.find((job) => job.specification_id === specificationId);
	const parametersFor = (spec: RecordValue) => spec.parameters && typeof spec.parameters === 'object' ? spec.parameters as RecordValue : {};
	const artifactFor = (specificationId: unknown) => {
		const job = jobFor(specificationId);
		return artifacts.find((artifact) => artifact.artifact_id === job?.artifact_id);
	};
	const statusFor = (spec: RecordValue) => String(jobFor(spec.specification_id)?.status ?? spec.status ?? 'planned');
	const statusLabel = (status: string) => ({ planned: 'Ready to start', dispatching: 'Starting', submitted: 'Queued', queued: 'Queued', running: 'Running', validating: 'Checking result', indexed: 'Complete', succeeded: 'Complete', failed: 'Failed', dispatch_failed: 'Could not start', artifact_invalid: 'Result needs attention', cancelled: 'Cancelled' }[status] ?? status);
	$: {
		const currentJobs = jobs;
		const currentArtifacts = artifacts;
		rows = specifications.map((spec) => {
			const job = currentJobs.find((item) => item.specification_id === spec.specification_id);
			const artifact = currentArtifacts.find((item) => item.artifact_id === job?.artifact_id);
			return { spec, job, artifact, status: String(job?.status ?? spec.status ?? 'planned') };
		});
	}
	$: selected = rows.find((row) => String(row.spec.specification_id) === selectedRunId);

	function openRun(datasetId = selectedDatasetId) {
		selectedDatasetId = datasetId;
		if (!selectedRecipeId && recipes.length) selectedRecipeId = String(recipes[0].recipe_id);
		showRunForm = true;
	}
	function setNested(target: Record<string, unknown>, path: string, value: unknown) { const keys = path.split('.'); let current = target; for (const key of keys.slice(0, -1)) current = (current[key] as Record<string, unknown>) ?? (current[key] = {} as Record<string, unknown>) as Record<string, unknown>; current[keys.at(-1)!] = value; }
	function nestedValue(source: Record<string, unknown>, path: string) { return path.split('.').reduce<unknown>((value, key) => value && typeof value === 'object' ? (value as Record<string, unknown>)[key] : undefined, source); }
	function configOverrides() { const overrides: Record<string, unknown> = {}; for (const field of setupFields()) setNested(overrides, String(field.path), setupValue(field)); return overrides; }
	function setSetupValue(field: RecordValue, event: Event) { const input = event.currentTarget as HTMLInputElement; let value: unknown = input.type === 'checkbox' ? input.checked : input.value; if (field.type === 'number') value = Number(value); if (field.type === 'list') value = String(value).split(',').map((part) => Number(part.trim())).filter(Number.isFinite); setupValues = { ...setupValues, [String(field.path)]: value }; }
	async function loadModelSetup() { if (!recipe()?.model) return; loadingSetup = true; setupError = ''; modelSetup = null; try { modelSetup = await api.modelSetup(String(recipe()?.model)); setupValues = Object.fromEntries(setupFields().map((field) => [String(field.path), field.value])); } catch (error) { setupError = error instanceof Error ? error.message : 'Model defaults are unavailable.'; } finally { loadingSetup = false; } }
	$: if (selectedRecipeId && selectedRecipeId !== observedRecipeId) { observedRecipeId = selectedRecipeId; void loadModelSetup(); }
	$: if (gpuCount > availableGpus()) gpuCount = availableGpus();
	async function dispatch(spec: RecordValue, announce = true) {
		const endpoint = readyEndpoint();
		if (!endpoint) throw new Error('No ready Oracle Serve endpoint is available. The run was saved and can be started when compute is ready.');
		dispatchingId = String(spec.specification_id);
		dispatchError = '';
		try {
			const check = await api.preflight(dispatchingId, String(endpoint.endpoint_id));
			if (!check.ready) throw new Error(Array.isArray(check.reasons) ? check.reasons.join('; ') : 'Preflight did not pass.');
			await api.dispatch(dispatchingId, String(endpoint.endpoint_id));
			if (announce) await onchanged(`Started ${text(spec.name)} on ${text(endpoint.name)}.`);
		} catch (error) {
			dispatchError = error instanceof Error ? error.message : 'Could not start this run.';
			if (announce) onfailure(dispatchError);
			throw error;
		} finally { dispatchingId = ''; }
	}
	async function startExisting(spec: RecordValue) { try { await dispatch(spec); } catch { /* The inline preflight error is the useful result. */ } }
	async function inspectArtifact(id: string) { selectedRunId = ''; selectedArtifactId = id; artifactDetail = null; artifactHistory = []; loadingArtifact = true; try { const [detail, history] = await Promise.all([api.artifactDetail(id), api.artifactHistory(id)]); artifactDetail = detail; artifactHistory = records(history.rows ?? history.history); } catch (error) { onfailure(error instanceof Error ? error.message : 'Could not load model details.'); } finally { loadingArtifact = false; } }
	async function startTraining() {
		if (!selectedDatasetId || !selectedRecipeId || !runName.trim()) return;
		creating = true;
		try {
			if (editingSpecificationId) {
				await api.updatePlannedSpecification(editingSpecificationId, { name: runName.trim(), resources: { gpu_count: gpuCount }, config_overrides: configOverrides() });
				selectedRunId = editingSpecificationId; editingSpecificationId = ''; showRunForm = false;
				await onchanged(`Updated ${runName.trim()}.`);
				return;
			}
			const created = await api.createTrainingExperiment({ name: runName.trim(), description: runDescription.trim(), dataset_id: selectedDatasetId, recipe_ids: [selectedRecipeId], seeds: [123], resources: { gpu_count: gpuCount }, config_overrides: configOverrides() });
			const planned = (await api.specifications(String(created.experiment_id))).specifications;
			const first = planned[0];
			if (!first) throw new Error('The run plan was created without a runnable entry.');
			selectedRunId = String(first.specification_id);
			showRunForm = false;
			runName = ''; runDescription = '';
			try {
				await dispatch(first, false);
				await onchanged(`Created and started ${text(first.name)}. It is now visible in the run queue.`);
			} catch (error) {
				await onchanged(`Created ${text(first.name)} and saved it in the run queue. Start it when compute is ready.`);
				onfailure(error instanceof Error ? error.message : 'The run was created but could not be started.');
			}
		} catch (error) { onfailure(error instanceof Error ? error.message : 'Could not create the training run.'); }
		finally { creating = false; }
	}
	async function selectRun(specificationId: string) {
		selectedArtifactId = '';
		selectedRunId = specificationId;
		const job = jobFor(specificationId);
		if (!job) { events = []; return; }
		try { events = (await api.jobEvents(String(job.job_id))).events; }
		catch { events = []; }
	}
	async function editRun(row: { spec: RecordValue }) {
		try {
			const detail = await api.specificationDetail(String(row.spec.specification_id));
			const specification = asRecord(detail.specification);
			const parameters = asRecord(specification.parameters);
			selectedDatasetId = String(parameters.dataset_id ?? '');
			selectedRecipeId = String(parameters.recipe_id ?? '');
			runName = String(specification.name ?? '');
			gpuCount = Number(asRecord(specification.resources).gpu_count ?? 0);
			runDescription = '';
			editingSpecificationId = String(specification.specification_id);
			await loadModelSetup();
			const savedConfig = asRecord(detail.config);
			setupValues = Object.fromEntries(setupFields().map((field) => [String(field.path), nestedValue(savedConfig, String(field.path)) ?? field.value]));
			showRunForm = true;
		} catch (error) { onfailure(error instanceof Error ? error.message : 'Could not open this run for editing.'); }
	}
</script>

<section class="control-hero">
	<div><p class="eyebrow">ORACLE BUILDER</p><h1>Run and organize your models.</h1><p>Choose data and a training recipe, start the run, and follow its result here.</p></div>
	<div class="control-actions"><button class="secondary" on:click={() => showAssets = !showAssets}>{showAssets ? 'Hide assets' : 'Add assets'}</button><button on:click={() => openRun()}>New training run</button></div>
</section>

{#if showRunForm}<section class="panel run-composer"><div class="panel-head"><div><p class="eyebrow">{editingSpecificationId ? 'EDIT PLANNED RUN' : 'NEW TRAINING RUN'}</p><h2>{editingSpecificationId ? 'Update this training run' : 'Choose what to train'}</h2><p>{editingSpecificationId ? 'Changes apply before the run is handed to Oracle Serve.' : 'This will start one run immediately when Oracle Serve is ready.'}</p></div><button class="secondary small" on:click={() => { showRunForm = false; editingSpecificationId = ''; }}>Close</button></div>
	<div class="form-grid"><label>Run name<input bind:value={runName} placeholder="e.g. ResNet baseline" /></label><label>Dataset<select bind:value={selectedDatasetId}>{#each datasets as dataset}<option value={dataset.dataset_id}>{text(dataset.name)} · {text(dataset.lifecycle)}</option>{/each}</select></label><label>Training recipe <span class="info-chip" title="A recipe supplies the architecture and validated baseline configuration for this run." aria-label="Recipe help">i</span><select bind:value={selectedRecipeId}>{#each recipes as item}<option value={item.recipe_id}>{text(item.name)} · {text(item.model)}</option>{/each}</select><small>Architecture: {text(recipe()?.model)} · {text(recipe()?.task)}</small></label><label>Requested GPUs <span class="info-chip" title="This is limited to GPUs reported by the currently ready compute worker." aria-label="GPU request help">i</span><select bind:value={gpuCount}>{#each gpuChoices() as count}<option value={count}>{count}</option>{/each}</select><small>{readyEndpoint() ? availableGpus() ? `${text(readyEndpoint()?.name)} has ${availableGpus()} GPU(s) available.` : `${text(readyEndpoint()?.name)} is CPU-only; this run will use 0 GPUs.` : 'No compute endpoint is ready.'}</small></label><label class="wide">Notes (optional)<textarea rows="2" bind:value={runDescription} placeholder="What are you testing?"></textarea></label></div>
	{#if loadingSetup}<p class="empty">Loading model options…</p>{:else if setupError}<p class="inline-error">Model options could not be loaded: {setupError}. The recipe defaults will still be used.</p>{:else if modelSetup}<section class="model-options"><div><p class="eyebrow">MODEL OPTIONS</p><h3>{text(recipe()?.model)} configuration</h3><p>These values begin with the architecture’s TOML defaults and apply only to this run.</p></div>{#each [['model', 'Model', 'Architecture capacity and regularization.'], ['training', 'Training', 'How the model learns from the dataset.'], ['data', 'Data', 'Input contract; class count is normally inferred.']] as group}{#if fieldsFor(group[0]).length}<fieldset><legend>{group[1]} <span class="info-chip" title={group[2]} aria-label={`${group[1]} option help`}>i</span></legend><div class="option-grid">{#each fieldsFor(group[0]) as field}<label>{String(field.path).replace(`${group[0]}.`, '').replaceAll('_', ' ')} <span class="info-chip" title={fieldHelp(field.path)} aria-label={`${String(field.path)} help`}>i</span>{#if field.type === 'boolean'}<input type="checkbox" checked={Boolean(setupValue(field))} on:change={(event) => setSetupValue(field, event)} />{:else}<input type={field.type === 'number' ? 'number' : 'text'} step={field.type === 'number' ? 'any' : undefined} value={setupDisplay(field)} on:change={(event) => setSetupValue(field, event)} />{/if}</label>{/each}</div></fieldset>{/if}{/each}</section>{/if}
	<div class="run-composer-footer"><span>{editingSpecificationId ? 'Save changes, then start the run from its queue entry.' : readyEndpoint() ? `Will start on ${text(readyEndpoint()?.name)}` : 'No compute endpoint is ready; this will be saved as Ready to start.'}</span><button disabled={creating || !runName.trim() || !selectedDatasetId || !selectedRecipeId} on:click={startTraining}>{creating ? 'Saving…' : editingSpecificationId ? 'Save run changes' : readyEndpoint() ? 'Create & start training' : 'Create run plan'}</button></div>
</section>{/if}

{#if showAssets}<section class="control-assets"><AssetsView {datasets} {artifacts} {recipes} bind:selectedDatasetId onchanged={onchanged} onfailure={onfailure} onuseDataset={openRun} onmodelcreated={onchanged} /></section>{/if}

<div class="control-grid">
	<section class="panel inventory"><div class="panel-head"><div><p class="eyebrow">ASSETS</p><h2>Datasets</h2></div><span class="count">{datasets.length}</span></div>{#each datasets as dataset}<button class="inventory-row" on:click={() => openRun(String(dataset.dataset_id))}><span>▣</span><div><strong>{text(dataset.name)}</strong><small>{text(dataset.dataset_type)} · {text(dataset.lifecycle)}</small></div><em>Train</em></button>{:else}<p class="empty">Add a frozen dataset to start training.</p>{/each}<div class="inventory-divider"></div><div class="panel-head"><div><h2>Training recipes</h2></div><span class="count">{recipes.length}</span></div>{#each recipes as item}<button class="inventory-row" on:click={() => { selectedRecipeId = String(item.recipe_id); openRun(); }}><span>◈</span><div><strong>{text(item.name)}</strong><small>{text(item.model)} · {text(item.task)}</small></div><em>Use</em></button>{:else}<p class="empty">Add a TOML recipe to define a model.</p>{/each}</section>

	<section class="panel run-board"><div class="panel-head"><div><p class="eyebrow">RUN QUEUE</p><h2>All training work</h2><p>Plans, active work, and completed results stay together.</p></div><span class="count">{rows.length}</span></div>{#if rows.length}<div class="run-list">{#each rows as row}<button class:selected={selectedRunId === row.spec.specification_id} class="run-row" on:click={() => selectRun(String(row.spec.specification_id))}><span class="status {row.status}">{statusLabel(row.status)}</span><span><strong>{text(row.spec.name)}</strong><small>{datasetName(parametersFor(row.spec).dataset_id)} · {text(row.spec.action)}</small></span><small>{text(row.job?.updated_at ?? row.spec.created_at)}</small>{#if row.status === 'planned'}<span class="row-cta">Start →</span>{:else}<span>›</span>{/if}</button>{/each}</div>{:else}<div class="empty"><strong>No runs yet</strong><p>Create a training run; it will appear here before and after it starts.</p><button on:click={() => openRun()}>New training run</button></div>{/if}</section>

	<section class="panel run-detail"><div class="panel-head"><div><p class="eyebrow">DETAILS</p><h2>{selected ? text(selected.spec.name) : 'Select a run'}</h2></div></div>{#if selected}{@const row = selected}<dl class="detail-list"><div><dt>Status</dt><dd><span class="status {row.status}">{statusLabel(row.status)}</span></dd></div><div><dt>Dataset</dt><dd>{datasetName(parametersFor(row.spec).dataset_id)}</dd></div><div><dt>Compute request</dt><dd>{text((row.spec.resources as RecordValue | undefined)?.gpu_count ?? 0)} GPU(s)</dd></div><div><dt>Worker</dt><dd>{text(row.job?.worker_id)}</dd></div></dl>{#if row.status === 'planned'}<div class="row-actions"><button class="secondary small" on:click={() => editRun(row)}>Edit run</button><button disabled={dispatchingId === row.spec.specification_id} on:click={() => startExisting(row.spec)}>{dispatchingId === row.spec.specification_id ? 'Checking…' : 'Start training'}</button></div>{#if dispatchError}<p class="inline-error" role="alert">{dispatchError}</p>{/if}{:else if active(row.status)}<p class="inline-success">This run is active. Its status refreshes automatically.</p>{:else if failed(row.status)}<p class="inline-error">{text(row.job?.error ?? 'This run needs attention.')}</p>{:else if row.artifact}<div class="result-summary"><strong>Model ready</strong><p>{text(row.artifact.name)} · {text(row.artifact.architecture)}</p></div>{/if}{#if events.length}<details open><summary>Run events</summary><ol class="event-list">{#each events as event}<li><time>{text(event.timestamp)}</time>{text(event.message)}</li>{/each}</ol></details>{/if}{:else}<p class="empty">Select a row to start it or inspect its current status.</p>{/if}</section>
</div>

{#if selected || selectedArtifactId}
	<button class="inspector-scrim" aria-label="Close inspector" on:click={() => { selectedRunId = ''; selectedArtifactId = ''; artifactDetail = null; }}></button>
{/if}

<section class="panel model-shelf"><div class="panel-head"><div><p class="eyebrow">RESULTS</p><h2>Completed models</h2><p>Select a model to inspect its metrics, configuration, and outputs.</p></div><span class="count">{artifacts.length}</span></div>{#if artifacts.length}<div class="artifact-strip">{#each artifacts as item}<button class:selected={selectedArtifactId === item.artifact_id} on:click={() => inspectArtifact(String(item.artifact_id))}><strong>{text(item.name)}</strong><small>{text(item.architecture)} · {text(item.task)}</small><span>{text(item.status)}</span></button>{/each}</div>{:else}<p class="empty">Completed models will appear here after their runs finish.</p>{/if}</section>

{#if selectedArtifactId}<section class="panel artifact-inspector"><div class="panel-head"><div><p class="eyebrow">MODEL INSPECTOR</p><h2>{text(artifact().name ?? artifacts.find((item) => item.artifact_id === selectedArtifactId)?.name)}</h2><p>{text(artifact().architecture)} · {text(artifact().task)} · {text(artifact().lifecycle)}</p></div><button class="secondary small" on:click={() => { selectedArtifactId = ''; artifactDetail = null; }}>Close</button></div>{#if loadingArtifact}<p class="empty">Loading model details…</p>{:else if artifactDetail}<div class="metrics compact-metrics">{#each artifactMetrics().slice(0, 6) as metric}<div><span>{metric[0].replaceAll('_', ' ')}</span><strong>{Number(metric[1]).toFixed(4)}</strong></div>{:else}<div><span>Dataset</span><strong>{datasetName(artifact().dataset_id)}</strong></div><div><span>Architecture</span><strong>{text(artifact().architecture)}</strong></div>{/each}</div><div class="two-col"><section><h3>Model information</h3><dl class="detail-list"><div><dt>Artifact status</dt><dd>{text(artifact().status)}</dd></div><div><dt>Dataset</dt><dd>{datasetName(artifact().dataset_id)}</dd></div><div><dt>Runtime</dt><dd>{text(asRecord(artifactDetail.runtime).runtime_seconds)} s</dd></div><div><dt>History rows</dt><dd>{artifactHistory.length}</dd></div></dl>{#if artifactDetail.architecture && asRecord(artifactDetail.architecture).summary}<details><summary>Architecture summary</summary><pre>{text(asRecord(artifactDetail.architecture).summary)}</pre></details>{/if}</section><section><h3>Configuration</h3><p class="empty">The recorded model and dataset contract are available below.</p><details><summary>View configuration</summary><pre>{JSON.stringify(artifactDetail.config, null, 2)}</pre></details></section></div><div class="row-actions"><button on:click={() => evidenceArtifactId = selectedArtifactId}>View outputs & evidence</button></div>{/if}</section>{/if}

{#if evidenceArtifactId}<ArtifactEvidence artifactId={evidenceArtifactId} onclose={() => evidenceArtifactId = ''} {onfailure} />{/if}
