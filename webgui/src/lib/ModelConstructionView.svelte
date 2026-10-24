<script lang="ts">
	import { onMount } from 'svelte';
	import { api, type RecordValue } from '$lib/api';
	import ConfigField from '$lib/config/ConfigField.svelte';
	import StageAdvancedOptions from '$lib/StageAdvancedOptions.svelte';
	import TrainingControls from '$lib/TrainingControls.svelte';
	import PosthocControls from '$lib/PosthocControls.svelte';
	import SelfSupervisedControls from '$lib/SelfSupervisedControls.svelte';
	import { configWithValue, copyConfig, getPath, setPath } from '$lib/config/paths';
	import type { ConfigFieldDefinition, ConfigurationSchema } from '$lib/config/types';

	export let onnotice: (message: string) => void;
	export let onfailure: (message: string) => void;
	export let ontrain: (definitionId: string) => void;

	const records = (value: unknown): RecordValue[] => Array.isArray(value) ? value as RecordValue[] : [];
	const record = (value: unknown): RecordValue => value && typeof value === 'object' ? value as RecordValue : {};
	const text = (value: unknown) => value == null || value === '' ? '—' : String(value);
	const value = getPath;
	const copy = copyConfig;
	const definitionTask = (definition: RecordValue) => text(
		getPath(record(definition.config), 'run.task', definition.task ?? record(definition.architecture).task)
	);
	const definitionModel = (definition: RecordValue) => text(record(definition.architecture).family ?? getPath(record(definition.config), 'encoder.family'));
	const definitionVariant = (definition: RecordValue) => text(record(definition.architecture).variant ?? getPath(record(definition.config), 'encoder.variant'));
	const optionLabel = (item: string) => item.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
	const formatTimestamp = (item: unknown) => {
		if (typeof item !== 'string') return '—';
		const date = new Date(item);
		return Number.isNaN(date.getTime()) ? item : date.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
	};
	type DefinitionSort = 'name' | 'created_at' | 'updated_at' | 'model' | 'variant';

	let definitions: RecordValue[] = [];
	let templates: RecordValue[] = [];
	let templateId = '';
	let taskChoice = '';
	let modelChoice = '';
	let variantChoice = '';
	let definitionSort: DefinitionSort = 'name';
	let sortAscending = true;
	let selected: RecordValue | null = null;
	let creating = false;
	let name = 'Untitled V2 classifier';
	let config: RecordValue = {};
	let loading = true;
	let saving = false;
	let report: RecordValue | null = null;
	let dragIndex: number | null = null;
	let schemaFields: RecordValue[] = [];
	let schemaDefaults: RecordValue = {};
	let appliedTemplateId = '';
	let metadataSources: { value: string; label: string }[] = [
		{ value: 'roi.original_width_px', label: 'ROI original width (px)' },
		{ value: 'roi.original_height_px', label: 'ROI original height (px)' },
		{ value: 'roi.bounding_box_area_px', label: 'ROI bounding-box area (px²)' },
		{ value: 'roi.aspect_ratio', label: 'ROI aspect ratio' }
	];
	const transforms = [['none', 'None'], ['log', 'Log (x > 0)'], ['log1p', 'Log(1 + x) (x > −1)'], ['sqrt', 'Square root (x ≥ 0)'], ['inverse', 'Reciprocal (x ≠ 0)']];
	$: selectedTemplate = templates.find((item) => String(item.template_id) === templateId);
	$: currentTask = String(value(config, 'run.task', selectedTemplate?.task ?? 'classification'));
	$: taskChoices = [...new Set(templates.map((item) => String(item.task ?? 'classification')))].sort();
	$: if (taskChoices.length && !taskChoices.includes(taskChoice)) taskChoice = taskChoices[0];
	$: modelChoices = [...new Set(templates.filter((item) => String(item.task ?? 'classification') === taskChoice).map((item) => String(item.model ?? record(item.architecture).family ?? 'unknown')))].sort();
	$: if (modelChoices.length && !modelChoices.includes(modelChoice)) modelChoice = modelChoices[0];
	$: variantChoices = [...new Set(templates.filter((item) => String(item.task ?? 'classification') === taskChoice && String(item.model ?? record(item.architecture).family ?? 'unknown') === modelChoice).map((item) => String(item.variant ?? record(item.architecture).variant ?? 'default')))].sort();
	$: if (variantChoices.length && !variantChoices.includes(variantChoice)) variantChoice = variantChoices[0];
	$: choiceTemplate = templates.find((item) => String(item.task ?? 'classification') === taskChoice && String(item.model ?? record(item.architecture).family ?? 'unknown') === modelChoice && String(item.variant ?? record(item.architecture).variant ?? 'default') === variantChoice);
	$: if (choiceTemplate && templateId !== String(choiceTemplate.template_id)) templateId = String(choiceTemplate.template_id);
	$: sortedDefinitions = [...definitions].sort((left, right) => {
		const stringValue = (definition: RecordValue) => definitionSort === 'model' ? definitionModel(definition) : definitionSort === 'variant' ? definitionVariant(definition) : definitionSort === 'name' ? text(definition.name) : text(definition[definitionSort]);
		const compared = stringValue(left).localeCompare(stringValue(right), undefined, { numeric: true, sensitivity: 'base' });
		return sortAscending ? compared : -compared;
	});
	$: if (!selected && selectedTemplate && appliedTemplateId !== templateId) { config = copy(record(selectedTemplate.config)); appliedTemplateId = templateId; }
	function fields(): RecordValue[] { const current = value(config, 'metadata.fields', []); return records(current); }
	function mutate(path: string, raw: unknown) { config = configWithValue(config, path, raw); report = null; }
	function field(path: string): ConfigFieldDefinition { return (schemaFields.find((item) => item.path === path) as ConfigFieldDefinition | undefined) ?? { path, label: path.split('.').at(-1)?.replaceAll('_', ' ') }; }
	function addMetadata() { const next = copy(config); const current = fields(); const source = metadataSources[0]?.value ?? 'roi.original_width_px'; setPath(next, 'metadata.fields', [...current, { name: `field_${current.length + 1}`, source, transform: 'none' }]); config = next; }
	function addRoiContext() {
		const next = copy(config);
		const current = [...fields()];
		for (const preset of [{ name: 'roi_area', source: 'roi.bounding_box_area_px', transform: 'log' }, { name: 'roi_aspect_ratio', source: 'roi.aspect_ratio', transform: 'none' }]) {
			if (current.some((item) => item.source === preset.source)) continue;
			let name = preset.name;
			for (let suffix = 2; current.some((item) => item.name === name); suffix++) name = `${preset.name}_${suffix}`;
			current.push({ ...preset, name });
		}
		setPath(next, 'metadata.fields', current);
		config = next;
		report = null;
	}
	function removeMetadata(index: number) { const next = copy(config); setPath(next, 'metadata.fields', fields().filter((_, item) => item !== index)); config = next; }
	function reorderMetadata(index: number) { if (dragIndex === null || dragIndex === index) return; const items = [...fields()]; const [moved] = items.splice(dragIndex, 1); items.splice(index, 0, moved); const next = copy(config); setPath(next, 'metadata.fields', items); config = next; dragIndex = null; }
	function updateMetadata(index: number, key: string, raw: string) { const next = copy(config); const items = fields().map((item, itemIndex) => itemIndex === index ? { ...item, [key]: raw } : item); setPath(next, 'metadata.fields', items); config = next; }
	function friendlyMetadataLabel(path: string) { return path.replace(/^metadata\./, 'Metadata · ').replaceAll('.', ' · ').replaceAll('_', ' '); }
	async function loadMetadataSources() {
		try {
			const datasets = (await api.datasets()).datasets.filter((dataset) => String(dataset.dataset_type ?? '').toLowerCase() === 'classification');
			const results = await Promise.all(datasets.map(async (dataset) => api.datasetScalarFields(String(dataset.dataset_id ?? dataset.id)).catch(() => ({ fields: [] as RecordValue[], item_count: 0 }))));
			const discovered = results.flatMap((result, index) => records(result.fields).map((field) => ({ value: String(field.source ?? ''), label: `${String(field.label ?? friendlyMetadataLabel(String(field.source ?? '')))} · ${String(datasets[index]?.name ?? datasets[index]?.dataset_id ?? '')}` })).filter((field) => field.value));
			metadataSources = [...metadataSources, ...discovered].filter((source, index, list) => list.findIndex((item) => item.value === source.value) === index);
		} catch { /* Built-in ROI measurements remain available without dataset preview access. */ }
	}
	function syncCreationChoice(template: RecordValue | undefined) {
		if (!template) return;
		taskChoice = String(template.task ?? 'classification');
		modelChoice = String(template.model ?? record(template.architecture).family ?? 'unknown');
		variantChoice = String(template.variant ?? record(template.architecture).variant ?? 'default');
		templateId = String(template.template_id);
	}
	function setSort(column: DefinitionSort) { if (definitionSort === column) sortAscending = !sortAscending; else { definitionSort = column; sortAscending = column === 'name'; } }
	async function load() { loading = true; try { const [definitionResult, templateResult] = await Promise.all([api.modelDefinitions(), api.modelDefinitionTemplates()]); definitions = records(definitionResult.definitions); templates = records(templateResult.templates); syncCreationChoice(templates.find((item) => String(item.template_id) === templateId) ?? templates[0]); } catch { definitions = []; templates = []; } finally { loading = false; } }
	async function loadSchema() {
		try {
			const schema = await api.configurationSchema() as ConfigurationSchema;
			schemaFields = schema.fields ?? [];
			schemaDefaults = schema.defaults ?? {};
			// Defaults are authored and resolved by the server.  Do not add browser
			// fallback presets: doing so would create a second configuration authority.
			if (!selected && !Object.keys(config).length) config = copy(schemaDefaults);
		} catch { schemaFields = []; }
	}
	function startNew() { selected = null; creating = true; name = `Untitled ${optionLabel(taskChoice || 'classification')} definition`; appliedTemplateId = ''; config = copy(record(templates.find((item) => String(item.template_id) === templateId)?.config)); report = null; }
	async function open(id: string) { try { selected = record(await api.modelDefinition(id)); creating = false; name = String(selected.name ?? 'Untitled V2 definition'); config = copy(record(selected.config)); report = null; } catch (error) { onfailure(error instanceof Error ? error.message : 'Could not open model definition.'); } }
	async function create() { if (!templateId) { onfailure('Choose a maintained V2 template first.'); return; } saving = true; try { selected = record(await api.createModelDefinition({ name, template_id: templateId, config })); await load(); if (selected.definition_id) await open(String(selected.definition_id)); onnotice('Created a versioned V2 model definition from the maintained template.'); } catch (error) { onfailure(error instanceof Error ? error.message : 'Could not create model definition.'); } finally { saving = false; } }
	async function save() { if (!selected?.definition_id) return create(); saving = true; try { selected = record(await api.updateModelDefinition(String(selected.definition_id), { name, config, expected_revision: Number(selected.revision) })); await load(); if (selected.definition_id) await open(String(selected.definition_id)); onnotice('Saved a new immutable definition revision.'); } catch (error) { onfailure(error instanceof Error ? error.message : 'Could not save model definition.'); } finally { saving = false; } }
	async function duplicate(definition: RecordValue) { if (!definition.definition_id) return; saving = true; try { const result = await api.duplicateModelDefinition(String(definition.definition_id), { revision: Number(definition.revision), lineage_kind: 'sensitivity' }); selected = record(result); creating = false; await load(); if (selected.definition_id) await open(String(selected.definition_id)); onnotice('Created the next version from the pinned definition revision.'); } catch (error) { onfailure(error instanceof Error ? error.message : 'Could not duplicate model definition.'); } finally { saving = false; } }
	async function remove(definition: RecordValue) {
		const id = String(definition.definition_id ?? ''); if (!id || saving) return;
		if (!window.confirm(`Delete “${text(definition.name)}”? Definitions that are pinned by a queued or historical run are protected.`)) return;
		saving = true;
		try { await api.deleteModelDefinition(id); if (selected?.definition_id === id) { selected = null; creating = false; config = {}; } await load(); onnotice('Unused model definition deleted.'); }
		catch (error) { onfailure(error instanceof Error ? error.message : 'Could not delete model definition.'); }
		finally { saving = false; }
	}

	onMount(() => { void load(); void loadSchema(); void loadMetadataSources(); });
</script>

<section class="studio-intro">
	<div><p class="eyebrow">MODEL DEFINITIONS</p><h1>Choose a definition, then construct it.</h1><p>Browse reusable, versioned definitions in the library. New definitions start from a task, model, and variant; maintained TOMLs supply the validated defaults behind that choice.</p></div>
</section>

{#if loading}<section class="panel"><p class="empty">Loading model drafts…</p></section>{:else}
<div class="construction-grid">
	<aside class="draft-rail panel"><div class="panel-head"><div><h2>Definition library</h2><p>Reusable V2 configurations.</p></div><span class="count">{definitions.length}</span></div><button class="new-definition" on:click={startNew}>+ Define new model</button>{#if definitions.length}<div class="definition-table-wrap"><table class="definition-table"><thead><tr><th><button on:click={() => setSort('name')}>Name {definitionSort === 'name' ? sortAscending ? '↑' : '↓' : ''}</button></th><th><button on:click={() => setSort('created_at')}>Created {definitionSort === 'created_at' ? sortAscending ? '↑' : '↓' : ''}</button></th><th><button on:click={() => setSort('updated_at')}>Edited {definitionSort === 'updated_at' ? sortAscending ? '↑' : '↓' : ''}</button></th><th><button on:click={() => setSort('model')}>Model {definitionSort === 'model' ? sortAscending ? '↑' : '↓' : ''}</button></th><th><button on:click={() => setSort('variant')}>Variant {definitionSort === 'variant' ? sortAscending ? '↑' : '↓' : ''}</button></th><th><span class="sr-only">Actions</span></th></tr></thead><tbody>{#each sortedDefinitions as definition}<tr class:selected={selected?.definition_id === definition.definition_id}><td><button class="definition-name" on:click={() => open(String(definition.definition_id))}><strong>{text(definition.name)}</strong><small>{definitionTask(definition)} · revision {text(definition.revision)}</small></button></td><td>{formatTimestamp(definition.created_at)}</td><td>{formatTimestamp(definition.updated_at)}</td><td>{definitionModel(definition)}</td><td>{definitionVariant(definition)}</td><td><div class="definition-actions"><button class="secondary small" disabled={saving} on:click={() => duplicate(definition)}>Duplicate</button><button class="quiet-button small delete-definition" disabled={saving} on:click={() => remove(definition)}>Delete</button></div></td></tr>{/each}</tbody></table></div>{:else}<div class="library-empty"><strong>Start your first runnable path</strong><small>1. Choose task, model, and variant<br />2. Freeze a training set<br />3. Validate it in Queue</small></div>{/if}</aside>
	<div class="construction-main">
		{#if !selected && !creating}<section class="constructor-empty panel"><p class="eyebrow">NEXT STEP · 1 OF 3</p><h2>{definitions.length ? 'Open a definition to continue' : 'Create your first model definition'}</h2><p>{definitions.length ? 'Open a versioned definition from the library to review or create its next revision. When it is ready, Queue pairs that exact revision with frozen data.' : 'Begin from a maintained template. Saving creates a reusable, versioned definition; Queue later pairs it with a frozen training set and validates compute.'}</p><button on:click={startNew}>{definitions.length ? 'Define a new model' : 'Choose a maintained template'}</button></section>{:else}
		<section class="parameter-estimate panel" aria-live="polite"><div><p class="eyebrow">VERSIONED DEFINITION</p><h2>{selected ? `Revision ${String(selected.revision)}` : 'New definition'}</h2><p>Output classes are derived from the frozen training set. Queue validation builds the exact model, materializes the TOML, and checks the selected compute endpoint.</p></div></section>
		<SelfSupervisedControls {config} fieldDefinitions={schemaFields} on:change={(event) => { config = event.detail.config; report = null; }} />
		{#if currentTask === 'segmentation'}<section class="panel task-specific-controls"><header><div><p class="eyebrow">SEGMENTATION WORKFLOW</p><h2>Spatial target controls</h2><p>These fields are specific to candidate-mask refinement and spatial output geometry.</p></div></header><div class="editor-fields"><ConfigField definition={field('data.output_shape')} value={value(config, 'data.output_shape')} {config} on:change={(event) => mutate('data.output_shape', event.detail.value)} /><ConfigField definition={field('training.segmentation_target')} value={value(config, 'training.segmentation_target')} {config} on:change={(event) => mutate('training.segmentation_target', event.detail.value)} /><ConfigField definition={field('tiling.enabled')} value={value(config, 'tiling.enabled')} {config} on:change={(event) => mutate('tiling.enabled', event.detail.value)} /></div></section>{/if}
		{#if currentTask === 'classification'}<section class="panel task-specific-controls"><header><div><p class="eyebrow">CLASSIFICATION WORKFLOW</p><h2>Class prediction controls</h2><p>The frozen training set supplies the final class vocabulary at queue-validation time.</p></div></header><div class="editor-fields"><ConfigField definition={field('classifier.type')} value={value(config, 'classifier.type')} {config} on:change={(event) => mutate('classifier.type', event.detail.value)} /><ConfigField definition={field('training.class_weights.mode')} value={value(config, 'training.class_weights.mode')} {config} on:change={(event) => mutate('training.class_weights.mode', event.detail.value)} />{#if value(config, 'training.class_weights.mode') === 'power_law'}<ConfigField definition={field('training.class_weights.alpha')} value={value(config, 'training.class_weights.alpha')} {config} on:change={(event) => mutate('training.class_weights.alpha', event.detail.value)} />{/if}</div></section>{/if}
		<section class="panel module-editor builder-section"><header><div><p class="eyebrow">01 · DEFINITION DETAILS</p><h2>Name & starting point</h2><p>Choose a task, model family, and supported variant. The matching maintained configuration supplies safe, versioned defaults.</p></div></header><div class="editor-fields"><label>Definition name<input bind:value={name} placeholder="e.g. exp001-resnet-128" /></label>{#if selected}<label>Task<input value={currentTask} disabled /></label><label>Model<input value={definitionModel(selected)} disabled /></label><label>Variant<input value={definitionVariant(selected)} disabled /></label>{:else}<label>Task<select bind:value={taskChoice}>{#each taskChoices as task}<option value={task}>{optionLabel(task)}</option>{/each}</select></label><label>Model<select bind:value={modelChoice}>{#each modelChoices as model}<option value={model}>{optionLabel(model)}</option>{/each}</select></label><label>Variant<select bind:value={variantChoice}>{#each variantChoices as variant}<option value={variant}>{optionLabel(variant)}</option>{/each}</select></label>{/if}</div><p class="template-capability">{#if selected}This definition remains pinned to its saved task and architecture. Duplicate it to explore a different starting point.{:else}<strong>{text(selectedTemplate?.name)}</strong> provides the maintained default configuration. Change the three selectors to load a compatible alternative before creating.{/if}</p></section>
		<section class="panel module-editor builder-section"><header><div><p class="eyebrow">02 · INPUT</p><h2>Input & geometry</h2><p>Channels, ROI placement, and preprocessing.</p></div></header><div class="editor-fields"><ConfigField definition={field('data.input_shape')} value={value(config, 'data.input_shape')} {config} on:change={(event) => mutate('data.input_shape', event.detail.value)} /><ConfigField definition={field('input.channels')} value={value(config, 'input.channels')} {config} on:change={(event) => mutate('input.channels', event.detail.value)} /><ConfigField definition={field('input.geometry.type')} value={value(config, 'input.geometry.type')} {config} on:change={(event) => mutate('input.geometry.type', event.detail.value)} /></div><StageAdvancedOptions {config} fieldDefinitions={schemaFields} description="Input refinement, geometry, and tiling." sections={['data', 'input', 'preprocessing', 'tiling']} excludePaths={['data.input_shape', 'input.channels', 'input.geometry.type']} on:change={(event) => { config = event.detail.config; report = null; }} /></section>
		<section class="panel module-editor builder-section"><header><div><p class="eyebrow">03 · CNN ENCODER</p><h2>Backbone & stem</h2><p>Architecture, stem behavior, and normalization are defined by the V2 catalog.</p></div></header><div class="editor-fields"><ConfigField definition={field('encoder.family')} value={value(config, 'encoder.family')} {config} on:change={(event) => mutate('encoder.family', event.detail.value)} /><ConfigField definition={field('encoder.variant')} value={value(config, 'encoder.variant')} {config} on:change={(event) => mutate('encoder.variant', event.detail.value)} /><ConfigField definition={field('stem.kernel_size')} value={value(config, 'stem.kernel_size')} {config} on:change={(event) => mutate('stem.kernel_size', event.detail.value)} /><ConfigField definition={field('stem.stride')} value={value(config, 'stem.stride')} {config} on:change={(event) => mutate('stem.stride', event.detail.value)} /><ConfigField definition={field('normalization.type')} value={value(config, 'normalization.type')} {config} on:change={(event) => mutate('normalization.type', event.detail.value)} />{#if value(config, 'encoder.family') === 'simple_cnn'}<ConfigField definition={field('encoder.num_layers')} value={value(config, 'encoder.num_layers', 3)} {config} on:change={(event) => mutate('encoder.num_layers', event.detail.value)} /><ConfigField definition={field('encoder.base_filters')} value={value(config, 'encoder.base_filters', 32)} {config} on:change={(event) => mutate('encoder.base_filters', event.detail.value)} /><ConfigField definition={field('encoder.filter_multiplier')} value={value(config, 'encoder.filter_multiplier', 2)} {config} on:change={(event) => mutate('encoder.filter_multiplier', event.detail.value)} />{/if}</div><StageAdvancedOptions {config} fieldDefinitions={schemaFields} description="Encoder topology, stem, and normalization." sections={['architecture', 'encoder', 'stem', 'normalization']} excludePaths={['encoder.family', 'encoder.variant', 'encoder.num_layers', 'encoder.base_filters', 'encoder.filter_multiplier', 'stem.kernel_size', 'stem.stride', 'normalization.type']} on:change={(event) => { config = event.detail.config; report = null; }} /></section>
		<section class="panel module-editor builder-section"><header><div><p class="eyebrow">04 · REPRESENTATION</p><h2>Pooling & embedding</h2><p>Convert encoder features into the image representation.</p></div></header><div class="editor-fields"><ConfigField definition={field('pooling.type')} value={value(config, 'pooling.type')} {config} on:change={(event) => mutate('pooling.type', event.detail.value)} /><ConfigField definition={field('image_embedding.projection.type')} value={value(config, 'image_embedding.projection.type')} {config} on:change={(event) => mutate('image_embedding.projection.type', event.detail.value)} /><ConfigField definition={field('image_embedding.projection.output_dim')} value={value(config, 'image_embedding.projection.output_dim')} {config} on:change={(event) => mutate('image_embedding.projection.output_dim', event.detail.value)} /></div><StageAdvancedOptions {config} fieldDefinitions={schemaFields} description="Fine-grained pooling and image-embedding settings." sections={['pooling', 'image_embedding']} excludePaths={['pooling.type', 'image_embedding.projection.type', 'image_embedding.projection.output_dim']} on:change={(event) => { config = event.detail.config; report = null; }} /></section>
		{#if currentTask === 'classification'}<section class="panel module-editor builder-section"><header><div><p class="eyebrow">05 · METADATA & FUSION</p><h2>Optional scalar context</h2><p>Choose an available ROI measurement or numeric dataset metadata field; paths never need to be typed.</p></div></header><div class="editor-fields"><ConfigField definition={field('metadata.encoder.type')} value={value(config, 'metadata.encoder.type')} {config} on:change={(event) => mutate('metadata.encoder.type', event.detail.value)} /><ConfigField definition={field('fusion.type')} value={value(config, 'fusion.type')} {config} on:change={(event) => mutate('fusion.type', event.detail.value)} /></div><div class="ordered-editor"><div class="panel-head"><div><h3>Metadata fields</h3><p>Drag to set stable feature order.</p></div><div class="metadata-actions"><button class="secondary small" on:click={addRoiContext}>Add ROI area + aspect ratio</button><button class="secondary small" on:click={addMetadata}>Add field</button></div></div>{#each fields() as metadataField, index}<article draggable="true" on:dragstart={() => dragIndex = index} on:dragover|preventDefault on:drop={() => reorderMetadata(index)}><span class="drag-handle" aria-label="Drag to reorder">⋮⋮</span><input value={String(metadataField.name ?? '')} aria-label="Metadata field name" on:change={(event) => updateMetadata(index, 'name', event.currentTarget.value)} /><label><span class="sr-only">Scalar source</span><select aria-label="Scalar source" value={String(metadataField.source ?? '')} on:change={(event) => updateMetadata(index, 'source', event.currentTarget.value)}>{#if !metadataSources.some((source) => source.value === String(metadataField.source ?? ''))}<option value={String(metadataField.source ?? '')}>Unavailable saved field: {String(metadataField.source ?? '')}</option>{/if}{#each metadataSources as source}<option value={source.value}>{source.label}</option>{/each}</select></label><label><span class="sr-only">Transform</span><select aria-label="Scalar transform" value={String(metadataField.transform === 'identity' ? 'none' : metadataField.transform ?? 'none')} on:change={(event) => updateMetadata(index, 'transform', event.currentTarget.value)}>{#each transforms as [transform, label]}<option value={transform}>{label}</option>{/each}</select></label><button class="icon-button" aria-label="Remove field" on:click={() => removeMetadata(index)}>×</button></article>{:else}<p class="empty">No metadata inputs. Add one only when it is available at inference.</p>{/each}</div><StageAdvancedOptions {config} fieldDefinitions={schemaFields} description="Metadata encoder internals and fusion." sections={['metadata', 'fusion']} excludePrefixes={['metadata.augmentation.']} excludePaths={['metadata.fields', 'metadata.encoder.type', 'fusion.type']} on:change={(event) => { config = event.detail.config; report = null; }} /></section>
		{/if}
		{#if currentTask === 'classification'}<section class="panel module-editor builder-section"><header><div><p class="eyebrow">06 · PRIMARY CLASSIFIER</p><h2>Prediction head</h2><p>Choose how the fused representation becomes a class prediction.</p></div></header><div class="editor-fields"><ConfigField definition={field('classifier.type')} value={value(config, 'classifier.type')} {config} on:change={(event) => mutate('classifier.type', event.detail.value)} />{#if value(config, 'classifier.type') === 'prototype'}<ConfigField definition={field('classifier.prototypes_per_class')} value={value(config, 'classifier.prototypes_per_class')} {config} on:change={(event) => mutate('classifier.prototypes_per_class', event.detail.value)} /><ConfigField definition={field('classifier.prototype_metric')} value={value(config, 'classifier.prototype_metric')} {config} on:change={(event) => mutate('classifier.prototype_metric', event.detail.value)} />{/if}</div><StageAdvancedOptions {config} fieldDefinitions={schemaFields} description="Head dimensions and classifier regularization." sections={['classifier']} excludePaths={['classifier.type', 'classifier.prototypes_per_class', 'classifier.prototype_metric']} on:change={(event) => { config = event.detail.config; report = null; }} /></section>{/if}
		{#if currentTask === 'classification'}<section class="panel module-editor builder-section"><header><div><p class="eyebrow">07 · POST-HOC PREDICTOR</p><h2>Optional downstream estimator</h2><p>Fits a classifier on training embeddings, selects candidates using validation macro F1, and reports held-out test metrics.</p></div></header><PosthocControls settings={record(value(config, 'posthoc', {}))} on:change={(event) => mutate('posthoc', event.detail.settings)} /></section>{/if}
		<section class="panel module-editor builder-section"><header><div><p class="eyebrow">08 · TRAINING & EXECUTION</p><h2>Training controls</h2><p>Definitions capture the reusable protocol. Set the epoch budget when constructing a queued run so it is sealed with that dataset and allocation.</p></div></header><TrainingControls {config} fieldDefinitions={schemaFields} on:change={(event) => { config = event.detail.config; report = null; }} /></section>
		{#if currentTask !== 'embedding'}<section class="panel module-editor builder-section"><header><div><p class="eyebrow">09 · LIVE MONITORING</p><h2>Supervised dashboard watchlist</h2><p>Seal optional evidence thresholds into the definition. They notify in the live dashboard; they never change or stop training.</p></div></header><div class="editor-fields"><ConfigField definition={field('monitoring.primary_metric')} value={value(config, 'monitoring.primary_metric')} {config} on:change={(event) => mutate('monitoring.primary_metric', event.detail.value)} /><ConfigField definition={field('monitoring.target_enabled')} value={value(config, 'monitoring.target_enabled')} {config} on:change={(event) => mutate('monitoring.target_enabled', event.detail.value)} />{#if Boolean(value(config, 'monitoring.target_enabled', false))}<ConfigField definition={field('monitoring.target_metric')} value={value(config, 'monitoring.target_metric')} {config} on:change={(event) => mutate('monitoring.target_metric', event.detail.value)} /><ConfigField definition={field('monitoring.target_value')} value={value(config, 'monitoring.target_value')} {config} on:change={(event) => mutate('monitoring.target_value', event.detail.value)} />{/if}<ConfigField definition={field('monitoring.max_validation_loss_increase_enabled')} value={value(config, 'monitoring.max_validation_loss_increase_enabled')} {config} on:change={(event) => mutate('monitoring.max_validation_loss_increase_enabled', event.detail.value)} />{#if Boolean(value(config, 'monitoring.max_validation_loss_increase_enabled', false))}<ConfigField definition={field('monitoring.max_validation_loss_increase')} value={value(config, 'monitoring.max_validation_loss_increase')} {config} on:change={(event) => mutate('monitoring.max_validation_loss_increase', event.detail.value)} />{/if}<ConfigField definition={field('monitoring.max_generalization_gap_enabled')} value={value(config, 'monitoring.max_generalization_gap_enabled')} {config} on:change={(event) => mutate('monitoring.max_generalization_gap_enabled', event.detail.value)} />{#if Boolean(value(config, 'monitoring.max_generalization_gap_enabled', false))}<ConfigField definition={field('monitoring.max_generalization_gap')} value={value(config, 'monitoring.max_generalization_gap')} {config} on:change={(event) => mutate('monitoring.max_generalization_gap', event.detail.value)} />{/if}</div></section>{/if}
		{#if report}<section class:invalid={report.valid === false} class="validation-message {report.valid === false ? 'invalid' : 'valid'}"><strong>{report.valid === false ? 'Draft needs attention' : 'Draft validated'}</strong><span>{String(report.message ?? (report.valid === false ? report.error ?? 'Review the reported configuration constraints.' : 'The configuration can be planned for training.'))}</span></section>{/if}
		<section class="construction-footer panel"><div><strong>{selected ? 'Definition revision is saved.' : 'Definition is local until you create it.'}</strong><small>Open Queue to pair this immutable revision with a frozen training set and validate it before scheduling.</small></div><div class="studio-actions"><button disabled={saving} on:click={save}>{saving ? 'Saving…' : selected ? 'Save revision' : 'Create definition'}</button><button disabled={!selected?.definition_id} on:click={() => selected?.definition_id && ontrain(String(selected.definition_id))}>Open queue workflow</button></div></section>
		{/if}
	</div>
</div>{/if}
