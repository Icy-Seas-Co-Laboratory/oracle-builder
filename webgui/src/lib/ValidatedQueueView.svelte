<script lang="ts">
	import { onMount } from 'svelte';
	import { api, type RecordValue } from '$lib/api';
	import TrainingStatusModal from '$lib/TrainingStatusModal.svelte';

	export let datasets: RecordValue[] = [];
	export let computeEndpoints: RecordValue[] = [];
	export let onchanged: (message: string) => void | Promise<void>;
	export let onfailure: (message: string) => void;

	const records = (value: unknown): RecordValue[] => Array.isArray(value) ? value as RecordValue[] : [];
	const record = (value: unknown): RecordValue => value && typeof value === 'object' && !Array.isArray(value) ? value as RecordValue : {};
	const number = (value: unknown): number | null => typeof value === 'number' && Number.isFinite(value) ? value : typeof value === 'string' && value.trim() !== '' && Number.isFinite(Number(value)) ? Number(value) : null;
	const text = (value: unknown) => value == null || value === '' ? '—' : String(value);
	const formatNumber = (value: unknown, digits = 4) => { const parsed = number(value); return parsed == null ? '—' : parsed.toLocaleString(undefined, { maximumFractionDigits: digits }); };
	const formatDuration = (value: unknown) => {
		const seconds = number(value); if (seconds == null || seconds < 0) return '—';
		const hours = Math.floor(seconds / 3600), minutes = Math.floor((seconds % 3600) / 60);
		return hours ? `${hours}h ${minutes}m` : minutes ? `${minutes}m` : '< 1m';
	};
	let definitions: RecordValue[] = [];
	let queued: RecordValue[] = [];
	let jobs: RecordValue[] = [];
	let liveStatusByJob: Record<string, RecordValue> = {};
	let definitionId = '';
	let datasetId = '';
	let endpointId = '';
	let runName = '';
	let gpuIds: string[] = [];
	let batchSizeMode: 'manual' | 'auto' = 'manual';
	let batchSize = 16;
	let maximumBatchSize = 256;
	let batchDefinitionId = '';
	let selected = new Set<string>();
	let busy = false;
	let controllingJobId = '';
	let liveJob: RecordValue | null = null;

	$: frozenDatasets = datasets.filter((item) => item.lifecycle === 'frozen');
	$: if (!definitionId && definitions.length) definitionId = String(definitions[0].definition_id);
	$: if (!datasetId && frozenDatasets.length) datasetId = String(frozenDatasets[0].dataset_id);
	$: if (!endpointId && computeEndpoints.length) endpointId = String(computeEndpoints.find((item) => item.status === 'ready')?.endpoint_id ?? computeEndpoints[0].endpoint_id);
	$: selectedDefinition = definitions.find((item) => String(item.definition_id) === definitionId);
	$: if (selectedDefinition && batchDefinitionId !== String(selectedDefinition.definition_id)) {
		const configured = Number(((selectedDefinition.config as RecordValue | undefined)?.data as RecordValue | undefined)?.batch_size ?? 16);
		batchSize = Number.isFinite(configured) && configured >= 1 ? configured : 16;
		batchDefinitionId = String(selectedDefinition.definition_id);
	}
	$: selectedDataset = frozenDatasets.find((item) => String(item.dataset_id) === datasetId);
	$: selectedEndpoint = computeEndpoints.find((item) => String(item.endpoint_id) === endpointId);
	$: endpointGpus = Array.from(new Map(
		records(selectedEndpoint?.workers).flatMap((worker) => records((worker.capabilities as RecordValue | undefined)?.gpus))
			.filter((gpu) => gpu.id !== undefined && gpu.id !== null)
			.map((gpu) => [String(gpu.id), gpu])
	).values());
	$: availableGpuIds = new Set(endpointGpus.map((gpu) => String(gpu.id)));
	$: if (gpuIds.some((id) => !availableGpuIds.has(id))) gpuIds = gpuIds.filter((id) => availableGpuIds.has(id));
	$: endpointReady = Boolean(selectedEndpoint && selectedEndpoint.status === 'ready');
	$: canQueue = Boolean(definitionId && datasetId && endpointId && runName.trim());
	$: queueRequirements = [
		{ label: 'Definition revision', detail: selectedDefinition ? `${text(selectedDefinition.name)} · revision ${text(selectedDefinition.revision)} pinned` : 'Choose a versioned definition', state: selectedDefinition ? 'ready' : 'blocked' },
		{ label: 'Frozen training set', detail: selectedDataset ? `${text(selectedDataset.name)} verified` : frozenDatasets.length ? 'Choose a frozen revision' : 'Freeze a training set first', state: selectedDataset ? 'ready' : 'blocked' },
		{ label: 'Compute endpoint', detail: selectedEndpoint ? `${text(selectedEndpoint.name)} · ${endpointReady ? 'reachable' : text(selectedEndpoint.status)}` : 'Choose an endpoint', state: endpointReady ? 'ready' : selectedEndpoint ? 'advisory' : 'blocked' },
		{ label: 'Allocation & telemetry', detail: endpointGpus.length ? (gpuIds.length ? `GPU ${gpuIds.join(', ')} selected` : 'CPU allocation selected') : selectedEndpoint ? 'CPU allocation; GPU telemetry unavailable' : 'Select an endpoint first', state: selectedEndpoint ? (endpointGpus.length ? 'ready' : 'advisory') : 'blocked' },
		{ label: 'Run name', detail: runName.trim() ? 'Name is ready to seal' : 'Enter a descriptive run name', state: runName.trim() ? 'ready' : 'blocked' },
	];
	$: missingRequirements = queueRequirements.filter((check) => check.state === 'blocked');
	$: clearableQueued = queued.filter((item) => ['ready', 'needs_attention', 'waiting_for_resources'].includes(String(item.status)) && (!endpointId || String(item.preflight_endpoint_id) === endpointId));
	const activeJob = (job: RecordValue | undefined) => ['dispatching', 'submitted', 'queued', 'running', 'paused', 'validating'].includes(String(job?.status));
	const jobFor = (item: RecordValue) => jobs.find((job) => String(job.queued_run_id) === String(item.queued_run_id));
	const gpuLabel = (gpu: RecordValue) => {
		const free = gpu.free_memory_mib, total = gpu.total_memory_mib;
		return free !== undefined && total !== undefined ? `GPU ${text(gpu.id)} · ${text(free)} / ${text(total)} MiB free` : `GPU ${text(gpu.id)}`;
	};
	const allocationLabel = (resources: RecordValue | undefined) => {
		const ids = Array.isArray(resources?.gpu_ids) ? resources!.gpu_ids.map(String) : null;
		return ids ? (ids.length ? `GPU ${ids.join(', ')}` : 'CPU') : `${text(resources?.gpu_count ?? 0)} GPU request`;
	};
	function normalizedLiveStatus(result: RecordValue): RecordValue {
		const outer = record(result.status ?? result.training_status ?? result);
		const snapshot = record(outer.snapshot);
		const progress = record(snapshot.progress);
		const timing = record(snapshot.timing);
		const metrics = record(snapshot.metrics);
		return {
			...outer, ...snapshot, progress,
			batch: progress.completed_batches ?? progress.batch ?? snapshot.batch,
			total_batches: progress.total_batches ?? snapshot.total_batches,
			epoch: progress.epoch ?? snapshot.epoch,
			total_epochs: progress.total_epochs ?? snapshot.total_epochs,
			total_eta_seconds: timing.total_eta_seconds ?? snapshot.total_eta_seconds,
			current_metrics: record(metrics.current_batch ?? snapshot.current_batch_metrics),
			completed_metrics: record(metrics.last_completed_epoch ?? snapshot.latest_metrics)
		};
	}
	function queueProgress(live: RecordValue | undefined): number | null {
		const progress = record(live?.progress);
		const direct = number(live?.percent ?? progress.percent ?? progress.progress);
		if (direct != null) return Math.max(0, Math.min(100, direct));
		const batch = number(live?.batch ?? progress.completed_batches ?? progress.batch);
		const total = number(live?.total_batches ?? progress.total_batches ?? progress.batches_per_epoch);
		return batch != null && total ? Math.max(0, Math.min(100, batch / total * 100)) : null;
	}
	function liveMetric(live: RecordValue | undefined, key: string) {
		const current = record(live?.current_metrics);
		const completed = record(live?.completed_metrics);
		return current[key] ?? completed[key];
	}
	const supportsControl = (live: RecordValue | undefined, action: string) => Array.isArray(live?.controls) && live.controls.map(String).includes(action);
	function toggleGpu(id: string) {
		gpuIds = gpuIds.includes(id) ? [] : [id];
	}

	async function load() {
		try {
			const [definitionResult, queueResult, jobResult] = await Promise.all([api.modelDefinitions(), api.queuedRuns(), api.jobs(true)]);
			definitions = records(definitionResult.definitions); queued = records(queueResult.queued_runs); jobs = records(jobResult.jobs);
			const active = jobs.filter(activeJob);
			const snapshots = await Promise.all(active.map(async (job) => {
				const id = String(job.job_id ?? job.id ?? '');
				if (!id) return [id, {}] as const;
				try { return [id, normalizedLiveStatus(await api.jobTrainingStatus(id))] as const; }
				catch { return [id, {}] as const; }
			}));
			liveStatusByJob = Object.fromEntries(snapshots);
		} catch (error) { onfailure(error instanceof Error ? error.message : 'Could not load definitions and queue.'); }
	}
	function toggle(id: string) {
		const next = new Set(selected); next.has(id) ? next.delete(id) : next.add(id); selected = next;
	}
	async function validateAndQueue() {
		if (!definitionId || !datasetId || !endpointId || !runName.trim()) return;
		busy = true;
		try {
			const row = await api.validateAndQueueDefinition(definitionId, {
				name: runName.trim(), dataset_id: datasetId, endpoint_id: endpointId,
				revision: Number(selectedDefinition?.revision), resources: { gpu_count: gpuIds.length, gpu_ids: gpuIds },
				batch_size_mode: batchSizeMode,
				batch_size: batchSizeMode === 'manual' ? batchSize : undefined,
				maximum_batch_size: batchSizeMode === 'auto' ? maximumBatchSize : undefined,
			});
			await load(); runName = '';
			await onchanged(row.status === 'ready' ? 'Validated and added the immutable run to the queue.' : 'Queue entry was saved, but preflight needs attention.');
		} catch (error) { onfailure(error instanceof Error ? error.message : 'Could not validate and queue this run.'); }
		finally { busy = false; }
	}
	async function start(allReady = false) {
		if (!endpointId || (!allReady && !selected.size)) return;
		busy = true;
		try {
			const result = await api.startQueuedRuns({ endpoint_id: endpointId, queued_run_ids: [...selected], all_ready: allReady });
			await load(); selected = new Set();
			await onchanged(Number(records(result.dispatched).length) ? 'Scheduler started the next fitting run.' : 'Selected runs are authorized and will wait for available compute.');
		} catch (error) { onfailure(error instanceof Error ? error.message : 'Could not start the selected queue entries.'); }
		finally { busy = false; }
	}
	async function cancel(id: string) {
		try { await api.cancelQueuedRun(id); await load(); await onchanged('Removed the queued run.'); }
		catch (error) { onfailure(error instanceof Error ? error.message : 'Could not cancel this queued run.'); }
	}
	async function controlJob(job: RecordValue, action: 'pause' | 'resume' | 'cancel') {
		const id = String(job.job_id ?? job.id ?? '');
		if (!id || controllingJobId) return;
		if (action === 'cancel' && !window.confirm('Cancel this active training run? The process will stop and the run cannot be resumed from the queue.')) return;
		controllingJobId = id;
		try {
			if (action === 'pause') await api.pauseJob(id);
			else if (action === 'resume') await api.resumeJob(id);
			else await api.cancelJob(id);
			await load();
			await onchanged(action === 'pause' ? 'Training run paused. Its process and in-memory state are preserved.' : action === 'resume' ? 'Training run resumed.' : 'Cancellation requested. The worker will stop the active process.');
		} catch (error) { onfailure(error instanceof Error ? error.message : `Could not ${action} the active run.`); }
		finally { controllingJobId = ''; }
	}
	async function resetStuck() {
		if (!endpointId) return;
		busy = true;
		try {
			const report = await api.resetStuckJobs(endpointId);
			await load();
			const reset = records(report.reset).length;
			await onchanged(reset ? `${reset} stale dispatch ${reset === 1 ? 'was' : 'were'} reset. Re-authorize a recovered run to dispatch it.` : 'No missing compute jobs were found. Active runs were left untouched.');
		} catch (error) { onfailure(error instanceof Error ? error.message : 'Could not reconcile stuck jobs.'); }
		finally { busy = false; }
	}
	async function clearQueue() {
		if (!endpointId || !clearableQueued.length || !window.confirm(`Clear ${clearableQueued.length} non-running queue ${clearableQueued.length === 1 ? 'entry' : 'entries'} for this endpoint? Submitted and running compute jobs are never cleared.`)) return;
		busy = true;
		try {
			const report = await api.clearQueuedRuns(endpointId);
			await load();
			const cleared = records(report.cleared).length;
			await onchanged(`${cleared} non-running queue ${cleared === 1 ? 'entry was' : 'entries were'} cleared. Active compute jobs were left untouched.`);
		} catch (error) { onfailure(error instanceof Error ? error.message : 'Could not clear the queue.'); }
		finally { busy = false; }
	}
	onMount(() => {
		void load();
		const timer = window.setInterval(() => void load(), 5_000);
		return () => window.clearInterval(timer);
	});
</script>

<section class="studio-intro"><div><p class="eyebrow">VALIDATED QUEUE</p><h1>Queue training deliberately.</h1><p>Validation seals a model-definition revision and frozen dataset into an immutable TOML. Nothing starts until you explicitly authorize it below.</p></div></section>

{#if !definitions.length || !frozenDatasets.length}
	<section class="workflow-guidance panel" aria-label="Next training step">
		<div><p class="eyebrow">NEXT STEP</p><h2>{!definitions.length ? 'Create a model definition' : 'Freeze a training set'}</h2><p>{!definitions.length ? 'A maintained template becomes a versioned definition in Construction. Return here to pair it with frozen data.' : 'Only frozen training-set revisions can be sealed into a reproducible run.'}</p></div>
		<span class="workflow-step">{!definitions.length ? '1 · Definition' : '2 · Frozen data'}</span>
	</section>
{/if}

<section class="panel training-form">
	<div class="panel-head"><div><h2>Validate and queue a run</h2><p>Definitions carry the scientific and training settings. This step only selects frozen data and compute.</p></div></div>
	<div class="editor-fields">
		<label>Model definition<select bind:value={definitionId}>{#each definitions as definition}<option value={String(definition.definition_id)}>{text(definition.name)} · revision {text(definition.revision)}</option>{/each}</select></label>
		<label>Frozen training set<select bind:value={datasetId}>{#each frozenDatasets as dataset}<option value={String(dataset.dataset_id)}>{text(dataset.name)} · {text(dataset.fingerprint_sha256).slice(0, 10)}</option>{/each}</select></label>
		<label>Compute endpoint<select bind:value={endpointId}>{#each computeEndpoints as endpoint}<option value={String(endpoint.endpoint_id)}>{text(endpoint.name)} · {text(endpoint.status)}</option>{/each}</select></label>
		<div class="gpu-allocation"><span>GPU allocation</span>{#if endpointGpus.length}<div class="gpu-chips"><button type="button" class:active={!gpuIds.length} on:click={() => gpuIds = []}>CPU</button>{#each endpointGpus as gpu}<button type="button" class:active={gpuIds.includes(String(gpu.id))} aria-pressed={gpuIds.includes(String(gpu.id))} on:click={() => toggleGpu(String(gpu.id))}>{gpuLabel(gpu)}</button>{/each}</div><small>Select one exact GPU. This allocation and its runtime device policy are sealed with the queued run.</small>{:else}<small>No GPU inventory is available for this endpoint. Queueing will use an explicit CPU allocation until GPU telemetry is visible.</small>{/if}</div>
		<div class="batch-allocation"><span>Training batch size</span><div class="choice-tabs"><button type="button" class:active={batchSizeMode === 'manual'} on:click={() => batchSizeMode = 'manual'}>Manual</button><button type="button" class:active={batchSizeMode === 'auto'} on:click={() => batchSizeMode = 'auto'}>Auto-tune</button></div>{#if batchSizeMode === 'manual'}<label>Batch size<input type="number" min="1" bind:value={batchSize} /></label><small>Sealed for this run only; it does not change the model definition.</small>{:else}<label>Maximum to test<input type="number" min="1" bind:value={maximumBatchSize} /></label><small>Before training begins, Auto-tune runs isolated forward/backward probes on this selected allocation up to this limit, keeps a safety margin, then seals and displays the chosen batch size with the run.</small>{/if}</div>
	</div>
	<label>Queued run name<input bind:value={runName} placeholder="e.g. exp001-resnet-128 · frozen-v4" /></label>
	<div class="queue-preflight" aria-live="polite"><div class="preflight-heading"><div><p class="eyebrow">LAUNCH READINESS</p><h3>{missingRequirements.length ? `${missingRequirements.length} item${missingRequirements.length === 1 ? '' : 's'} needed before validation` : endpointReady ? 'Ready to validate and queue' : 'Ready to validate; endpoint needs attention'}</h3></div><span class:ready={canQueue} class:advisory={!canQueue && Boolean(selectedEndpoint)} class="preflight-state">{canQueue ? 'FORM COMPLETE' : 'INCOMPLETE'}</span></div><div class="preflight-checks">{#each queueRequirements as check}<div class:ready={check.state === 'ready'} class:advisory={check.state === 'advisory'} class:blocked={check.state === 'blocked'} class="preflight-check"><span aria-hidden="true">{check.state === 'ready' ? '✓' : check.state === 'advisory' ? '!' : '○'}</span><div><strong>{check.label}</strong><small>{check.detail}</small></div></div>{/each}</div></div>
	<div class="run-composer-footer"><span aria-live="polite">{busy && batchSizeMode === 'auto' ? 'Calibrating a safe batch size on the selected compute allocation. This can take several minutes.' : missingRequirements.length ? `To enable validation: ${missingRequirements.map((check) => check.label.toLowerCase()).join(', ')}.` : selectedDefinition ? `${text(selectedDefinition.name)} revision ${text(selectedDefinition.revision)} will be pinned.` : 'Choose a definition.'}</span><button disabled={busy || !canQueue} on:click={validateAndQueue}>{busy ? batchSizeMode === 'auto' ? 'Calibrating batch size…' : 'Validating…' : 'Validate & Queue Run'}</button></div>
</section>

<section class="queue-recovery panel"><div><p class="eyebrow">QUEUE RECOVERY</p><h2>Resolve stale state safely</h2><p>Reconcile marks a job stale only when this endpoint confirms its remote record is gone. Clear cancels local, non-running entries only.</p></div><div class="row-actions"><button class="secondary small" disabled={busy || !endpointId} on:click={resetStuck}>Reconcile stuck jobs</button><button class="quiet-button small" disabled={busy || !clearableQueued.length} on:click={clearQueue}>Clear non-running queue{clearableQueued.length ? ` (${clearableQueued.length})` : ''}</button></div></section>

<section class="panel"><div class="panel-head"><div><h2>Run queue</h2><p>Select ready entries validated for the selected endpoint, then authorize the scheduler. Active runs publish progress and their latest training signal here.</p></div><div class="row-actions"><button class="secondary small" disabled={busy || !selected.size} on:click={() => start(false)}>Start selected</button><button class="small" disabled={busy || !queued.some((item) => item.status === 'ready' && item.preflight_endpoint_id === endpointId)} on:click={() => start(true)}>Start all ready here</button></div></div>
	{#if queued.length}<div class="queue-table-wrap"><table><thead><tr><th></th><th>Run</th><th>Definition</th><th>Dataset</th><th>Preflight</th><th>Status</th><th>Progress</th><th>Live signal</th><th></th></tr></thead><tbody>{#each queued as item}{@const job = jobFor(item)}{@const live = liveStatusByJob[String(job?.job_id ?? job?.id ?? '')]}{@const progress = queueProgress(live)}<tr class:attention={item.status === 'needs_attention'}><td><input type="checkbox" disabled={item.status !== 'ready' || item.preflight_endpoint_id !== endpointId} checked={selected.has(String(item.queued_run_id))} aria-label={`Select ${text(item.name)}`} on:change={() => toggle(String(item.queued_run_id))} /></td><td><strong>{text(item.name)}</strong><small>Allocation: {allocationLabel(item.resources as RecordValue | undefined)} · Batch: {text(item.batch_size)}</small></td><td>{text(item.definition_name)}<small>revision {text(item.definition_revision)}</small></td><td>{text(item.dataset_id)}<small>{text(item.dataset_fingerprint_sha256).slice(0, 10)}</small></td><td><span class:valid={item.preflight_status === 'valid'} class="status">{item.preflight_status === 'valid' ? 'Validated' : 'Needs attention'}</span>{#if item.failure_reason}<small>{text(item.failure_reason)}</small>{/if}</td><td><span class="status {text(item.status)}">{text(item.status).replaceAll('_', ' ')}</span>{#if item.start_authorized}<small>Start authorized</small>{/if}</td><td class="queue-progress">{#if activeJob(job) && progress != null}<div class="queue-progress-label"><strong>{Math.round(progress)}%</strong><span>epoch {text(live?.epoch)} / {text(live?.total_epochs)}</span></div><div class="queue-progress-track" aria-label={`${Math.round(progress)}% through the current epoch`}><i style={`width:${progress}%`}></i></div><small>Batch {text(live?.batch)} / {text(live?.total_batches)} · ETA {formatDuration(live?.total_eta_seconds)}</small>{:else if activeJob(job)}<small>Awaiting worker telemetry…</small>{:else}<small>—</small>{/if}</td><td class="queue-live-signal">{#if activeJob(job) && live && Object.keys(live).length}<div><span>Loss <strong>{formatNumber(liveMetric(live, 'loss'))}</strong></span><span>Accuracy <strong>{formatNumber(liveMetric(live, 'accuracy'))}</strong></span><span>Macro F1 <strong>{formatNumber(liveMetric(live, 'macro_f1') ?? liveMetric(live, 'val_macro_f1'))}</strong></span></div>{#if liveMetric(live, 'val_macro_f1') != null}<small>Validated F1 {formatNumber(liveMetric(live, 'val_macro_f1'))}</small>{/if}{:else}<small>—</small>{/if}</td><td><div class="job-controls">{#if job && activeJob(job)}{#if supportsControl(live, 'resume')}<button class="secondary small" disabled={Boolean(controllingJobId)} on:click={() => controlJob(job, 'resume')}>{controllingJobId === String(job.job_id) ? 'Working…' : 'Resume'}</button>{:else if supportsControl(live, 'pause')}<button class="secondary small" disabled={Boolean(controllingJobId)} on:click={() => controlJob(job, 'pause')}>{controllingJobId === String(job.job_id) ? 'Working…' : 'Pause'}</button>{/if}{#if supportsControl(live, 'cancel')}<button class="quiet-button small job-cancel" disabled={Boolean(controllingJobId)} on:click={() => controlJob(job, 'cancel')}>Cancel</button>{/if}<button class="secondary small" on:click={() => liveJob = job}>Live dashboard</button>{:else if ['ready', 'needs_attention', 'waiting_for_resources'].includes(String(item.status))}<button class="secondary small" on:click={() => cancel(String(item.queued_run_id))}>Cancel</button>{/if}</div></td></tr>{/each}</tbody></table></div>{:else}<p class="empty">No validated runs yet. Select a model definition and frozen dataset above.</p>{/if}
</section>

{#if liveJob}<TrainingStatusModal job={liveJob} runName={text(queued.find((item) => String(item.queued_run_id) === String(liveJob?.queued_run_id))?.name ?? liveJob.job_id)} onclose={() => liveJob = null} />{/if}

<style>
	.queue-table-wrap { overflow-x: auto; margin: 0 -0.2rem; }
	.queue-table-wrap table { min-width: 1120px; }
	.queue-progress { min-width: 10.8rem; }
	.queue-progress-label { display: flex; justify-content: space-between; gap: .5rem; font-size: .72rem; }
	.queue-progress-label span { color: var(--muted); }
	.queue-progress-track { height: .38rem; overflow: hidden; margin: .32rem 0; border-radius: 999px; background: #dbeafe; }
	.queue-progress-track i { display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg, #0f766e, #2dd4bf); }
	.queue-progress small, .queue-live-signal small { display: block; color: var(--muted); font-size: .67rem; white-space: nowrap; }
	.queue-live-signal { min-width: 13rem; }
	.queue-live-signal > div { display: flex; flex-wrap: wrap; gap: .3rem .55rem; }
	.queue-live-signal span { color: var(--muted); font-size: .68rem; white-space: nowrap; }
	.queue-live-signal strong { color: var(--ink); }
	.job-controls { display: flex; flex-wrap: wrap; gap: .3rem; min-width: 9rem; }
	.job-cancel { color: #b91c1c; border-color: #fecaca; }
</style>
