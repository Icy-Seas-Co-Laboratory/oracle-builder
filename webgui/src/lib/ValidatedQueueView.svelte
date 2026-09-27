<script lang="ts">
	import { onMount } from 'svelte';
	import { api, ApiRequestError, type RecordValue } from '$lib/api';
	import { subscribeOperationalRefresh } from '$lib/operational-refresh';
	import TrainingStatusModal from '$lib/TrainingStatusModal.svelte';

	export let datasets: RecordValue[] = [];
	export let computeEndpoints: RecordValue[] = [];
	export let onchanged: (message: string) => void | Promise<void>;
	export let onfailure: (message: string) => void;

	const records = (value: unknown): RecordValue[] => Array.isArray(value) ? value as RecordValue[] : [];
	const record = (value: unknown): RecordValue => value && typeof value === 'object' && !Array.isArray(value) ? value as RecordValue : {};
	const strings = (value: unknown): string[] => Array.isArray(value) ? value.map(String) : [];
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
	let epochs = 10;
	let batchDefinitionId = '';
	let selected = new Set<string>();
	let busy = false;
	let controllingJobId = '';
	let liveJob: RecordValue | null = null;
	let logJob: RecordValue | null = null;
	let logEvents: RecordValue[] = [];
	let logsLoading = false;
	let clearingQueuedRunId = '';
	let batchOverrideQueuedRunId = '';
	let batchOverrideValue = 1;
	let loadController: AbortController | undefined;
	let loadInFlight: Promise<void> | undefined;
	let validationOperationId = '';
	let validationOperation: RecordValue | null = null;
	let validationFeedback = '';
	let refreshingValidationOperation = false;

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
	// Serve's raw resources and the orchestrator's derived capacity each carry
	// useful fields.  Merge them so a raw resources object cannot hide live
	// worker-slot telemetry from the Queue screen.
	$: endpointResources = { ...record(selectedEndpoint?.resources ?? selectedEndpoint?.scheduler_resources), ...record(selectedEndpoint?.capacity) };
	$: endpointQueue = record(selectedEndpoint?.queue);
	$: endpointWorkerSlots = number(endpointResources.worker_slots ?? endpointResources.slots ?? endpointResources.total_slots ?? selectedEndpoint?.worker_slots);
	$: endpointActiveSlots = number(endpointResources.active_slots ?? endpointResources.slots_in_use ?? endpointResources.running_slots ?? selectedEndpoint?.active_slots);
	$: endpointFreeSlots = number(endpointResources.free_slots ?? endpointResources.available_slots ?? selectedEndpoint?.free_slots)
		?? (endpointWorkerSlots != null && endpointActiveSlots != null ? Math.max(0, endpointWorkerSlots - endpointActiveSlots) : null);
	$: endpointCpuCapacity = number(endpointResources.cpu_capacity ?? endpointResources.total_cpu ?? selectedEndpoint?.cpu_capacity);
	$: endpointCpuInUse = number(endpointResources.cpu_in_use ?? endpointResources.used_cpu ?? selectedEndpoint?.cpu_in_use);
	$: endpointGpuLeases = strings(endpointResources.gpu_leases ?? selectedEndpoint?.gpu_leases);
	$: endpointSchedulingState = String(selectedEndpoint?.scheduling_status ?? selectedEndpoint?.scheduler_status ?? (selectedEndpoint?.scheduling_enabled === false ? 'paused' : 'automatic'));
	$: endpointGpus = Array.from(new Map(
		records(selectedEndpoint?.workers).flatMap((worker) => records((worker.capabilities as RecordValue | undefined)?.gpus))
			.filter((gpu) => gpu.id !== undefined && gpu.id !== null)
			.map((gpu) => [String(gpu.id), gpu])
	).values());
	$: availableGpuIds = new Set(endpointGpus.map((gpu) => String(gpu.id)));
	$: if (gpuIds.some((id) => !availableGpuIds.has(id))) gpuIds = gpuIds.filter((id) => availableGpuIds.has(id));
	$: endpointReady = Boolean(selectedEndpoint && selectedEndpoint.status === 'ready');
	$: validEpochs = Number.isInteger(epochs) && epochs >= 1;
	$: canQueue = Boolean(definitionId && datasetId && endpointId && runName.trim() && validEpochs);
	$: validationInProgress = ['queued', 'running', 'validating'].includes(String(validationOperation?.status));
	$: queueRequirements = [
		{ label: 'Definition revision', detail: selectedDefinition ? `${text(selectedDefinition.name)} · revision ${text(selectedDefinition.revision)} pinned` : 'Choose a versioned definition', state: selectedDefinition ? 'ready' : 'blocked' },
		{ label: 'Frozen training set', detail: selectedDataset ? `${text(selectedDataset.name)} verified` : frozenDatasets.length ? 'Choose a frozen revision' : 'Freeze a training set first', state: selectedDataset ? 'ready' : 'blocked' },
		{ label: 'Compute endpoint', detail: selectedEndpoint ? `${text(selectedEndpoint.name)} · ${endpointReady ? 'reachable' : text(selectedEndpoint.status)}` : 'Choose an endpoint', state: endpointReady ? 'ready' : selectedEndpoint ? 'advisory' : 'blocked' },
		{ label: 'Allocation & telemetry', detail: endpointGpus.length ? (gpuIds.length ? `GPU ${gpuIds.join(', ')} selected` : 'CPU allocation selected') : selectedEndpoint ? 'CPU allocation; GPU telemetry unavailable' : 'Select an endpoint first', state: selectedEndpoint ? (endpointGpus.length ? 'ready' : 'advisory') : 'blocked' },
		{ label: 'Training epochs', detail: validEpochs ? `${epochs} epoch${epochs === 1 ? '' : 's'} will be sealed with this run` : 'Enter a positive whole number', state: validEpochs ? 'ready' : 'blocked' },
		{ label: 'Run name', detail: runName.trim() ? 'Name is ready to seal' : 'Enter a descriptive run name', state: runName.trim() ? 'ready' : 'blocked' },
	];
	$: missingRequirements = queueRequirements.filter((check) => check.state === 'blocked');
	$: clearableQueued = queued.filter((item) => ['ready', 'needs_attention', 'waiting_for_resources'].includes(String(item.status)) && (!endpointId || String(item.preflight_endpoint_id) === endpointId));
	const activeJob = (job: RecordValue | undefined) => ['dispatching', 'submitted', 'queued', 'running', 'paused', 'validating'].includes(String(job?.status));
	const terminalJob = (job: RecordValue | undefined) => ['failed', 'cancelled', 'indexed', 'artifact_invalid', 'dispatch_failed', 'succeeded'].includes(String(job?.status));
	const clearableTerminal = (item: RecordValue, job: RecordValue | undefined) => terminalJob(job) || (!job && ['failed', 'cancelled', 'complete'].includes(String(item.status)));
	const jobFor = (item: RecordValue) => jobs.find((job) => String(job.queued_run_id) === String(item.queued_run_id));
	const gpuLabel = (gpu: RecordValue) => {
		const free = gpu.free_memory_mib, total = gpu.total_memory_mib;
		return free !== undefined && total !== undefined ? `GPU ${text(gpu.id)} · ${text(free)} / ${text(total)} MiB free` : `GPU ${text(gpu.id)}`;
	};
	const allocationLabel = (resources: RecordValue | undefined) => {
		const ids = Array.isArray(resources?.gpu_ids) ? resources!.gpu_ids.map(String) : null;
		return ids ? (ids.length ? `GPU ${ids.join(', ')}` : 'CPU') : `${text(resources?.gpu_count ?? 0)} GPU request`;
	};
	const dispatchDetail = (item: RecordValue, job: RecordValue | undefined) => {
		const status = String(item.status ?? '');
		const explicit = item.resource_wait_reason ?? item.wait_reason ?? item.dispatch_reason ?? item.failure_reason;
		if (status === 'waiting_for_resources') return String(explicit || 'Authorized; waiting for compatible compute resources.');
		if (status === 'dispatching' || item.dispatch_claimed_at || item.dispatch_claim_owner) return String(explicit || 'Scheduler has claimed this run and is checking capacity.');
		if (item.start_authorized && !job) return String(explicit || 'Authorized for automatic scheduling.');
		if (status === 'ready') return 'Awaiting authorization.';
		return explicit ? String(explicit) : '';
	};
	const statusLabel = (item: RecordValue, job: RecordValue | undefined) => {
		if (String(item.status) === 'waiting_for_resources') return 'Waiting for resources';
		if (String(item.status) === 'dispatching' || item.dispatch_claimed_at) return 'Dispatching';
		if (item.start_authorized && !job && String(item.status) === 'ready') return 'Authorized';
		return text(item.status).replaceAll('_', ' ');
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
	const workerUnavailable = (live: RecordValue | undefined) => Boolean(live) && (live?.available === false || live?.stale === true);
	const workerUnavailableDetail = (live: RecordValue | undefined) => String(live?.message || 'The compute worker is not publishing a live status for this run.');
	const lastKnownUpdate = (live: RecordValue | undefined) => text(live?.updated_at ?? live?.last_updated_at);
	function liveMetric(live: RecordValue | undefined, key: string) {
		const current = record(live?.current_metrics);
		const completed = record(live?.completed_metrics);
		return current[key] ?? completed[key];
	}
	const supportsControl = (live: RecordValue | undefined, action: string) => Array.isArray(live?.controls) && live.controls.map(String).includes(action);
	const canOverrideBatch = (item: RecordValue) => !item.start_authorized && ['ready', 'needs_attention', 'waiting_for_resources'].includes(String(item.status));
	const batchOverrideDetail = (item: RecordValue) => {
		const execution = record(item.batch_execution);
		if (execution.mode !== 'manual_override') return '';
		const tuned = number(execution.auto_tuned_batch_size);
		return tuned != null ? `Auto-tuned ${tuned}; manually set to ${text(item.batch_size)}.` : 'Manual queue-level batch override.';
	};
	function toggleGpu(id: string) {
		gpuIds = gpuIds.includes(id) ? [] : [id];
	}

	function load(supersede = false): Promise<void> {
		if (loadInFlight && !supersede) return loadInFlight;
		if (supersede) loadController?.abort();
		loadController = new AbortController();
		const signal = loadController.signal;
		const task = (async () => {
		try {
			// The workspace coordinator owns remote reconciliation. Queue reads its
			// cached job state so one screen cannot multiply control-plane work.
			const [definitionResult, queueResult, jobResult] = await Promise.all([api.modelDefinitions({ signal }), api.queuedRuns({ signal }), api.jobs(false, { signal })]);
			definitions = records(definitionResult.definitions); queued = records(queueResult.queued_runs); jobs = records(jobResult.jobs);
			const active = jobs.filter(activeJob);
			const snapshots = await Promise.all(active.map(async (job) => {
				const id = String(job.job_id ?? job.id ?? '');
				if (!id) return [id, {}] as const;
				try { return [id, normalizedLiveStatus(await api.jobTrainingStatus(id))] as const; }
				catch { return [id, {}] as const; }
			}));
			liveStatusByJob = Object.fromEntries(snapshots);
		} catch (error) {
			if (!(error instanceof DOMException && error.name === 'AbortError')) onfailure(error instanceof Error ? error.message : 'Could not load definitions and queue.');
		} finally {
			if (loadController?.signal === signal) loadInFlight = undefined;
		}
		})();
		loadInFlight = task;
		return task;
	}
	function toggle(id: string) {
		const next = new Set(selected); next.has(id) ? next.delete(id) : next.add(id); selected = next;
	}
	async function refreshValidationOperation() {
		if (!validationOperationId || refreshingValidationOperation) return;
		refreshingValidationOperation = true;
		try {
			const result = await api.operation(validationOperationId);
			const operation = record(result.operation ?? result);
			validationOperation = operation;
			const status = String(operation.status);
			const latestEvent = record(operation.latest_event);
			const stage = text(latestEvent.message);
			if (status === 'queued') validationFeedback = stage || 'Validation is queued. The scheduler will begin as soon as capacity is available.';
			else if (status === 'running' || status === 'validating') validationFeedback = stage || 'Validating the immutable definition, frozen dataset, and selected compute allocation…';
			else if (status === 'completed') {
				validationFeedback = 'Validation completed and the immutable run is ready in the queue.';
				validationOperationId = '';
				await load(true);
				await onchanged('Validated and added the immutable run to the queue.');
			} else if (status === 'failed') {
				validationFeedback = `Validation needs attention: ${text(operation.error)}`;
				validationOperationId = '';
				onfailure(validationFeedback);
			}
		} catch (error) {
			// The durable operation remains queryable after a transient refresh failure.
			if (!(error instanceof ApiRequestError && error.status === 404)) onfailure(error instanceof Error ? error.message : 'Could not refresh validation status.');
		} finally { refreshingValidationOperation = false; }
	}
	async function validateAndQueue() {
		if (!definitionId || !datasetId || !endpointId || !runName.trim() || validationInProgress) return;
		busy = true;
		const request = {
			name: runName.trim(), dataset_id: datasetId, endpoint_id: endpointId,
			revision: Number(selectedDefinition?.revision), resources: { gpu_count: gpuIds.length, gpu_ids: gpuIds },
			batch_size_mode: batchSizeMode,
			batch_size: batchSizeMode === 'manual' ? batchSize : undefined,
			maximum_batch_size: batchSizeMode === 'auto' ? maximumBatchSize : undefined,
			epochs,
		};
		try {
			try {
				const scheduled = await api.validateAndQueueDefinitionSchedule(definitionId, request);
				const operation = record(scheduled.operation ?? scheduled);
				const operationId = String(operation.operation_id ?? '');
				if (!operationId) throw new Error('The scheduler accepted validation without returning an operation ID.');
				validationOperationId = operationId;
				validationOperation = operation;
				validationFeedback = 'Validation accepted and queued. Progress will update here as the scheduler works.';
				runName = '';
				await refreshValidationOperation();
				return;
			} catch (error) {
				if (!(error instanceof ApiRequestError && error.status === 404)) throw error;
			}
			const row = await api.validateAndQueueDefinition(definitionId, request);
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
			await onchanged(Number(records(result.dispatched).length) ? 'Runs were authorized; automatic scheduling has started compatible work.' : 'Runs are authorized. The scheduler will start each one automatically when compatible compute is free.');
		} catch (error) { onfailure(error instanceof Error ? error.message : 'Could not start the selected queue entries.'); }
		finally { busy = false; }
	}
	async function cancel(id: string) {
		try { await api.cancelQueuedRun(id); await load(); await onchanged('Removed the queued run.'); }
		catch (error) { onfailure(error instanceof Error ? error.message : 'Could not cancel this queued run.'); }
	}
	function editBatchSize(item: RecordValue) {
		batchOverrideQueuedRunId = String(item.queued_run_id);
		batchOverrideValue = number(item.batch_size) ?? 1;
	}
	async function saveBatchSize(item: RecordValue) {
		const id = String(item.queued_run_id ?? '');
		if (!id || !Number.isInteger(batchOverrideValue) || batchOverrideValue < 1) return;
		busy = true;
		try {
			await api.updateQueuedRunBatchSize(id, batchOverrideValue);
			batchOverrideQueuedRunId = '';
			await load();
			await onchanged(`Batch size set to ${batchOverrideValue} for this queued run. It remains unstarted and will be rechecked at launch.`);
		} catch (error) { onfailure(error instanceof Error ? error.message : 'Could not update the queued batch size.'); }
		finally { busy = false; }
	}
	async function showLogs(job: RecordValue) {
		const id = String(job.job_id ?? job.id ?? '');
		if (!id) return;
		logJob = job; logEvents = []; logsLoading = true;
		try {
			const result = await api.jobEvents(id);
			logEvents = records(result.events);
		} catch (error) {
			onfailure(error instanceof Error ? error.message : 'Could not load run logs.');
		} finally { logsLoading = false; }
	}
	async function clearTerminal(item: RecordValue, job: RecordValue | undefined) {
		const id = String(item.queued_run_id ?? '');
		if (!id || clearingQueuedRunId) return;
		if (!window.confirm('Clear this finished run from the active queue? Its configuration, job record, and logs will be retained for diagnosis.')) return;
		clearingQueuedRunId = id;
		try {
			await api.clearTerminalQueuedRun(id);
			if (logJob && String(logJob.job_id ?? logJob.id) === String(job?.job_id ?? job?.id)) { logJob = null; logEvents = []; }
			await load();
			await onchanged('Finished run cleared from the active queue. Its logs remain available in the run audit.');
		} catch (error) { onfailure(error instanceof Error ? error.message : 'Could not clear the finished queue entry.'); }
		finally { clearingQueuedRunId = ''; }
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
		const unsubscribe = subscribeOperationalRefresh(() => { void load(); void refreshValidationOperation(); });
		return () => { unsubscribe(); loadController?.abort(); };
	});
</script>

<section class="studio-intro"><div><p class="eyebrow">VALIDATED QUEUE</p><h1>Queue training deliberately.</h1><p>Validation seals a model-definition revision and frozen dataset into an immutable TOML. Authorize ready runs below; the scheduler starts each compatible run automatically as capacity becomes free.</p></div></section>

{#if !definitions.length || !frozenDatasets.length}
	<section class="workflow-guidance panel" aria-label="Next training step">
		<div><p class="eyebrow">NEXT STEP</p><h2>{!definitions.length ? 'Create a model definition' : 'Freeze a training set'}</h2><p>{!definitions.length ? 'A maintained template becomes a versioned definition in Construction. Return here to pair it with frozen data.' : 'Only frozen training-set revisions can be sealed into a reproducible run.'}</p></div>
		<span class="workflow-step">{!definitions.length ? '1 · Definition' : '2 · Frozen data'}</span>
	</section>
{/if}

	<section class="panel training-form">
	<div class="panel-head"><div><h2>Validate and queue a run</h2><p>Definitions carry reusable architecture and training protocol. This step sets the frozen data, compute, and run-specific epoch budget.</p></div></div>
	<div class="editor-fields">
		<label>Model definition<select bind:value={definitionId}>{#each definitions as definition}<option value={String(definition.definition_id)}>{text(definition.name)} · revision {text(definition.revision)}</option>{/each}</select></label>
		<label>Frozen training set<select bind:value={datasetId}>{#each frozenDatasets as dataset}<option value={String(dataset.dataset_id)}>{text(dataset.name)} · {text(dataset.fingerprint_sha256).slice(0, 10)}</option>{/each}</select></label>
		<label>Compute endpoint<select bind:value={endpointId}>{#each computeEndpoints as endpoint}<option value={String(endpoint.endpoint_id)}>{text(endpoint.name)} · {text(endpoint.status)}</option>{/each}</select></label>
		<div class="gpu-allocation"><span>GPU allocation</span>{#if endpointGpus.length}<div class="gpu-chips"><button type="button" class:active={!gpuIds.length} on:click={() => gpuIds = []}>CPU</button>{#each endpointGpus as gpu}<button type="button" class:active={gpuIds.includes(String(gpu.id))} aria-pressed={gpuIds.includes(String(gpu.id))} on:click={() => toggleGpu(String(gpu.id))}>{gpuLabel(gpu)}</button>{/each}</div><small>Select one exact GPU. This allocation and its runtime device policy are sealed with the queued run.</small>{:else}<small>No GPU inventory is available for this endpoint. Queueing will use an explicit CPU allocation until GPU telemetry is visible.</small>{/if}</div>
		<div class="batch-allocation"><span>Training batch size</span><div class="choice-tabs"><button type="button" class:active={batchSizeMode === 'manual'} on:click={() => batchSizeMode = 'manual'}>Manual</button><button type="button" class:active={batchSizeMode === 'auto'} on:click={() => batchSizeMode = 'auto'}>Auto-tune</button></div>{#if batchSizeMode === 'manual'}<label>Batch size<input type="number" min="1" bind:value={batchSize} /></label><small>Sealed for this run only; it does not change the model definition.</small>{:else}<label>Maximum to test<input type="number" min="1" bind:value={maximumBatchSize} /></label><small>Starts at one sample per allocated GPU and doubles by powers of two. When GPU allocator telemetry is available, it chooses the first 30–80% VRAM candidate; otherwise it falls back to an OOM-bound safety margin.</small>{/if}</div>
		<label>Epochs per training phase<input type="number" min="1" bind:value={epochs} /></label>
	</div>
	<label>Queued run name<input bind:value={runName} placeholder="e.g. exp001-resnet-128 · frozen-v4" /></label>
	<div class="queue-preflight" aria-live="polite"><div class="preflight-heading"><div><p class="eyebrow">LAUNCH READINESS</p><h3>{missingRequirements.length ? `${missingRequirements.length} item${missingRequirements.length === 1 ? '' : 's'} needed before validation` : endpointReady ? 'Ready to validate and queue' : 'Ready to validate; endpoint needs attention'}</h3></div><span class:ready={canQueue} class:advisory={!canQueue && Boolean(selectedEndpoint)} class="preflight-state">{canQueue ? 'FORM COMPLETE' : 'INCOMPLETE'}</span></div><div class="preflight-checks">{#each queueRequirements as check}<div class:ready={check.state === 'ready'} class:advisory={check.state === 'advisory'} class:blocked={check.state === 'blocked'} class="preflight-check"><span aria-hidden="true">{check.state === 'ready' ? '✓' : check.state === 'advisory' ? '!' : '○'}</span><div><strong>{check.label}</strong><small>{check.detail}</small></div></div>{/each}</div></div>
	<div class="run-composer-footer"><span aria-live="polite">{validationFeedback || (busy && batchSizeMode === 'auto' ? 'Calibrating a safe batch size on the selected compute allocation. This can take several minutes.' : missingRequirements.length ? `To enable validation: ${missingRequirements.map((check) => check.label.toLowerCase()).join(', ')}.` : selectedDefinition ? `${text(selectedDefinition.name)} revision ${text(selectedDefinition.revision)} will be pinned.` : 'Choose a definition.')}</span><button disabled={busy || validationInProgress || !canQueue} on:click={validateAndQueue}>{validationInProgress ? String(validationOperation?.status) === 'queued' ? 'Validation queued…' : 'Validating…' : busy ? batchSizeMode === 'auto' ? 'Calibrating batch size…' : 'Validating…' : 'Validate & Queue Run'}</button></div>
</section>

<section class="queue-recovery panel"><div><p class="eyebrow">QUEUE RECOVERY</p><h2>Resolve stale state safely</h2><p>Reconcile marks a job stale only when this endpoint confirms its remote record is gone. Clear cancels local, non-running entries; use Logs and Clear on a finished row to diagnose it, then remove it from the active queue without deleting its audit trail.</p></div><div class="row-actions"><button class="secondary small" disabled={busy || !endpointId} on:click={resetStuck}>Reconcile stuck jobs</button><button class="quiet-button small" disabled={busy || !clearableQueued.length} on:click={clearQueue}>Clear non-running queue{clearableQueued.length ? ` (${clearableQueued.length})` : ''}</button></div></section>

<section class="panel endpoint-capacity" aria-live="polite">
	<div class="panel-head"><div><p class="eyebrow">AUTOMATIC SCHEDULING</p><h2>{selectedEndpoint ? `${text(selectedEndpoint.name)} capacity` : 'Select a compute endpoint'}</h2><p>{selectedEndpoint ? `Ready runs are started automatically in priority order when their sealed allocation fits. Scheduling is ${endpointSchedulingState}.` : 'Choose an endpoint above to see its latest capacity.'}</p></div><span class="status {endpointSchedulingState === 'paused' ? 'needs_attention' : 'ready'}">{endpointSchedulingState}</span></div>
	{#if selectedEndpoint}<div class="capacity-grid">
		<div><span>Worker slots</span><strong>{endpointFreeSlots != null && endpointWorkerSlots != null ? `${endpointFreeSlots} free / ${endpointWorkerSlots}` : endpointWorkerSlots ?? '—'}</strong><small>{endpointActiveSlots != null ? `${endpointActiveSlots} active` : 'Live slot telemetry unavailable'}</small></div>
		<div><span>CPU capacity</span><strong>{endpointCpuCapacity != null ? `${Math.max(0, endpointCpuCapacity - (endpointCpuInUse ?? 0))} free / ${endpointCpuCapacity}` : '—'}</strong><small>{endpointCpuInUse != null ? `${endpointCpuInUse} in use` : 'Live CPU telemetry unavailable'}</small></div>
		<div><span>GPU leases</span><strong>{endpointGpuLeases.length ? endpointGpuLeases.join(', ') : 'None reported'}</strong><small>{endpointGpus.length ? `${endpointGpus.length} GPU${endpointGpus.length === 1 ? '' : 's'} advertised` : 'GPU inventory unavailable'}</small></div>
		<div><span>Compute queue</span><strong>{number(endpointQueue.depth) != null ? `${text(endpointQueue.depth)} waiting` : '—'}</strong><small>{number(endpointQueue.capacity) != null ? `${text(endpointQueue.capacity)} queue capacity` : 'Queue telemetry unavailable'}</small></div>
	</div>{/if}
</section>

<section class="panel"><div class="panel-head"><div><h2>Run queue</h2><p>Select ready entries validated for the selected endpoint, then authorize automatic scheduling. Active runs publish progress and their latest training signal here.</p></div><div class="row-actions"><button class="secondary small" disabled={busy || !selected.size} on:click={() => start(false)}>Authorize selected</button><button class="small" disabled={busy || !queued.some((item) => item.status === 'ready' && item.preflight_endpoint_id === endpointId)} on:click={() => start(true)}>Authorize all ready here</button></div></div>
	{#if queued.length}<div class="queue-table-wrap"><table><thead><tr><th></th><th>Run</th><th>Definition</th><th>Dataset</th><th>Preflight</th><th>Scheduling</th><th>Progress</th><th>Live signal</th><th></th></tr></thead><tbody>{#each queued as item}{@const job = jobFor(item)}{@const live = liveStatusByJob[String(job?.job_id ?? job?.id ?? '')]}{@const progress = queueProgress(live)}{@const schedulingDetail = dispatchDetail(item, job)}{@const batchDetail = batchOverrideDetail(item)}{@const unavailable = workerUnavailable(live)}<tr class:attention={item.status === 'needs_attention' || unavailable}><td><input type="checkbox" disabled={item.status !== 'ready' || item.preflight_endpoint_id !== endpointId} checked={selected.has(String(item.queued_run_id))} aria-label={`Select ${text(item.name)}`} on:change={() => toggle(String(item.queued_run_id))} /></td><td><strong>{text(item.name)}</strong><small>Allocation: {allocationLabel(item.resources as RecordValue | undefined)} · Batch: {text(item.batch_size)} · Epochs: {text(item.epochs)}</small></td><td>{text(item.definition_name)}<small>revision {text(item.definition_revision)}</small></td><td>{text(item.dataset_id)}<small>{text(item.dataset_fingerprint_sha256).slice(0, 10)}</small></td><td><span class:valid={item.preflight_status === 'valid'} class="status">{item.preflight_status === 'valid' ? 'Validated' : 'Needs attention'}</span>{#if item.failure_reason}<small>{text(item.failure_reason)}</small>{/if}{#if batchDetail}<small>{batchDetail}</small>{/if}</td><td class="queue-scheduling">{#if unavailable}<span class="status needs_attention">Worker unavailable</span><small class="queue-interrupted">{workerUnavailableDetail(live)}</small>{:else}<span class="status {text(item.status)}">{statusLabel(item, job)}</span>{#if schedulingDetail}<small>{schedulingDetail}</small>{/if}{/if}{#if item.dispatch_claimed_at}<small>Claimed {text(item.dispatch_claimed_at)}</small>{/if}</td><td class="queue-progress">{#if unavailable && progress != null}<div class="queue-progress-label"><strong>Last known {Math.round(progress)}%</strong><span>epoch {text(live?.epoch)} / {text(live?.total_epochs)}</span></div><div class="queue-progress-track stale" aria-label={`Last known ${Math.round(progress)}% through the current epoch`}><i style={`width:${progress}%`}></i></div><small>Last recorded batch {text(live?.batch)} / {text(live?.total_batches)} · {lastKnownUpdate(live)}</small>{:else if unavailable}<small class="queue-interrupted">No persisted progress was found. Reconcile this missing job before retrying.</small>{:else if activeJob(job) && progress != null}<div class="queue-progress-label"><strong>{Math.round(progress)}%</strong><span>epoch {text(live?.epoch)} / {text(live?.total_epochs)}</span></div><div class="queue-progress-track" aria-label={`${Math.round(progress)}% through the current epoch`}><i style={`width:${progress}%`}></i></div><small>Batch {text(live?.batch)} / {text(live?.total_batches)} · ETA {formatDuration(live?.total_eta_seconds)}</small>{:else if activeJob(job)}<small>Awaiting worker telemetry…</small>{:else}<small>—</small>{/if}</td><td class="queue-live-signal">{#if unavailable}<p class="queue-interrupted"><strong>Live connection lost</strong>{workerUnavailableDetail(live)}</p>{:else if activeJob(job) && live && Object.keys(live).length}<div><span>Loss <strong>{formatNumber(liveMetric(live, 'loss'))}</strong></span><span>Accuracy <strong>{formatNumber(liveMetric(live, 'accuracy'))}</strong></span><span>Macro F1 <strong>{formatNumber(liveMetric(live, 'macro_f1') ?? liveMetric(live, 'val_macro_f1'))}</strong></span></div>{#if liveMetric(live, 'val_macro_f1') != null}<small>Validated F1 {formatNumber(liveMetric(live, 'val_macro_f1'))}</small>{/if}{:else}<small>—</small>{/if}</td><td><div class="job-controls">{#if job}<button class="secondary small" on:click={() => showLogs(job)}>Logs</button>{/if}{#if job && activeJob(job)}{#if unavailable}<button class="secondary small" disabled={busy || !endpointId} on:click={resetStuck}>Reconcile</button>{:else}{#if supportsControl(live, 'resume')}<button class="secondary small" disabled={Boolean(controllingJobId)} on:click={() => controlJob(job, 'resume')}>{controllingJobId === String(job.job_id) ? 'Working…' : 'Resume'}</button>{:else if supportsControl(live, 'pause')}<button class="secondary small" disabled={Boolean(controllingJobId)} on:click={() => controlJob(job, 'pause')}>{controllingJobId === String(job.job_id) ? 'Working…' : 'Pause'}</button>{/if}{#if supportsControl(live, 'cancel')}<button class="quiet-button small job-cancel" disabled={Boolean(controllingJobId)} on:click={() => controlJob(job, 'cancel')}>Cancel</button>{/if}{/if}<button class="secondary small" on:click={() => liveJob = job}>{unavailable ? 'Last known status' : 'Live dashboard'}</button>{:else if canOverrideBatch(item)}{#if batchOverrideQueuedRunId === String(item.queued_run_id)}<input class="queue-batch-override" type="number" min="1" bind:value={batchOverrideValue} aria-label={`Batch size for ${text(item.name)}`} /><button class="secondary small" disabled={busy || !Number.isInteger(batchOverrideValue) || batchOverrideValue < 1} on:click={() => saveBatchSize(item)}>Apply</button><button class="quiet-button small" disabled={busy} on:click={() => batchOverrideQueuedRunId = ''}>Cancel</button>{:else}<button class="secondary small" disabled={busy} on:click={() => editBatchSize(item)}>Adjust batch</button>{/if}<button class="quiet-button small" disabled={busy} on:click={() => cancel(String(item.queued_run_id))}>Cancel run</button>{:else if clearableTerminal(item, job)}<button class="quiet-button small" disabled={Boolean(clearingQueuedRunId)} on:click={() => clearTerminal(item, job)}>{clearingQueuedRunId === String(item.queued_run_id) ? 'Clearing…' : 'Clear'}</button>{:else if ['ready', 'needs_attention', 'waiting_for_resources', 'dispatching'].includes(String(item.status))}<button class="secondary small" on:click={() => cancel(String(item.queued_run_id))}>Cancel</button>{/if}</div></td></tr>{/each}</tbody></table></div>{:else}<p class="empty">No validated runs yet. Select a model definition and frozen dataset above.</p>{/if}
</section>

{#if liveJob}<TrainingStatusModal job={liveJob} runName={text(queued.find((item) => String(item.queued_run_id) === String(liveJob?.queued_run_id))?.name ?? liveJob.job_id)} onclose={() => liveJob = null} />{/if}

{#if logJob}
	<section class="panel run-log" aria-label="Run logs">
		<div class="panel-head"><div><p class="eyebrow">RUN DIAGNOSTICS</p><h2>Execution log · {text(logJob.job_id ?? logJob.id)}</h2><p>Worker output and lifecycle events are retained after a failed or cleared run.</p></div><button class="secondary small" on:click={() => { logJob = null; logEvents = []; }}>Close logs</button></div>
		{#if logsLoading}<p class="empty">Loading run events…</p>
		{:else if logEvents.length}<ol class="log-events">{#each logEvents as event}<li><time>{text(event.timestamp)}</time><div><strong>{text(event.event_type ?? event.type)}</strong><p>{text(event.message)}</p>{#if event.data && Object.keys(record(event.data)).length}<details><summary>Event details</summary><pre>{JSON.stringify(record(event.data), null, 2)}</pre></details>{/if}</div></li>{/each}</ol>
		{:else}<p class="empty">No persisted worker events are available yet. Refresh the queue after the worker reports its terminal state.</p>{/if}
	</section>
{/if}

<style>
	.endpoint-capacity { margin-top: 1rem; }
	.capacity-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: .7rem; }
	.capacity-grid > div { display: grid; gap: .22rem; padding: .75rem .85rem; border: 1px solid var(--line); border-radius: .65rem; background: var(--surface-soft); min-width: 0; }
	.capacity-grid span, .capacity-grid small { color: var(--muted); font-size: .72rem; }
	.capacity-grid strong { overflow-wrap: anywhere; font-size: .92rem; }
	.queue-table-wrap { overflow-x: auto; margin: 0 -0.2rem; }
	.queue-table-wrap table { min-width: 1220px; }
	.queue-scheduling { min-width: 13rem; }
	.queue-scheduling small { display: block; margin-top: .28rem; color: var(--muted); font-size: .67rem; overflow-wrap: anywhere; }
	.queue-progress { min-width: 10.8rem; }
	.queue-progress-label { display: flex; justify-content: space-between; gap: .5rem; font-size: .72rem; }
	.queue-progress-label span { color: var(--muted); }
	.queue-progress-track { height: .38rem; overflow: hidden; margin: .32rem 0; border-radius: 999px; background: #dbeafe; }
	.queue-progress-track i { display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg, #0f766e, #2dd4bf); }
	.queue-progress-track.stale { background: #fee2e2; }
	.queue-progress-track.stale i { background: linear-gradient(90deg, #dc2626, #f59e0b); }
	.queue-progress small, .queue-live-signal small { display: block; color: var(--muted); font-size: .67rem; white-space: nowrap; }
	.queue-live-signal { min-width: 13rem; }
	.queue-live-signal > div { display: flex; flex-wrap: wrap; gap: .3rem .55rem; }
	.queue-live-signal span { color: var(--muted); font-size: .68rem; white-space: nowrap; }
	.queue-live-signal strong { color: var(--ink); }
	.queue-interrupted { display: block; margin: .35rem 0 0; color: #991b1b !important; font-size: .68rem; line-height: 1.4; white-space: normal !important; }
	.queue-interrupted strong { display: block; color: #991b1b; font-size: .7rem; }
	.job-controls { display: flex; flex-wrap: wrap; gap: .3rem; min-width: 9rem; }
	.job-cancel { color: #b91c1c; border-color: #fecaca; }
	.run-log { margin-top: 1rem; }
	.log-events { display: grid; gap: .7rem; max-height: 32rem; overflow: auto; margin: 0; padding: 0; list-style: none; }
	.log-events li { display: grid; grid-template-columns: 11rem minmax(0, 1fr); gap: .8rem; padding: .75rem .9rem; border: 1px solid var(--line); border-radius: .65rem; background: var(--surface-soft); }
	.log-events time { color: var(--muted); font-size: .75rem; font-variant-numeric: tabular-nums; }
	.log-events p { margin: .2rem 0 0; overflow-wrap: anywhere; }
	.log-events details { margin-top: .45rem; color: var(--muted); font-size: .78rem; }
	.log-events pre { overflow: auto; padding: .55rem; border-radius: .4rem; background: var(--ink); color: #e2e8f0; font-size: .72rem; }
	@media (max-width: 900px) { .capacity-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
	@media (max-width: 760px) { .capacity-grid { grid-template-columns: 1fr; } .log-events li { grid-template-columns: 1fr; gap: .35rem; } }
</style>
