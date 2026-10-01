<script lang="ts">
	import { onMount } from 'svelte';
	import { subscribeOperationalRefresh } from '$lib/operational-refresh';
	import { api, type RecordValue } from '$lib/api';

	export let onchanged: (message: string) => void = () => {};
	export let onfailure: (message: string) => void = () => {};
	export let mode: 'queue' | 'workers' = 'queue';

	let queuedRuns: RecordValue[] = [];
	let selectedRunIds: string[] = [];
	let startAfterVerification = false;
	let queueAction = false;
	let queueActionMessage = '';
	const selectable = (run: RecordValue) => ['pending_verification', 'ready', 'needs_attention'].includes(String(run.status)) && !run.start_authorized;
	const runStatus = (run: RecordValue) => run.status === 'queued' && ['failed', 'cancelled', 'dispatch_failed', 'artifact_invalid'].includes(String(run.latest_job_status)) ? 'Previous attempt failed' : ({ pending_verification: 'Awaiting verification', verifying: 'Verifying', ready: 'Verified · ready', needs_attention: 'Needs verification', queued: 'Waiting to train', indexed: 'Complete' }[String(run.status)] ?? String(run.status).replaceAll('_', ' '));
	$: selectedRuns = queuedRuns.filter((run) => selectedRunIds.includes(String(run.queued_run_id)));
	$: canVerify = selectedRuns.length > 0 && selectedRuns.every(selectable);
	$: canStart = selectedRuns.length > 0 && selectedRuns.every((run) => run.status === 'ready' && run.preflight_status === 'valid');
	function toggleRun(id: string, checked: boolean) { selectedRunIds = checked ? [...selectedRunIds, id] : selectedRunIds.filter((value) => value !== id); }
	async function actOnSelection(action: 'verify' | 'start') {
		queueAction = true; queueActionMessage = '';
		try {
			if (action === 'verify') {
				await api.verifyQueuedRuns({ queued_run_ids: selectedRunIds, start_after_verification: startAfterVerification });
				queueActionMessage = startAfterVerification ? 'Verification requested. Each passing run will start automatically.' : 'Verification requested. Passing runs will wait for you to start training.';
			} else {
				for (const poolId of new Set(selectedRuns.map((run) => String(run.worker_pool_id)))) {
					await api.authorizeQueuedRuns({ worker_pool_id: poolId, queued_run_ids: selectedRuns.filter((run) => run.worker_pool_id === poolId).map((run) => run.queued_run_id) });
				}
				queueActionMessage = 'Training authorized. Runs will start when their verified worker is available.';
			}
			selectedRunIds = []; startAfterVerification = false;
			onchanged(queueActionMessage);
		} catch (error) { onfailure(error instanceof Error ? error.message : 'Queue action failed.'); }
		finally { queueAction = false; await load(); }
	}
	let pools: RecordValue[] = [];
	let workers: RecordValue[] = [];
	let leases: RecordValue[] = [];
	let definitions: RecordValue[] = [];
	let datasets: RecordValue[] = [];
	let loading = true;
	let creating = false;
	let showCreate = false;
	let poolName = '';
	let allowedActions = 'train';
	let maxWorkers = '';
	let issuedJoinToken = '';
	let selectedPoolId = '';
	let enqueuing = false;
	let queueName = '';
	let editingRunId = '';
	let editingDefinitionName = '';
	let editingOriginalDefinitionId = '';
	let editingRevision = 0;
	let useLatestRevision = false;
	let editingPoolNotice = '';
	let selectedDefinitionId = '';
	let selectedDatasetId = '';
	let batchSizeMode: 'manual' | 'auto' = 'manual';
	let batchSize = 16;
	let maximumBatchSize = 256;
	let epochs = 10;
	let gpuCount = 0;
	let queueFeedback: { state: 'working' | 'success' | 'error'; message: string; queueId?: string } | null = null;

	const records = (value: unknown): RecordValue[] => Array.isArray(value) ? value as RecordValue[] : [];
	const object = (value: unknown): RecordValue => value && typeof value === 'object' && !Array.isArray(value) ? value as RecordValue : {};
	const text = (value: unknown, fallback = '—') => value === undefined || value === null || value === '' ? fallback : String(value);
	const actions = (value: unknown): string[] => Array.isArray(value) ? value.map(String) : [];
	const output = (lease: RecordValue) => object(lease.output_ref);
	const isActive = (lease: RecordValue) => ['active', 'acknowledged', 'uploading'].includes(String(lease.status));
	const gpuCapacity = (worker: RecordValue) => {
		const ids = object(worker.capabilities).gpu_ids;
		return Array.isArray(ids) ? ids.length : 0;
	};
	const workerSummary = (worker: RecordValue) => {
		const capabilities = object(worker.capabilities);
		const supported = actions(capabilities.actions).join(', ') || 'no actions';
		const gpuText = `${gpuCapacity(worker)} GPU${gpuCapacity(worker) === 1 ? '' : 's'}`;
		const cpu = typeof capabilities.cpu_capacity === 'number' ? ` · ${capabilities.cpu_capacity} CPU` : '';
		return `${supported} · ${gpuText}${cpu}`;
	};
	$: activeLeases = leases.filter(isActive);
	$: publishedOutputs = leases.filter((lease) => Object.keys(output(lease)).length > 0);
	$: workerCountFor = (poolId: unknown) => workers.filter((worker) => String(worker.pool_id) === String(poolId)).length;
	$: selectedPool = pools.find((pool) => String(pool.pool_id) === selectedPoolId);
	$: selectedPoolWorkers = workers.filter((worker) => String(worker.pool_id) === selectedPoolId);
	$: poolAllowsTraining = selectedPool?.enabled !== false && actions(selectedPool?.allowed_actions).includes('train');
	$: trainingWorkers = selectedPoolWorkers.filter((worker) => worker.state !== 'offline' && actions(object(worker.capabilities).actions).includes('train'));
	$: gpuWorkers = trainingWorkers.filter((worker) => gpuCapacity(worker) >= gpuCount && object(worker.capabilities).training_verification_v1 === true && object(worker.capabilities).work_unit_v2 === true);
	$: selectedWorkerMessage = !selectedPool
		? 'Choose a worker pool.'
		: !poolAllowsTraining
			? 'This pool does not allow training work. Choose a pool whose allowed actions include train.'
			: !trainingWorkers.length
				? 'No registered worker in this pool advertises the train action yet. You may queue work now; it will wait safely for one.'
				: !trainingWorkers.some((worker) => object(worker.capabilities).training_verification_v1 === true)
					? 'Restart training workers with the updated Oracle Builder to enable verification.'
				: gpuCount > 0 && !gpuWorkers.length
					? `No registered training worker in this pool currently advertises ${gpuCount} GPU${gpuCount === 1 ? '' : 's'}. The job can be queued, but cannot start until compatible capacity registers.`
					: `${gpuWorkers.length} compatible worker${gpuWorkers.length === 1 ? '' : 's'} can claim this run when idle.`;

	async function load() {
		loading = true;
		try {
			const [poolResult, workerResult, leaseResult, definitionResult, datasetResult, queueResult] = await Promise.all([
				api.workerPools(), api.registeredWorkers(), api.workerLeases(), api.modelDefinitions(), api.datasets(), api.queuedRuns()
			]);
			queuedRuns = records(queueResult.queued_runs);
			selectedRunIds = selectedRunIds.filter((id) => queuedRuns.some((run) => String(run.queued_run_id) === id && selectable(run)));
			pools = records(poolResult.pools);
			workers = records(workerResult.workers);
			leases = records(leaseResult.leases);
			definitions = records(definitionResult.definitions);
			datasets = records(datasetResult.datasets).filter((dataset) => String(dataset.lifecycle) === 'frozen');
			if (!selectedPoolId && pools[0]) selectedPoolId = String(pools[0].pool_id);
			if (!selectedDefinitionId && definitions[0]) selectedDefinitionId = String(definitions[0].definition_id);
			if (!selectedDatasetId && datasets[0]) selectedDatasetId = String(datasets[0].dataset_id);
		} catch (error) {
			onfailure(error instanceof Error ? error.message : 'Could not load pull-worker fleet state.');
		} finally { loading = false; }
	}

	async function queueDefinition() {
		if (!selectedDefinitionId || !selectedDatasetId || !selectedPoolId || !queueName.trim()) return;
		if (!poolAllowsTraining) {
			const message = 'Choose a worker pool that permits the train action before queueing training work.';
			queueFeedback = { state: 'error', message };
			onfailure(message);
			return;
		}
		if ((batchSizeMode === 'manual' && (!Number.isInteger(batchSize) || batchSize < 1))
			|| (batchSizeMode === 'auto' && (!Number.isInteger(maximumBatchSize) || maximumBatchSize < 1))
			|| !Number.isInteger(epochs) || epochs < 1 || !Number.isInteger(gpuCount) || gpuCount < 0) {
			onfailure('Batch settings and epochs must be positive whole numbers; GPU count cannot be negative.');
			return;
		}
		enqueuing = true;
		queueFeedback = { state: 'working', message: editingRunId ? 'Saving an updated queued run…' : 'Adding the definition and frozen input to the queue…' };
		try {
			const payload = {
				name: queueName.trim(), dataset_id: selectedDatasetId, worker_pool_id: selectedPoolId,
				resources: { gpu_count: gpuCount }, batch_size_mode: batchSizeMode, epochs,
				...(editingRunId ? { definition_id: selectedDefinitionId, ...(selectedDefinitionId === editingOriginalDefinitionId && !useLatestRevision ? { definition_revision: editingRevision } : {}) } : {}),
				...(batchSizeMode === 'manual' ? { batch_size: batchSize } : { maximum_batch_size: maximumBatchSize })
			};
			const queued = editingRunId ? await api.replaceQueuedRun(editingRunId, payload) : await api.queueDefinition(selectedDefinitionId, payload);
			const queueId = text(queued.queued_run_id, '');
			if (!queueId) throw new Error('The Orchestrator did not return the durable queued-run id.');
			const successMessage = editingRunId ? 'Updated run is queued. Select it below to verify before training.' : 'Added to the queue. Select this run below to verify it before training.';
			selectedRunIds = [...selectedRunIds, queueId];
			if (editingRunId) cancelEdit(); else queueName = '';
			queueFeedback = { state: 'success', queueId, message: successMessage };
			await load();
			onchanged('Training run is queued. Verification and start are separate steps.');
		} catch (error) {
			const message = error instanceof Error ? error.message : 'Could not add the training run to the queue.';
			queueFeedback = { state: 'error', message };
			onfailure(message);
		}
		finally { enqueuing = false; }
	}

	function cancelEdit() {
		editingRunId = ''; editingDefinitionName = ''; editingOriginalDefinitionId = ''; editingRevision = 0; useLatestRevision = false; editingPoolNotice = '';
		queueName = ''; queueFeedback = null;
		selectedDefinitionId = String(definitions[0]?.definition_id ?? '');
		selectedDatasetId = String(datasets[0]?.dataset_id ?? '');
		selectedPoolId = String(pools[0]?.pool_id ?? '');
		batchSizeMode = 'manual'; batchSize = 16; maximumBatchSize = 256; epochs = 10; gpuCount = 0;
	}

	function editRun(run: RecordValue) {
		editingRunId = String(run.queued_run_id);
		editingDefinitionName = `${text(run.definition_name)} · rev ${text(run.definition_revision)}`;
		editingOriginalDefinitionId = String(run.definition_id);
		editingRevision = Number(run.definition_revision);
		useLatestRevision = false;
		queueName = String(run.name ?? '');
		selectedDefinitionId = String(run.definition_id);
		selectedDatasetId = String(run.dataset_id);
		gpuCount = Number(object(run.resources).gpu_count ?? 0);
		const compatible = (pool: RecordValue) => pool.enabled !== false && actions(pool.allowed_actions).includes('train') && workers.some((worker) => String(worker.pool_id) === String(pool.pool_id) && worker.state !== 'offline' && actions(object(worker.capabilities).actions).includes('train') && object(worker.capabilities).training_verification_v1 === true && object(worker.capabilities).work_unit_v2 === true && gpuCapacity(worker) >= gpuCount);
		const originalPool = pools.find((pool) => String(pool.pool_id) === String(run.worker_pool_id));
		const chosenPool = originalPool && compatible(originalPool) ? originalPool : pools.find(compatible) ?? originalPool;
		selectedPoolId = String(chosenPool?.pool_id ?? run.worker_pool_id);
		editingPoolNotice = originalPool && chosenPool && originalPool.pool_id !== chosenPool.pool_id ? `The original pool has no compatible verifier. Selected ${text(chosenPool.name)}; review this choice before saving.` : '';
		const policy = object(object(run.batch_execution).requested_policy ?? run.batch_execution);
		batchSizeMode = policy.mode === 'auto' ? 'auto' : 'manual';
		batchSize = Number(policy.batch_size ?? run.batch_size ?? 16);
		maximumBatchSize = Number(policy.maximum_batch_size ?? 256);
		epochs = Number(run.epochs ?? policy.epochs ?? 10);
		queueFeedback = null;
		document.getElementById('queue-run-form')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
	}

	async function createPool() {
		const parsedActions = allowedActions.split(',').map((action) => action.trim()).filter(Boolean);
		const parsedMax = maxWorkers.trim() ? Number(maxWorkers) : undefined;
		if (!poolName.trim() || !parsedActions.length || (parsedMax !== undefined && (!Number.isInteger(parsedMax) || parsedMax < 1))) {
			onfailure('Enter a pool name, at least one supported action, and an optional whole-number worker limit.');
			return;
		}
		creating = true; issuedJoinToken = '';
		try {
			const created = await api.createWorkerPool({ name: poolName.trim(), allowed_actions: parsedActions, ...(parsedMax !== undefined ? { max_workers: parsedMax } : {}) });
			issuedJoinToken = text(created.registration_token, '');
			poolName = ''; allowedActions = 'train'; maxWorkers = ''; showCreate = false;
			await load();
			onchanged(`Created pull-worker pool ${text(created.name)}. Save its join token now; it will not be shown again.`);
		} catch (error) { onfailure(error instanceof Error ? error.message : 'Could not create the worker pool.'); }
		finally { creating = false; }
	}

	onMount(() => {
		void load();
		return subscribeOperationalRefresh(() => { if (!loading) void load(); });
	});
</script>

<section class="fleet-intro panel">
	{#if mode === 'queue'}
		<div><p class="eyebrow">TRAINING QUEUE</p><h2>Queue. Verify. Start when ready.</h2><p>Select an immutable definition, a frozen dataset, and the worker capacity it needs. Select one or more queued runs to check them on a worker, review the results, then start training.</p></div>
	{:else}
		<div><p class="eyebrow">WORKER FLEET</p><h2>Manage the execution capacity behind queued work.</h2><p>Create admission pools, register workers, and inspect durable leases. This page never submits a training run.</p></div>
	{/if}
	<div class="row-actions"><button class="secondary small" disabled={loading} on:click={load}>{loading ? 'Refreshing…' : 'Refresh'}</button>{#if mode === 'workers'}<button class="small" on:click={() => showCreate = !showCreate}>{showCreate ? 'Close' : 'Create pool'}</button>{/if}</div>
</section>

{#if mode === 'workers' && issuedJoinToken}
	<section class="fleet-token" role="status"><strong>Pool join token — save it now</strong><code>{issuedJoinToken}</code><p>Give this token only to the deployment process that will run <code>oracle-worker pull</code>. It cannot be retrieved later.</p><button class="quiet-button small" on:click={() => issuedJoinToken = ''}>I saved it</button></section>
{/if}

{#if mode === 'workers' && showCreate}
	<section class="panel fleet-create"><div class="panel-head"><div><p class="eyebrow">NEW ADMISSION BOUNDARY</p><h2>Create a worker pool</h2><p>Pool actions are an allowlist enforced before a worker can lease a work unit.</p></div></div><div class="fleet-form"><label>Name<input bind:value={poolName} placeholder="e.g. GPU training workers" /></label><label>Allowed actions<input bind:value={allowedActions} placeholder="e.g. train" /><small>Comma-separated supported WorkUnit actions.</small></label><label>Maximum workers (optional)<input type="number" min="1" bind:value={maxWorkers} placeholder="No limit" /></label></div><div class="row-actions"><button disabled={creating} on:click={createPool}>{creating ? 'Creating…' : 'Create pool'}</button></div></section>
{/if}

{#if mode === 'queue'}
	<section id="queue-run-form" class="panel fleet-enqueue"><div class="panel-head"><div><p class="eyebrow">{editingRunId ? 'EDIT QUEUED RUN' : 'SUBMIT'}</p><h2>{editingRunId ? 'Update training run' : 'Prepare a training run'}</h2><p>{editingRunId ? 'Saving creates a new sealed run and archives this queue entry. Prior jobs remain in the history; verify the updated run before starting.' : 'Adding a run pins the definition and dataset. Verification and training happen only when you request them below.'}</p></div>{#if editingRunId}<button class="secondary small" disabled={enqueuing} on:click={cancelEdit}>Cancel edit</button>{/if}</div>{#if editingPoolNotice}<p class="queue-result" role="status">{editingPoolNotice}</p>{/if}{#if pools.length && definitions.length && datasets.length}<div class="fleet-form run-form"><label>Run name<input bind:value={queueName} placeholder="e.g. resnet18-isiisnet-1epoch" aria-describedby="run-name-help" /><small id="run-name-help">A human-readable label for this immutable run.</small></label><label>Definition<select bind:value={selectedDefinitionId}>{#each definitions as definition}<option value={String(definition.definition_id)}>{text(definition.name)} · rev {editingRunId && String(definition.definition_id) === editingOriginalDefinitionId && !useLatestRevision ? editingRevision : text(definition.revision)}</option>{/each}</select>{#if editingRunId}<small>Original: {editingDefinitionName}. Changing definitions creates a new sealed run.</small>{#if selectedDefinitionId === editingOriginalDefinitionId && Number(definitions.find((definition) => String(definition.definition_id) === selectedDefinitionId)?.revision) !== editingRevision}<small class="latest-revision"><input type="checkbox" bind:checked={useLatestRevision} /> Use latest revision</small>{/if}{/if}</label><label>Frozen dataset<select bind:value={selectedDatasetId}>{#each datasets as dataset}<option value={String(dataset.dataset_id)}>{text(dataset.name)}</option>{/each}</select></label><label>Worker pool<select bind:value={selectedPoolId}>{#each pools as pool}<option value={String(pool.pool_id)}>{text(pool.name)} · {actions(pool.allowed_actions).join(', ')}</option>{/each}</select></label><label>GPUs required<input type="number" min="0" bind:value={gpuCount} /><small>Use 0 for CPU-only work.</small></label><label>Batch sizing<select bind:value={batchSizeMode}><option value="manual">Set manually</option><option value="auto">Automatically calibrate on worker</option></select><small>{batchSizeMode === 'auto' ? 'Verification tests safe sizes on a compatible worker.' : 'Use a known-safe batch size.'}</small></label>{#if batchSizeMode === 'manual'}<label>Batch size<input type="number" min="1" bind:value={batchSize} /></label>{:else}<label>Maximum batch size<input type="number" min="1" bind:value={maximumBatchSize} /><small>Calibration chooses a conservative safe size at or below this limit.</small></label>{/if}<label>Epochs<input type="number" min="1" bind:value={epochs} /></label><div class="enqueue-action"><button disabled={enqueuing || !queueName.trim() || !poolAllowsTraining} on:click={queueDefinition}>{enqueuing ? 'Saving…' : editingRunId ? 'Save updated run' : 'Add to queue'}</button><small>{editingRunId ? 'The updated run will need verification before training.' : 'Adding to the queue does not start verification or training.'}</small></div></div><div class:warning={!poolAllowsTraining || (gpuCount > 0 && trainingWorkers.length > 0 && !gpuWorkers.length)} class="capacity-guidance" role="status"><strong>Worker capacity check</strong><span>{selectedWorkerMessage}</span><small>Selected pool: {text(selectedPool?.name)} · {selectedPoolWorkers.length} registered · {trainingWorkers.length} train-capable · {gpuWorkers.length} matching GPU request</small></div>{#if queueFeedback}<div class:success={queueFeedback.state === 'success'} class:error={queueFeedback.state === 'error'} class:working={queueFeedback.state === 'working'} class="queue-feedback" role={queueFeedback.state === 'error' ? 'alert' : 'status'}><strong>{queueFeedback.state === 'working' ? 'Preparing portable work' : queueFeedback.state === 'success' ? 'Training run queued' : 'Could not queue training run'}</strong><span>{queueFeedback.message}</span>{#if queueFeedback.queueId}<small>Queued run: <code>{queueFeedback.queueId}</code></small>{/if}</div>{/if}{:else}<p class="empty">Before queueing: create a worker pool in Workers, save a versioned definition, and freeze a dataset revision.</p>{/if}</section>
{/if}

{#if mode === 'queue'}
<section class="panel queue-workflow">
	<div class="panel-head"><div><p class="eyebrow">YOUR TRAINING QUEUE</p><h2>Review and launch</h2><p>Verification checks the data and model, probes batch capacity, and records the result. Training uses the same worker; reverify to choose a new worker.</p></div><span class="status">{queuedRuns.length} runs</span></div>
	<div class="queue-toolbar">
		<span>{selectedRuns.length} selected</span>
		<label class="verify-start"><input type="checkbox" bind:checked={startAfterVerification} disabled={queueAction} /> Start training after successful verification</label>
		<button class="secondary" disabled={queueAction || !canVerify} on:click={() => actOnSelection('verify')}>{startAfterVerification ? 'Verify & start selected' : 'Verify selected'}</button>
		<button disabled={queueAction || !canStart} on:click={() => actOnSelection('start')}>Start selected</button>
	</div>
	{#if queueActionMessage}<p class="queue-result" role="status">{queueActionMessage}</p>{/if}
	{#if queuedRuns.length}<div class="table-scroll"><table class="fleet-table"><thead><tr><th><input type="checkbox" aria-label="Select all unstarted runs" disabled={queueAction} checked={queuedRuns.some(selectable) && queuedRuns.filter(selectable).every((run) => selectedRunIds.includes(String(run.queued_run_id)))} on:change={(event) => selectedRunIds = event.currentTarget.checked ? queuedRuns.filter(selectable).map((run) => String(run.queued_run_id)) : []} /></th><th>Run / definition</th><th>Stage</th><th>Batch / epochs</th><th>Verification details</th><th>Actions</th></tr></thead><tbody>
	{#each queuedRuns as run}
		{@const policy = object(run.batch_execution)}
		{@const report = object(run.preflight_report)}
		{@const splitManifest = object(report.split_manifest)}
		{@const coverage = object(splitManifest.coverage)}
		<tr><td><input type="checkbox" aria-label={`Select ${text(run.name)}`} disabled={queueAction || !selectable(run)} checked={selectedRunIds.includes(String(run.queued_run_id))} on:change={(event) => toggleRun(String(run.queued_run_id), event.currentTarget.checked)} /></td><td><strong>{text(run.name)}</strong><small>{text(run.definition_name)} · rev {text(run.definition_revision)}</small><small>{text(pools.find((pool) => pool.pool_id === run.worker_pool_id)?.name)}</small></td><td><span class="status {text(run.status)}">{runStatus(run)}</span>{#if run.status === 'verifying'}<small>{run.start_authorized ? 'Will start if verification passes' : 'Will wait for your start command'}</small>{/if}</td><td><strong>{policy.mode === 'auto' ? 'Automatic · pending' : `${text(run.batch_size)} / batch`}</strong><small>{text(run.epochs)} epochs</small></td><td>{#if run.latest_job_error && run.can_edit}<small class="queue-problem">{text(run.latest_job_error)}</small>{:else if run.failure_reason}<small class="queue-problem">{text(run.failure_reason)}</small>{:else if run.status === 'verifying'}<small>Waiting for or checking on a verification-capable worker.</small>{:else if report.ready}<strong>Preflight passed</strong><small>{text(workers.find((worker) => worker.worker_id === report.worker_id)?.name, text(report.worker_id))}</small>{:else}<small>{actions(report.reasons).join('; ') || 'Select this run to verify.'}</small>{/if}{#if report.ready}<details><summary>View checks</summary><ul>{#each actions(report.checks) as check}<li>{check.replaceAll('_', ' ')}</li>{/each}</ul>{#if Object.keys(coverage).length}<strong>Split coverage {coverage.valid === false ? 'needs attention' : 'verified'}</strong><small>Minimum per class: {text(coverage.minimum_per_class)}</small><dl class="split-coverage">{#each Object.entries(object(coverage.class_counts)) as [classId, counts]}<dt>Class {classId}</dt><dd>{#each Object.entries(object(counts)) as [split, count], index}{index ? ' · ' : ''}{split}: {count}{/each}</dd>{/each}</dl>{/if}<small>Verified {text(report.verified_at)}</small></details>{/if}</td><td>{#if run.can_edit}<button class="secondary small" disabled={enqueuing || queueAction} on:click={() => editRun(run)}>Edit</button>{:else}<small>{run.status === 'queued' ? 'Active or waiting on worker' : '—'}</small>{/if}</td></tr>
	{/each}</tbody></table></div>{:else}<p class="empty">Your queue is empty. Add a training run above to get started.</p>{/if}
</section>
{/if}

{#if mode === 'workers'}
<section class="fleet-stats" aria-label="Pull worker fleet summary"><article><span>Worker pools</span><strong>{pools.length}</strong></article><article><span>Registered workers</span><strong>{workers.length}</strong></article><article><span>Active leases</span><strong>{activeLeases.length}</strong></article><article><span>Published outputs</span><strong>{publishedOutputs.length}</strong></article></section>

<section class="panel"><div class="panel-head"><div><p class="eyebrow">ADMISSION</p><h2>Worker pools</h2><p>Use the one-time pool join token with the worker CLI. Join secrets are intentionally never listed here.</p></div></div>{#if pools.length}<div class="table-scroll"><table class="fleet-table"><thead><tr><th>Pool</th><th>Allowed actions</th><th>Capacity</th><th>Workers</th><th>Created</th></tr></thead><tbody>{#each pools as pool}<tr><td><strong>{text(pool.name)}</strong><small class="code">{text(pool.pool_id)}</small></td><td>{actions(pool.allowed_actions).join(', ') || '—'}</td><td>{pool.max_workers == null ? 'Unlimited' : text(pool.max_workers)}</td><td>{workerCountFor(pool.pool_id)}</td><td>{text(pool.created_at)}</td></tr>{/each}</tbody></table></div>{:else}<p class="empty">No pull-worker pools yet. Create a pool, then register an <code>oracle-worker pull</code> deployment with its one-time join token.</p>{/if}</section>

<section class="panel"><div class="panel-head"><div><p class="eyebrow">REGISTERED EXECUTORS · STEP 2 OF 2</p><h2>Workers claim compatible runs</h2><p>A worker must be registered in the selected pool, support training, and advertise enough GPUs for the request. Its detailed credentials remain private.</p></div></div>{#if workers.length}<div class="table-scroll"><table class="fleet-table"><thead><tr><th>Worker</th><th>Pool</th><th>State</th><th>What it can run</th><th>Last seen</th></tr></thead><tbody>{#each workers as worker}<tr><td><strong>{text(worker.name)}</strong><small class="code">{text(worker.worker_id)}</small>{#if worker.endpoint}<small>{text(worker.endpoint)}</small>{/if}</td><td>{text(pools.find((pool) => String(pool.pool_id) === String(worker.pool_id))?.name, text(worker.pool_id))}</td><td><span class="status {text(worker.state)}">{text(worker.state)}</span></td><td>{workerSummary(worker)}</td><td>{text(worker.last_seen_at)}</td></tr>{/each}</tbody></table></div>{:else}<p class="empty">No worker has registered. Create a pool, save its one-time join token, then start <code>oracle-worker pull</code>. Queued work remains safe until a worker joins.</p>{/if}</section>

<section class="panel"><div class="panel-head"><div><p class="eyebrow">DURABLE WORK CLAIMS</p><h2>Leases and published output</h2><p>A lease is a time-bounded claim. Output references appear only after validation, sealing, and successful publication.</p></div></div>{#if leases.length}<div class="table-scroll"><table class="fleet-table"><thead><tr><th>Lease</th><th>Worker</th><th>Job</th><th>Status</th><th>Expires / released</th><th>Published output</th></tr></thead><tbody>{#each leases as lease}<tr><td><code>{text(lease.lease_id)}</code></td><td>{text(workers.find((worker) => String(worker.worker_id) === String(lease.worker_id))?.name, text(lease.worker_id))}</td><td><code>{text(lease.job_id)}</code></td><td><span class="status {text(lease.status)}">{text(lease.status)}</span>{#if lease.outcome}<small>{text(lease.outcome)}</small>{/if}</td><td>{text(lease.released_at ?? lease.expires_at)}</td><td>{#if Object.keys(output(lease)).length}<strong>{text(output(lease).kind)}</strong><small class="code">{text(output(lease).artifact_id)} · rev {text(output(lease).revision)}</small>{:else}<small>Not published</small>{/if}</td></tr>{/each}</tbody></table></div>{:else}<p class="empty">No durable pull-worker leases have been issued yet.</p>{/if}</section>
{/if}

<style>
	.queue-toolbar { display:flex; flex-wrap:wrap; align-items:center; gap:.75rem; padding:.9rem 0; border-block:1px solid var(--line); margin-bottom:1rem; }
	.queue-toolbar > span { font-size:.8rem; color:var(--muted); margin-right:auto; }
	.verify-start { display:flex; align-items:center; gap:.5rem; font-size:.8rem; }
	.queue-result { padding:.7rem; background:var(--teal-soft); border-radius:6px; color:var(--teal-dark); }
	.latest-revision { display:flex; align-items:center; gap:.4rem; }
	.latest-revision input[type=checkbox] { width:1rem; height:1rem; margin:0; }
	.queue-problem { color:var(--red) !important; }
	.queue-workflow input[type=checkbox] { width:1rem; height:1rem; accent-color:var(--teal); }
	.queue-workflow details { margin-top:.4rem; font-size:.75rem; }
	.queue-workflow summary { cursor:pointer; color:var(--teal); }
	.split-coverage { display:grid; grid-template-columns:auto minmax(0,1fr); gap:.2rem .55rem; margin:.45rem 0; font-size:.67rem; }.split-coverage dt { color:var(--muted); }.split-coverage dd { margin:0; overflow-wrap:anywhere; }

	.fleet-intro { display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; }
	.fleet-intro p { max-width:48rem; margin-bottom:0; }
	.fleet-token { display:grid; gap:.45rem; margin:0 0 1rem; padding:1rem 1.1rem; border:1px solid #e5bd78; border-radius:10px; background:#fff8e9; color:#664414; }
	.queue-feedback,.capacity-guidance { display:grid; gap:.3rem; margin-top:.85rem; padding:.8rem .9rem; border:1px solid #b8d4ce; border-radius:8px; background:#f1f9f6; color:#174d43; font-size:.78rem; }.queue-feedback.error { border-color:#e0aaa3; background:#fff4f2; color:#8b3026; }.queue-feedback.working { border-color:#c3d0e5; background:#f3f7fd; color:#294e7a; }.queue-feedback span { line-height:1.35; }.queue-feedback small,.capacity-guidance small { color:inherit; opacity:.85; }.queue-feedback code { overflow-wrap:anywhere; }
	.capacity-guidance { grid-template-columns:auto minmax(0,1fr); align-items:baseline; border-color:#c3d0e5; background:#f3f7fd; color:#294e7a; }.capacity-guidance small { grid-column:1/-1; }.capacity-guidance.warning { border-color:#e7c68d; background:#fff8eb; color:#76531e; }
	.fleet-token code { overflow-wrap:anywhere; padding:.55rem .65rem; border:1px solid #ecd29b; border-radius:6px; background:#fff; color:#573a10; font-size:.75rem; }
	.fleet-token p { margin:0; color:#76531e; font-size:.78rem; }.fleet-token .quiet-button { justify-self:start; }
	.fleet-create { display:grid; gap:.8rem; }.fleet-form { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.8rem; }.fleet-form label { display:grid; gap:.35rem; color:#38535a; font-size:.78rem; font-weight:700; }.fleet-form input,.fleet-form select { width:100%; padding:.55rem .6rem; border:1px solid #cddad7; border-radius:6px; background:#fff; }.fleet-form small { color:var(--muted); font-weight:500; }.run-form { grid-template-columns:repeat(4,minmax(0,1fr)); align-items:end; }.enqueue-action { display:grid; gap:.35rem; align-self:end; padding-top:.1rem; }.enqueue-action small { line-height:1.3; }
	.fleet-stats { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.75rem; margin-bottom:1rem; }.fleet-stats article { padding:.9rem 1rem; border:1px solid var(--line); border-radius:10px; background:var(--surface); box-shadow:var(--shadow); }.fleet-stats span,.fleet-table small { display:block; color:var(--muted); font-size:.68rem; }.fleet-stats strong { display:block; margin-top:.2rem; color:var(--navy); font-size:1.45rem; }.fleet-table { width:100%; min-width:780px; border-collapse:collapse; }.fleet-table th,.fleet-table td { padding:.65rem .7rem; border-bottom:1px solid var(--line); color:#405a61; font-size:.76rem; text-align:left; vertical-align:top; }.fleet-table th { color:#6a8085; font-size:.65rem; letter-spacing:.06em; text-transform:uppercase; }.fleet-table td code { display:block; max-width:17rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:.67rem; }.fleet-table .code { font-family:'SFMono-Regular',Consolas,monospace; }
	@media (max-width:850px) { .fleet-intro { flex-direction:column; }.fleet-form,.fleet-stats { grid-template-columns:1fr 1fr; } }
	@media (max-width:520px) { .fleet-form,.fleet-stats { grid-template-columns:1fr; } }
</style>
