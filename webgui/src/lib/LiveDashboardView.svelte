<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { api, type RecordValue } from '$lib/api';
	import { subscribeOperationalRefresh } from '$lib/operational-refresh';
	import MetricPlot from '$lib/MetricPlot.svelte';

	export let onfailure: (message: string) => void = () => {};
	let jobs: RecordValue[] = [];
	let workers: RecordValue[] = [];
	let leases: RecordValue[] = [];
	let queuedRuns: RecordValue[] = [];
	let events: Record<string, RecordValue[]> = {};
	let timings: Record<string, RecordValue> = {};
	let uploads: Record<string, RecordValue> = {};
	let commands: Record<string, RecordValue[]> = {};
	let heartbeats: Record<string, RecordValue> = {};
	let executionRuns: Record<string, RecordValue> = {};
	let commandAction = '';
	let commandMessage = '';
	let loading = false;
	let initialized = false;
	let refreshError = '';
	let detailError = '';
	let updatedAt = '';
	let paused = false;
	let disposed = false;
	let selectedJobId = '';
	let traceMode: 'batch' | 'epoch' = 'batch';
	let runFilter: 'active' | 'all' = 'active';
	let dialog: HTMLDialogElement;
	let detail: { kind: string; title: string; metric?: string; data?: RecordValue } | null = null;
	let returnFocus: HTMLElement | null = null;

	const records = (value: unknown): RecordValue[] => Array.isArray(value) ? value as RecordValue[] : [];
	const object = (value: unknown): RecordValue => value && typeof value === 'object' && !Array.isArray(value) ? value as RecordValue : {};
	const number = (value: unknown): number | null => typeof value === 'number' && Number.isFinite(value) ? value : null;
	const text = (value: unknown, fallback = '—') => value === undefined || value === null || value === '' ? fallback : String(value);
	const label = (value: string) => value.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
	const format = (value: unknown) => number(value)?.toLocaleString(undefined, { maximumSignificantDigits: 5 }) ?? '—';
	const date = (value: unknown) => value && Number.isFinite(Date.parse(String(value))) ? new Date(String(value)).toLocaleString() : 'Not yet';
	const duration = (value: unknown) => { const seconds = number(value); return seconds === null ? '—' : seconds < 60 ? `${seconds.toFixed(1)}s` : `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`; };
	const bytes = (value: unknown) => { const n = number(value); return n === null ? '—' : n < 1048576 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1048576).toFixed(1)} MB`; };
	const activeStatus = new Set(['queued', 'leased', 'acknowledged', 'running', 'uploading', 'cancel_requested']);
	const visibleStatus = new Set([...activeStatus, 'paused']);
	const isVerification = (job: RecordValue) => object(object(object(job.work_unit).parameters).queue_execution).phase === 'verify';
	const logicalRunId = (job: RecordValue | null | undefined) => {
		const value = object(job?.work_unit).run_id;
		return typeof value === 'string' ? value : '';
	};
	const jobName = (job: RecordValue) => text(queuedRuns.find((run) => run.queued_run_id === job.queued_run_id)?.name, text(job.artifact_name, `Run ${String(job.job_id).slice(0, 8)}`));
	const workerName = (id: unknown) => text(workers.find((worker) => worker.worker_id === id)?.name, text(id, 'Waiting for a worker'));
	const latest = (job: RecordValue) => events[String(job.job_id)]?.at(-1);
	const metricInfo: Record<string, string> = {
		loss: 'The objective being minimized. Lower values generally indicate a better fit; compare validation loss to check generalization.',
		accuracy: 'Fraction of correct predictions. Accuracy alone can hide poor performance on rare classes.',
		macro_f1: 'F1 averaged equally across classes, combining precision and recall. Useful when class frequencies differ.',
		learning_rate: 'The optimizer step size reported by the worker. Scheduling changes can affect the learning curve.'
	};
	$: activeJobs = jobs.filter((job) => activeStatus.has(String(job.status)));
	$: pausedJobs = jobs.filter((job) => String(job.status) === 'paused');
	$: trainingJobs = activeJobs.filter((job) => job.action === 'train' && !isVerification(job));
	$: verificationJobs = activeJobs.filter(isVerification);
	$: pendingRuns = queuedRuns.filter((run) => ['pending_verification', 'needs_attention'].includes(String(run.status)));
	$: readyRuns = queuedRuns.filter((run) => run.status === 'ready');
	$: waitingJobs = activeJobs.filter((job) => job.status === 'queued');
	$: busyWorkers = workers.filter((worker) => worker.state === 'leased');
	$: unavailableWorkers = workers.filter((worker) => ['offline', 'unhealthy'].includes(String(worker.state)));
	$: activeLeases = leases.filter((lease) => ['active', 'acknowledged', 'uploading'].includes(String(lease.status)));
	$: focusedJob = jobs.find((job) => String(job.job_id) === selectedJobId) ?? trainingJobs.find((job) => job.status === 'running') ?? activeJobs[0] ?? pausedJobs[0] ?? jobs[0] ?? null;
	$: focusedRunId = logicalRunId(focusedJob);
	$: focusedExecutionRun = focusedRunId ? executionRuns[focusedRunId] ?? {} : {};
	$: focusedLease = focusedJob ? leases.find((lease) => String(lease.job_id) === String(focusedJob?.job_id)) ?? {} : {};
	$: committedCheckpoint = object(focusedExecutionRun.checkpoint_ref);
	$: focusedEvents = focusedJob ? events[String(focusedJob.job_id)] ?? [] : [];
	$: focusedCommands = focusedJob ? commands[String(focusedJob.job_id)] ?? [] : [];
	$: latestCommand = focusedCommands.at(-1);
	$: focusedHeartbeat = focusedJob?.worker_id ? heartbeats[String(focusedJob.worker_id)] ?? {} : {};
	$: snapshot = object(focusedEvents.filter((event) => event.event_type === 'training_progress').at(-1)?.data);
	$: liveMetrics = object(snapshot.metrics);
	$: completedMetrics = object(snapshot.completed_metrics);
	$: trace = records(traceMode === 'batch' ? snapshot.trace : snapshot.history);
	$: chartKeys = ['loss', 'val_loss', 'accuracy', 'val_accuracy', 'macro_f1', 'val_macro_f1'].filter((key) => trace.some((point) => number(point[key]) !== null));
	$: progress = object(snapshot.progress);
	$: timing = object(snapshot.timing);
	$: focusedTiming = focusedJob ? timings[String(focusedJob.job_id)] ?? {} : {};
	$: summary = object(focusedTiming.summary);
	$: stages = [...records(focusedTiming.stages), ...records(focusedTiming.derived_stages)];
	$: totalEpochs = number(progress.total_epochs);
	$: epoch = number(progress.epoch);
	$: progressPercent = totalEpochs && epoch !== null ? Math.min(100, Math.max(0, (epoch - 1 + ((number(progress.completed_batches) ?? 0) / (number(progress.total_batches) || 1))) / totalEpochs * 100)) : null;
	$: displayedJobs = runFilter === 'active' ? jobs.filter((job) => visibleStatus.has(String(job.status))) : jobs.slice(0, 30);
	$: publicationJobs = jobs.filter((job) => uploads[String(job.job_id)] || events[String(job.job_id)]?.some((event) => String(event.event_type).startsWith('output_'))).slice(0, 6);
	$: stats = [
		{ kind: 'runs', title: 'Training runs', value: trainingJobs.length, note: `${verificationJobs.length} verifying separately`, accent: 'teal' },
		{ kind: 'queue', title: 'Waiting for a worker', value: waitingJobs.length, note: `${readyRuns.length} verified · ready to start`, accent: 'amber' },
		{ kind: 'workers', title: 'Workers in use', value: `${busyWorkers.length} / ${workers.length}`, note: `${activeLeases.length} active leases`, accent: 'blue' },
		{ kind: 'alerts', title: 'Worker alerts', value: unavailableWorkers.length, note: unavailableWorkers.length ? 'Inspect heartbeat details' : 'No worker alerts reported', accent: unavailableWorkers.length ? 'red' : 'teal' }
	];

	async function load(manual = false) {
		if (loading || disposed || (paused && !manual)) return;
		loading = true;
		try {
			const [jobResult, workerResult, leaseResult, queueResult] = await Promise.all([api.jobs(), api.registeredWorkers(), api.workerLeases(), api.queuedRuns()]);
			if (disposed) return;
			jobs = records(jobResult.jobs); workers = records(workerResult.workers); leases = records(leaseResult.leases); queuedRuns = records(queueResult.queued_runs);
			const candidate = jobs.find((job) => String(job.job_id) === selectedJobId)
				?? jobs.find((job) => job.action === 'train' && !isVerification(job) && job.status === 'running')
				?? jobs.find((job) => activeStatus.has(String(job.status)))
				?? jobs.find((job) => String(job.status) === 'paused');
			const candidateRunId = logicalRunId(candidate);
			if (candidateRunId) {
				try {
					const execution = object((await api.executionRun(candidateRunId)).run);
					executionRuns = { ...executionRuns, [candidateRunId]: execution };
					const currentJobId = typeof execution.current_job_id === 'string' ? execution.current_job_id : '';
					// Selecting any unit keeps the spotlight on this logical run as
					// each immutable continuation becomes the current unit.
					if (currentJobId && jobs.some((job) => String(job.job_id) === currentJobId)) selectedJobId = currentJobId;
				} catch { /* V1 jobs have no execution-run record. */ }
			}
			// Every active job and the selected run gets telemetry, even when
			// it is older than the recent-history window.
			const selectedRunId = logicalRunId(jobs.find((job) => String(job.job_id) === selectedJobId));
			const visible = jobs.filter((job, index) => visibleStatus.has(String(job.status)) || String(job.job_id) === selectedJobId || (Boolean(selectedRunId) && logicalRunId(job) === selectedRunId) || index < 8);
			let failures = 0;
			await Promise.all(visible.map(async (job) => {
				const id = String(job.job_id);
				const results = await Promise.allSettled([api.jobEvents(id), api.jobTiming(id), api.jobOutputUpload(id), api.jobCommands(id)]);
				if (disposed) return;
				if (results[0].status === 'fulfilled') events[id] = records(results[0].value.events); else failures++;
				if (results[1].status === 'fulfilled') timings[id] = results[1].value; else failures++;
				if (results[2].status === 'fulfilled') { const upload = results[2].value.upload; if (upload) uploads[id] = upload; else delete uploads[id]; } else failures++;
				if (results[3].status === 'fulfilled') commands[id] = records(results[3].value.commands); else failures++;
			}));
			const activeWorkerIds = [...new Set(visible.map((job) => String(job.worker_id ?? '')).filter(Boolean))];
			await Promise.all(activeWorkerIds.map(async (id) => {
				const result = await api.workerHeartbeat(id).catch(() => null);
				if (result?.heartbeat) heartbeats[id] = result.heartbeat;
			}));
			if (disposed) return;
			events = { ...events }; timings = { ...timings }; uploads = { ...uploads }; commands = { ...commands }; heartbeats = { ...heartbeats }; executionRuns = { ...executionRuns };
			detailError = failures ? `${failures} detail requests failed. Some telemetry may be stale; the run and fleet summary is current.` : '';
			updatedAt = new Date().toLocaleTimeString(); initialized = true; refreshError = '';
		} catch (error) { refreshError = error instanceof Error ? error.message : 'Could not refresh execution state.'; if (manual) onfailure(refreshError); }
		finally { loading = false; }
	}
	async function selectJob(id: string) { selectedJobId = id; traceMode = 'batch'; await load(true); }
	function metrics(source: RecordValue) { const order = ['loss', 'accuracy', 'macro_f1', 'learning_rate']; return Object.entries(source).filter((entry): entry is [string, number] => number(entry[1]) !== null).sort(([a], [b]) => (order.includes(a) ? order.indexOf(a) : 99) - (order.includes(b) ? order.indexOf(b) : 99)).slice(0, 8); }
	async function openDetail(kind: string, title: string, metric?: string, data?: RecordValue) {
		returnFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
		if (focusedJob && ['metric', 'timing'].includes(kind)) selectedJobId = String(focusedJob.job_id);
		detail = { kind, title, metric, data }; await tick(); dialog.showModal();
	}
	function closeDetail() { dialog.close(); detail = null; returnFocus?.focus(); }
	function commandLabel(command: RecordValue | undefined) {
		if (!command) return 'No control request pending';
		return `${label(text(command.action))} · ${label(text(command.status, 'received'))}`;
	}
	function canCommand(action: string, job: RecordValue) {
		const segment = object(object(job.work_unit).parameters).execution_segment;
		const phase = String(object(job.work_unit).phase ?? '');
		if (action === 'pause' || action === 'yield') return Boolean(segment) && phase !== 'finalize' && ['running', 'leased', 'acknowledged'].includes(String(job.status));
		if (action === 'resume') return String(job.status) === 'paused';
		return ['queued', 'paused', 'leased', 'acknowledged', 'running', 'uploading', 'cancel_requested'].includes(String(job.status));
	}
	async function issueCommand(action: 'pause' | 'yield' | 'stop_now' | 'restart' | 'resume') {
		if (!focusedJob || commandAction) return;
		commandAction = action; commandMessage = '';
		try {
			const result = await api.issueJobCommand(String(focusedJob.job_id), { action, command_id: crypto.randomUUID() });
			const command = result.command;
			commands = { ...commands, [String(focusedJob.job_id)]: [...focusedCommands, command] };
			commandMessage = command.status === 'applied' ? `${label(action)} applied.` : `${label(action)} requested. Waiting for worker acknowledgement.`;
			await load(true);
		} catch (error) { commandMessage = error instanceof Error ? error.message : 'Could not request worker control.'; }
		finally { commandAction = ''; }
	}
	function publicationState(job: RecordValue) {
		if (['indexed', 'completed'].includes(String(job.status))) return 'Published';
		const history = events[String(job.job_id)] ?? [];
		if (history.filter((event) => ['output_publication_deferred', 'output_published'].includes(String(event.event_type))).at(-1)?.event_type === 'output_publication_deferred') return 'Recovery needed';
		return uploads[String(job.job_id)]?.status === 'finalized' ? 'Staged · sealing' : 'Transferring';
	}
	onMount(() => {
		void load();
		const unsubscribe = subscribeOperationalRefresh(() => { void load(); });
		return () => { disposed = true; unsubscribe(); };
	});
</script>

<div class="live-dashboard">
	<header class="dashboard-header">
		<div><p class="eyebrow">EXECUTION OVERVIEW</p><h2>Your training, in focus.</h2><p>Follow progress, inspect the details, and keep an eye on capacity.</p></div>
		<div class="refresh-controls"><span class="live-state" class:paused class:issue={Boolean(refreshError)}><i></i>{refreshError ? 'Refresh interrupted' : paused ? 'Updates paused' : 'Live updates'}</span><small>{updatedAt ? `Updated ${updatedAt}` : 'Connecting…'}</small><div><button class="secondary small" on:click={() => { paused = !paused; if (!paused) void load(); }}>{paused ? 'Resume' : 'Pause'}</button><button class="secondary small" disabled={loading} on:click={() => load(true)}>{loading ? 'Refreshing…' : 'Refresh'}</button></div></div>
	</header>
	{#if refreshError}<div class="refresh-warning" role="alert"><strong>Showing the last available snapshot.</strong> {refreshError}</div>{/if}
	{#if detailError}<div class="refresh-warning" role="status">{detailError}</div>{/if}
	<section class="dashboard-stats" aria-label="Execution summary">
		{#each stats as stat}<button class="stat-card {stat.accent}" on:click={() => openDetail(stat.kind, stat.title)} aria-label={`${stat.title}: ${stat.value}. View details`}><span>{stat.title}<b>↗</b></span><strong>{initialized ? stat.value : '—'}</strong><small>{stat.note}</small></button>{/each}
	</section>

	<div class="main-grid">
		<section class="panel focus-panel">
			<div class="section-heading"><div><p class="eyebrow">RUN SPOTLIGHT</p><h2>{focusedJob ? jobName(focusedJob) : 'Ready for your next experiment'}</h2></div>{#if focusedJob}<span class="status {text(focusedJob.status)}">{isVerification(focusedJob) ? 'Verification' : label(text(focusedJob.status))}</span>{/if}</div>
			{#if jobs.length}<label class="run-picker">Inspect a run<select value={focusedJob?.job_id} on:change={(event) => selectJob(event.currentTarget.value)}>{#each jobs as job}<option value={String(job.job_id)}>{jobName(job)} · {isVerification(job) ? 'verification' : text(job.status)} · {String(job.job_id).slice(0, 8)}</option>{/each}</select></label>{/if}
			{#if focusedJob}
				<p class="phase-message">{text(snapshot.message, text(latest(focusedJob)?.message, 'Waiting for the worker to report progress.'))}</p>
				<div class="run-context"><span>Worker <strong>{workerName(focusedJob.worker_id)}</strong></span><span>Telemetry <strong>{snapshot.updated_at ? date(snapshot.updated_at) : 'Not yet available'}</strong></span><span>Heartbeat <strong>{focusedHeartbeat.received_at ? date(focusedHeartbeat.received_at) : 'Not reported'}</strong></span></div>
				<div class="unit-context"><span>Run <strong>{text(focusedRunId)}</strong></span><span>Unit <strong>{text(object(focusedJob.work_unit).sequence)} · {label(text(object(focusedJob.work_unit).phase))}</strong></span><span>Attempt <strong>{text(focusedLease.generation ?? focusedLease.attempt_id, 'Not leased')}</strong></span><span>Committed cursor <strong>{text(focusedExecutionRun.completed_epoch, 'Not committed')}</strong></span><span>Checkpoint <strong>{text(committedCheckpoint.artifact_id, 'None committed')}</strong></span></div>
				<section class="command-strip" aria-label="Worker controls"><div><strong>Worker control</strong><small>{commandLabel(latestCommand)}</small></div><div class="command-actions"><button class="secondary small" disabled={!canCommand('pause', focusedJob) || Boolean(commandAction)} on:click={() => issueCommand('pause')}>Pause safely</button><button class="secondary small" disabled={!canCommand('yield', focusedJob) || Boolean(commandAction)} on:click={() => issueCommand('yield')}>Yield</button><button class="secondary small" disabled={!canCommand('restart', focusedJob) || Boolean(commandAction)} on:click={() => issueCommand('restart')}>Restart unit</button><button class="secondary small" disabled={!canCommand('resume', focusedJob) || Boolean(commandAction)} on:click={() => issueCommand('resume')}>Resume</button><button class="danger small" disabled={!canCommand('stop_now', focusedJob) || Boolean(commandAction)} on:click={() => issueCommand('stop_now')}>Stop now</button></div>{#if commandMessage}<p role="status">{commandMessage}</p>{/if}</section>
				{#if Object.keys(snapshot).length}
					<div class="progress-heading"><strong>Epoch {format(progress.epoch)} <span>/ {format(progress.total_epochs)}</span></strong><span>Batch {format(progress.completed_batches ?? progress.batch)} / {format(progress.total_batches)}</span></div>
					{#if progressPercent !== null}<progress max="100" value={progressPercent} aria-label="Training progress"></progress>{/if}
					<div class="metric-cards">{#each metrics(liveMetrics).slice(0, 4) as [key, value]}<button on:click={() => openDetail('metric', label(key), key)} title={metricInfo[key] ?? 'Click for metric details and sampled values.'}><span>{label(key)} <i>↗</i></span><strong>{format(value)}</strong><small>Current batch</small></button>{/each}</div>
					<div class="chart-heading"><h3>Learning curves</h3><div class="segmented" aria-label="Chart interval"><button class:chosen={traceMode === 'batch'} aria-pressed={traceMode === 'batch'} on:click={() => traceMode = 'batch'}>Batches</button><button class:chosen={traceMode === 'epoch'} aria-pressed={traceMode === 'epoch'} on:click={() => traceMode = 'epoch'}>Epochs</button></div></div>
					{#if chartKeys.length}<div class="chart-grid">{#each chartKeys as key}<button class="chart-card" on:click={() => openDetail('metric', label(key), key)} aria-label={`Expand ${label(key)} chart`}><header><strong>{label(key)}</strong><span>Explore ↗</span></header><MetricPlot points={trace} metric={key} title={label(key)} axis={traceMode === 'batch' ? 'Batch' : 'Epoch'} /></button>{/each}</div><p class="chart-note">Bounded worker samples · each chart has its own scale · click to explore values</p>{:else}<div class="quiet-empty">{traceMode === 'epoch' ? 'Completed epoch metrics will appear after the first epoch.' : 'Waiting for sampled batch metrics.'}</div>{/if}
				{:else}<div class="quiet-empty"><span class="empty-symbol">{isVerification(focusedJob) ? '✓' : '◌'}</span><h3>{isVerification(focusedJob) ? 'Preflight before training' : 'Waiting for training telemetry'}</h3><p>{isVerification(focusedJob) ? 'The worker checks the dataset, builds the model, and probes batch capacity. This job does not train or save a model.' : 'Metrics appear when the worker reports its first training batch.'}</p></div>{/if}
				<button class="timing-strip" on:click={() => openDetail('timing', 'Where the time goes')}><span>Elapsed<strong>{duration(timing.elapsed_seconds ?? summary.runtime_seconds)}</strong></span><span>Throughput<strong>{format(timing.batches_per_second)} <small>batches/s</small></strong></span><span>Queue wait<strong>{duration(summary.queue_seconds)}</strong></span><b>Timing details ↗</b></button>
			{:else}<div class="quiet-empty"><span class="empty-symbol">◈</span><p>Add a run to the queue, verify it, then start training. Progress will appear here.</p><a href="#queue">Open training queue →</a></div>{/if}
		</section>

		<aside class="dashboard-aside">
			<section class="panel capacity-panel"><div class="section-heading"><div><p class="eyebrow">CAPACITY</p><h2>Worker fleet</h2></div><button class="text-button" on:click={() => openDetail('workers', 'Worker fleet')}>Details ↗</button></div><div class="capacity-count"><strong>{workers.filter((worker) => worker.state === 'idle').length}</strong><span>idle workers<br /><small>{busyWorkers.length} busy · {unavailableWorkers.length} unavailable</small></span></div><div class="capacity-bar" aria-label={`${busyWorkers.length} of ${workers.length} workers busy`}>{#each workers as worker}<span class:busy={worker.state === 'leased'} class:unavailable={['offline', 'unhealthy'].includes(String(worker.state))} title={`${text(worker.name)}: ${text(worker.state)}`}></span>{/each}</div><div class="fleet-list">{#each workers.slice(0, 5) as worker}<button on:click={() => openDetail('worker', text(worker.name), undefined, worker)}><span class="worker-dot {text(worker.state)}"></span><span><strong>{text(worker.name)}</strong><small>{label(text(worker.state))}</small></span><b>↗</b></button>{/each}{#if !workers.length}<p class="empty-copy">No workers registered yet.</p><a href="#workers">Set up workers →</a>{/if}</div></section>
			<section class="panel queue-panel"><p class="eyebrow">NEXT UP</p><h2>Queue at a glance</h2><button on:click={() => openDetail('queue', 'Queue at a glance')}><span>Awaiting verification</span><strong>{pendingRuns.length}</strong></button><button on:click={() => openDetail('queue', 'Queue at a glance')}><span>Verifying now / waiting</span><strong>{verificationJobs.length}</strong></button><button on:click={() => openDetail('queue', 'Queue at a glance')}><span>Verified, ready to start</span><strong>{readyRuns.length}</strong></button><p>Verified runs wait for your start command unless you selected “verify and start.”</p><a href="#queue">Manage queue →</a></section>
		</aside>
	</div>

	<section class="panel runs-panel"><div class="section-heading"><div><p class="eyebrow">ACTIVITY</p><h2>Runs and verification</h2></div><div class="segmented"><button class:chosen={runFilter === 'active'} aria-pressed={runFilter === 'active'} on:click={() => runFilter = 'active'}>Active & paused ({activeJobs.length + pausedJobs.length})</button><button class:chosen={runFilter === 'all'} aria-pressed={runFilter === 'all'} on:click={() => runFilter = 'all'}>Recent</button></div></div>{#if displayedJobs.length}<div class="table-scroll"><table><thead><tr><th>Run</th><th>Stage / status</th><th>Worker</th><th>Latest update</th><th></th></tr></thead><tbody>{#each displayedJobs as job}<tr class:selected={job.job_id === focusedJob?.job_id}><td><strong>{jobName(job)}</strong><small>{String(job.job_id).slice(0, 8)}</small></td><td><span class="status {text(job.status)}">{label(text(job.status))}</span><small>{isVerification(job) ? 'Verification only' : label(text(job.action, 'Training'))}</small></td><td>{workerName(job.worker_id)}</td><td>{text(latest(job)?.message, text(job.error, 'Awaiting an event'))}<small>{date(latest(job)?.timestamp ?? job.updated_at)}</small></td><td><button class="secondary small" on:click={() => { void selectJob(String(job.job_id)); }}>Inspect</button></td></tr>{/each}</tbody></table></div>{:else}<div class="quiet-empty compact"><h3>{initialized ? 'No active work right now' : 'Loading execution state…'}</h3><p>{initialized ? 'Start a verified run from the queue, or switch to Recent to inspect previous work.' : 'Connecting to the Orchestrator.'}</p></div>{/if}</section>

	{#if publicationJobs.length}<section class="panel"><div class="section-heading"><div><p class="eyebrow">OUTPUTS</p><h2>Artifact transfers</h2><p>Track the handoff from worker output to a published artifact.</p></div></div><div class="transfer-grid">{#each publicationJobs as job}{@const upload = uploads[String(job.job_id)] ?? {}}<button class="transfer-card" on:click={() => openDetail('publication', jobName(job), undefined, { ...upload, job_id: job.job_id, publication_state: publicationState(job) })}><header><strong>{jobName(job)}</strong><span>{publicationState(job)}</span></header>{#if number(upload.archive_size)}<progress max={Number(upload.archive_size)} value={Number(upload.uploaded_bytes ?? 0)} aria-label="Output upload progress"></progress>{/if}<small>{bytes(upload.uploaded_bytes)} / {bytes(upload.archive_size)} <b>Inspect ↗</b></small></button>{/each}</div></section>{/if}
</div>

<dialog bind:this={dialog} on:cancel={(event) => { event.preventDefault(); closeDetail(); }} aria-labelledby="dashboard-detail-title">
	{#if detail}<header class="dialog-header"><div><p class="eyebrow">DASHBOARD DETAILS</p><h2 id="dashboard-detail-title">{detail.title}</h2></div><button class="secondary small" on:click={closeDetail} aria-label="Close details">Close ×</button></header><div class="dialog-body">
		{#if detail.kind === 'metric' && detail.metric}
			<p>{metricInfo[detail.metric.replace('val_', '')] ?? 'Numeric telemetry reported by the worker for this run.'} {detail.metric.startsWith('val_') ? 'This value is measured on the validation split.' : ''}</p><div class="detail-metrics"><span>Current batch<strong>{format(liveMetrics[detail.metric])}</strong></span><span>Last completed epoch<strong>{format(completedMetrics[detail.metric])}</strong></span></div><MetricPlot points={trace} metric={detail.metric} title={detail.title} axis={traceMode === 'batch' ? 'Batch' : 'Epoch'} expanded /><p class="chart-note">{traceMode === 'batch' ? 'Current-epoch batch trace (up to 60 samples).' : 'Completed epoch history (up to 50 epochs).'} Missing values are not interpolated.</p><details><summary>Inspect sampled values</summary><table><thead><tr><th>{traceMode === 'batch' ? 'Batch' : 'Epoch'}</th><th>{detail.title}</th></tr></thead><tbody>{#each trace as point, index}<tr><td>{text(point[traceMode] ?? index + 1)}</td><td>{format(point[detail.metric])}</td></tr>{/each}</tbody></table></details>
		{:else if detail.kind === 'timing'}<p>Measured worker stages can overlap. Derived intervals use durable timestamps; they are not an additive breakdown.</p><div class="detail-metrics"><span>Queue<strong>{duration(summary.queue_seconds)}</strong></span><span>Runtime<strong>{duration(summary.runtime_seconds)}</strong></span><span>Training<strong>{duration(summary.training_elapsed_seconds)}</strong></span></div><table><thead><tr><th>Stage</th><th>Duration</th></tr></thead><tbody>{#each stages as stage}<tr><td>{label(text(stage.stage))}</td><td>{duration(stage.duration_seconds)}</td></tr>{/each}</tbody></table>{#if !stages.length}<p>No stage timings reported yet.</p>{/if}
		{:else if ['workers', 'alerts', 'worker'].includes(detail.kind)}<p>Worker states and heartbeat timestamps are reported by the control plane.</p>{#each detail.kind === 'worker' && detail.data ? [detail.data] : detail.kind === 'alerts' ? unavailableWorkers : workers as worker}{@const heartbeat = heartbeats[String(worker.worker_id)] ?? {}}<article class="detail-record"><h3>{text(worker.name)} <span class="status">{text(worker.state)}</span></h3><dl><dt>Last heartbeat</dt><dd>{date(heartbeat.received_at ?? worker.last_seen_at)}</dd><dt>Worker ID</dt><dd>{text(worker.worker_id)}</dd><dt>Generation</dt><dd>{text(heartbeat.generation)}</dd><dt>GPUs</dt><dd>{Array.isArray(object(worker.capabilities).gpu_ids) ? (object(worker.capabilities).gpu_ids as unknown[]).length : 0}</dd><dt>Actions</dt><dd>{text(object(worker.capabilities).actions)}</dd><dt>Verification</dt><dd>{object(worker.capabilities).training_verification_v1 ? 'Supported' : 'Worker update required'}</dd></dl></article>{/each}{#if detail.kind === 'alerts' && !unavailableWorkers.length}<p>No offline or unhealthy workers reported.</p>{/if}
		{:else if detail.kind === 'queue'}<p>Only verified and explicitly started training runs can execute. Verification itself may wait for a compatible worker.</p>{#each queuedRuns.filter((run) => !['indexed', 'complete', 'cancelled', 'archived'].includes(String(run.status))) as run}<article class="detail-record"><h3>{text(run.name)}</h3><span class="status">{label(text(run.status))}</span><p>{text(run.failure_reason, run.status === 'ready' ? 'Verified. Waiting for your start command.' : run.status === 'verifying' ? 'Preflight is queued or running on a worker.' : 'Open the queue to review this run.')}</p></article>{/each}<a href="#queue" on:click={closeDetail}>Open training queue →</a>
		{:else if detail.kind === 'runs'}{#each activeJobs as job}<article class="detail-record"><h3>{jobName(job)}</h3><p>{isVerification(job) ? 'Verification' : 'Execution'} · {label(text(job.status))} · {workerName(job.worker_id)}</p><p>{text(latest(job)?.message, 'Waiting for an event')}</p><button class="secondary small" on:click={() => { void selectJob(String(job.job_id)); closeDetail(); }}>Inspect this run</button></article>{/each}{#if !activeJobs.length}<p>No active runs or verification jobs.</p>{/if}
		{:else if detail.kind === 'publication'}<p>Uploads are staged before the Orchestrator validates and publishes them. Interrupted transfers may retain recovery data on the worker.</p><dl>{#each Object.entries(detail.data ?? {}).filter(([key]) => !['uploaded_parts'].includes(key)) as [key, value]}<dt>{label(key)}</dt><dd>{typeof value === 'object' ? JSON.stringify(value) : text(value)}</dd>{/each}</dl>{/if}
	</div>{/if}
</dialog>

<style>
	.live-dashboard { --chart-blue:#56799c; }
	.dashboard-header { display:flex; align-items:flex-start; justify-content:space-between; gap:2rem; margin:.2rem 0 1.5rem; }
	.dashboard-header h2 { font-size:1.8rem; font-weight:650; letter-spacing:-.045em; }
	.dashboard-header p { margin-bottom:0; font-size:.85rem; }
	.refresh-controls { display:grid; justify-items:end; gap:.45rem; flex-shrink:0; }
	.refresh-controls > div { display:flex; gap:.4rem; }
	.refresh-controls > small { font-size:.68rem; color:var(--muted); }
	.live-state { display:flex; gap:.45rem; align-items:center; font-size:.73rem; font-weight:700; color:var(--teal-dark); }
	.live-state i { width:7px; height:7px; background:var(--teal); border-radius:50%; box-shadow:0 0 0 4px var(--teal-soft); }
	.live-state.paused,.live-state.issue { color:var(--amber); }.live-state.paused i,.live-state.issue i { background:var(--amber); box-shadow:none; }
	.refresh-warning { padding:.8rem 1rem; margin-bottom:1rem; border:1px solid #edce9f; background:#fffaef; color:#83530f; border-radius:9px; font-size:.8rem; }
	.dashboard-stats { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:1rem; margin-bottom:1.4rem; }
	.stat-card { text-align:left; background:var(--surface); border:1px solid var(--line); color:var(--ink); padding:1.1rem 1.2rem; box-shadow:var(--shadow); border-radius:12px; }
	.stat-card:hover:not(:disabled) { background:#fafffe; border-color:#91bfb7; transform:translateY(-2px); }
	.stat-card > span { display:flex; justify-content:space-between; gap:.5rem; color:var(--muted); font-size:.75rem; font-weight:600; }
	.stat-card b { color:#91a7aa; font-size:.85rem; }.stat-card > strong { display:block; font-size:2.2rem; font-weight:600; letter-spacing:-.06em; margin:.7rem 0 .25rem; font-variant-numeric:tabular-nums; }
	.stat-card small { color:var(--muted); font-size:.68rem; font-weight:500; }.stat-card.teal > strong { color:var(--teal); }.stat-card.blue > strong { color:var(--chart-blue); }.stat-card.amber > strong { color:var(--amber); }.stat-card.red > strong { color:var(--red); }
	.main-grid { display:grid; grid-template-columns:minmax(0,1fr) 290px; gap:1.2rem; align-items:start; }
	.panel { padding:1.25rem; border-radius:12px; }.section-heading { display:flex; justify-content:space-between; gap:1rem; align-items:flex-start; margin-bottom:1rem; }.section-heading p:not(.eyebrow) { margin:0; font-size:.78rem; }.section-heading h2 { overflow-wrap:anywhere; }
	.run-picker { display:grid; grid-template-columns:auto minmax(0,1fr); align-items:center; gap:1rem; font-size:.74rem; color:var(--muted); }.run-picker select { width:100%; min-width:0; border:1px solid var(--line); background:var(--surface-soft); padding:.6rem; border-radius:7px; color:var(--ink); font-size:.77rem; }
	.phase-message { margin:1rem 0 .5rem; font-size:.83rem; }.run-context { display:flex; flex-wrap:wrap; gap:.4rem 1.2rem; font-size:.68rem; color:var(--muted); margin-bottom:1.3rem; }.run-context strong { color:#49646b; margin-left:.3rem; font-weight:500; }
	.unit-context { display:flex; flex-wrap:wrap; gap:.45rem .9rem; padding:.65rem .75rem; margin:-.65rem 0 1rem; border-left:3px solid #a7d0c5; background:var(--surface-soft); color:var(--muted); font-size:.65rem; }.unit-context strong { color:var(--ink); font-weight:600; overflow-wrap:anywhere; }
	.command-strip { margin:0 0 1.1rem; padding:.75rem; border:1px solid var(--line); border-radius:8px; background:#f8fbfa; }.command-strip > div:first-child { display:grid; gap:.15rem; margin-bottom:.55rem; }.command-strip small,.command-strip p { color:var(--muted); font-size:.68rem; margin:0; }.command-actions { display:flex; flex-wrap:wrap; gap:.4rem; }.command-actions .danger { color:#8b3026; border-color:#d9aaa4; }.command-actions .danger:hover:not(:disabled) { background:#fff1ef; }.command-strip p { margin-top:.55rem; color:#295f52; }
	.progress-heading { display:flex; justify-content:space-between; gap:.5rem; align-items:center; font-size:.73rem; margin-bottom:.5rem; }.progress-heading strong { font-size:.9rem; }.progress-heading span { color:var(--muted); font-weight:500; }
	progress { display:block; appearance:none; border:0; width:100%; height:6px; border-radius:8px; overflow:hidden; background:#e6efed; }progress::-webkit-progress-bar { background:#e6efed; }progress::-webkit-progress-value { background:var(--teal); border-radius:8px; }progress::-moz-progress-bar { background:var(--teal); border-radius:8px; }
	.metric-cards { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.65rem; margin:1.2rem 0 1.5rem; }.metric-cards button { text-align:left; padding:.8rem; border:1px solid var(--line); background:var(--surface-soft); color:var(--ink); }.metric-cards button:hover { background:var(--teal-soft); }.metric-cards span { display:flex; justify-content:space-between; font-size:.68rem; color:var(--muted); }.metric-cards i { font-style:normal; }.metric-cards strong { display:block; font-size:1.3rem; letter-spacing:-.04em; margin:.5rem 0 .15rem; font-variant-numeric:tabular-nums; }.metric-cards small { color:var(--muted); font-size:.63rem; font-weight:500; }
	.chart-heading { display:flex; justify-content:space-between; align-items:center; margin-bottom:.75rem; }.chart-heading h3 { font-size:.88rem; margin:0; }.segmented { display:flex; gap:2px; background:#edf2f1; padding:3px; border-radius:7px; }.segmented button { background:transparent; color:var(--muted); font-size:.68rem; padding:.38rem .65rem; border-radius:5px; }.segmented button.chosen { background:white; color:var(--ink); box-shadow:0 1px 3px #102a3518; }
	.chart-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:.8rem; }.chart-card { min-width:0; text-align:left; background:white; border:1px solid var(--line); color:var(--ink); border-radius:9px; padding:.8rem; }.chart-card:hover { background:#fafffe; border-color:#9bc9c0; }.chart-card header { display:flex; justify-content:space-between; gap:.5rem; font-size:.74rem; margin-bottom:.3rem; }.chart-card header span { font-size:.65rem; color:var(--teal); font-weight:500; }.chart-note { color:var(--muted); font-size:.66rem; margin:.7rem 0 0; }
	.timing-strip { display:flex; justify-content:space-between; flex-wrap:wrap; gap:.8rem; width:100%; text-align:left; background:#f4f7f7; color:var(--ink); padding:1rem; margin-top:1.2rem; border:1px solid var(--line); }.timing-strip:hover { background:#ecf5f2; }.timing-strip span { font-size:.66rem; color:var(--muted); font-weight:500; }.timing-strip strong { display:block; margin-top:.3rem; color:var(--ink); font-size:.95rem; }.timing-strip strong small { font-size:.65rem; color:var(--muted); }.timing-strip b { align-self:center; color:var(--teal); font-size:.69rem; }
	.quiet-empty { padding:2.5rem 1rem; text-align:center; border:1px dashed var(--line); border-radius:10px; margin-top:1rem; }.quiet-empty p { max-width:31rem; margin:.5rem auto 1rem; font-size:.8rem; }.empty-symbol { display:block; color:var(--teal); font-size:2rem; margin-bottom:.7rem; }.quiet-empty.compact { padding:1.4rem; }.quiet-empty.compact p { margin-bottom:0; }
	a { color:var(--teal); font-size:.77rem; font-weight:700; text-decoration:none; }a:hover { text-decoration:underline; }
	.capacity-count { display:flex; gap:.8rem; align-items:center; }.capacity-count > strong { font-size:2.5rem; font-weight:500; letter-spacing:-.06em; }.capacity-count span { font-size:.8rem; }.capacity-count small { color:var(--muted); font-size:.65rem; }.capacity-bar { display:flex; gap:4px; height:8px; margin:1rem 0; }.capacity-bar span { flex:1; background:#baded5; border-radius:3px; }.capacity-bar span.busy { background:var(--teal); }.capacity-bar span.unavailable { background:#dca5a5; }
	.text-button { padding:.2rem 0; color:var(--teal); background:transparent; font-size:.68rem; }.text-button:hover { background:transparent; text-decoration:underline; }
	.fleet-list button { width:100%; display:flex; gap:.6rem; align-items:center; padding:.7rem 0; text-align:left; color:var(--ink); background:transparent; border-radius:0; border-bottom:1px solid #edf2f1; }.fleet-list button:hover { background:var(--surface-soft); }.fleet-list strong { font-size:.74rem; font-weight:600; overflow-wrap:anywhere; }.fleet-list small { display:block; font-size:.66rem; font-weight:500; color:var(--muted); margin-top:.15rem; }.fleet-list b { margin-left:auto; font-size:.75rem; color:var(--muted); }.worker-dot { width:7px; height:7px; border-radius:50%; background:#adc2c0; flex-shrink:0; }.worker-dot.leased { background:var(--teal); }.worker-dot.idle { background:#73ac95; }.worker-dot.offline,.worker-dot.unhealthy { background:var(--red); }.empty-copy { font-size:.8rem; }
	.queue-panel > button { display:flex; justify-content:space-between; align-items:center; gap:.6rem; width:100%; padding:.7rem 0; background:transparent; border-bottom:1px solid var(--line); border-radius:0; color:var(--ink); font-size:.72rem; font-weight:500; }.queue-panel > button:hover { background:var(--surface-soft); }.queue-panel > button strong { font-size:1rem; }.queue-panel > p:not(.eyebrow) { font-size:.71rem; margin:.9rem 0; }
	table { width:100%; border-collapse:collapse; font-size:.76rem; }th { text-align:left; color:var(--muted); font-size:.65rem; text-transform:uppercase; letter-spacing:.05em; font-weight:600; }th,td { padding:.8rem .65rem; border-bottom:1px solid var(--line); vertical-align:top; }td small { display:block; color:var(--muted); margin-top:.3rem; font-size:.67rem; }td strong { font-weight:600; }tr.selected { background:#f3faf7; }.runs-panel table { min-width:650px; }.runs-panel td:nth-child(4) { max-width:26rem; }
	.transfer-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.8rem; }.transfer-card { text-align:left; color:var(--ink); background:var(--surface-soft); border:1px solid var(--line); padding:.9rem; }.transfer-card:hover { background:var(--teal-soft); }.transfer-card header { display:grid; gap:.3rem; margin-bottom:.8rem; }.transfer-card strong { overflow-wrap:anywhere; font-size:.77rem; }.transfer-card header span { color:var(--teal); font-size:.66rem; font-weight:500; }.transfer-card > small { display:flex; justify-content:space-between; gap:.5rem; font-size:.65rem; font-weight:500; color:var(--muted); margin-top:.7rem; }.transfer-card b { color:var(--teal); }
	dialog { width:min(800px,94vw); max-height:88vh; padding:0; color:var(--ink); background:white; border:1px solid var(--line); border-radius:15px; box-shadow:0 30px 100px #102a3540; }dialog::backdrop { background:#102a3588; backdrop-filter:blur(4px); }.dialog-header { position:sticky; top:0; display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; padding:1.3rem 1.5rem; border-bottom:1px solid var(--line); background:white; z-index:1; }.dialog-header h2 { margin:0; font-size:1.3rem; }.dialog-body { padding:1.5rem; }.dialog-body > p { font-size:.83rem; }.detail-metrics { display:flex; flex-wrap:wrap; gap:1.5rem; padding:1rem; background:var(--surface-soft); border-radius:9px; }.detail-metrics span { font-size:.72rem; color:var(--muted); }.detail-metrics strong { display:block; margin-top:.3rem; color:var(--ink); font-size:1.25rem; }.detail-record { border:1px solid var(--line); border-radius:8px; padding:1rem; margin-bottom:.7rem; }.detail-record h3 { display:flex; justify-content:space-between; gap:1rem; }.detail-record p { font-size:.8rem; margin:.5rem 0; }dl { display:grid; grid-template-columns:140px minmax(0,1fr); gap:.65rem; font-size:.76rem; }dt { color:var(--muted); }dd { margin:0; overflow-wrap:anywhere; }summary { color:var(--teal); cursor:pointer; font-size:.78rem; margin:1rem 0; }
	@media (min-width:1600px) { .main-grid { grid-template-columns:minmax(0,1fr) 330px; }.chart-grid { grid-template-columns:repeat(3,minmax(0,1fr)); } }
	@media (max-width:1150px) { .main-grid { grid-template-columns:minmax(0,1fr); }.dashboard-aside { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }.dashboard-stats { gap:.6rem; }.stat-card { padding:.9rem; }.stat-card > strong { font-size:1.9rem; } }
	@media (max-width:650px) { .dashboard-header { flex-direction:column; gap:1rem; }.refresh-controls { display:flex; flex-wrap:wrap; align-items:center; gap:.6rem; }.dashboard-stats,.metric-cards { grid-template-columns:repeat(2,minmax(0,1fr)); }.dashboard-aside,.chart-grid,.transfer-grid { grid-template-columns:1fr; }.run-picker { grid-template-columns:1fr; gap:.4rem; }.panel { padding:1rem; }.dialog-body,.dialog-header { padding:1rem; }dl { grid-template-columns:1fr; gap:.3rem; }dd { margin-bottom:.4rem; } }
</style>
