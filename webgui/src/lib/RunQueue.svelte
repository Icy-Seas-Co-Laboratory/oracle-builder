<script lang="ts">
	import { api, type RecordValue } from '$lib/api';
	import EmptyState from '$lib/EmptyState.svelte';

	export let specifications: RecordValue[] = [];
	export let jobs: RecordValue[] = [];
	export let computeEndpoints: RecordValue[] = [];
	export let onchanged: (message: string) => void | Promise<void>;
	export let onfailure: (message: string) => void;
	export let oncreateexperiment: () => void;
	export let onviewartifact: (job: RecordValue) => void;

	let selectedEndpointId = '';
	let dispatchingId = '';
	let selectedJobId = '';
	let jobEvents: RecordValue[] = [];
	let observedJobs = jobs;

	$: if (!computeEndpoints.some((endpoint) => endpoint.endpoint_id === selectedEndpointId)) {
		selectedEndpointId = String(computeEndpoints.find((endpoint) => endpoint.status === 'ready')?.endpoint_id ?? computeEndpoints[0]?.endpoint_id ?? '');
	}
	$: if (jobs !== observedJobs) {
		observedJobs = jobs;
		if (selectedJobId) void reloadEvents();
	}

	const display = (value: unknown) => value === null || value === undefined || value === '' ? '—' : String(value);
	const isActiveJob = (job: RecordValue) => ['queued', 'running', 'submitted', 'dispatching', 'validating'].includes(String(job.status));
	const selectedEndpoint = () => computeEndpoints.find((endpoint) => endpoint.endpoint_id === selectedEndpointId);
	const endpointWorkers = (endpoint: RecordValue | undefined) => Array.isArray(endpoint?.workers) ? endpoint.workers as RecordValue[] : [];
	const endpointQueue = (endpoint: RecordValue | undefined) => endpoint?.queue && typeof endpoint.queue === 'object' ? endpoint.queue as RecordValue : null;
	const workerGpuCount = (worker: RecordValue) => {
		const capabilities = worker.capabilities && typeof worker.capabilities === 'object' ? worker.capabilities as RecordValue : null;
		return Array.isArray(capabilities?.gpus) ? capabilities.gpus.length : 0;
	};
	const jobStage = (job: RecordValue) => ({ dispatching: 'Contacting worker', submitted: 'Waiting', queued: 'Waiting', running: 'Running', validating: 'Validating artifact', indexed: 'Indexed', artifact_invalid: 'Invalid artifact', dispatch_failed: 'Dispatch failed', failed: 'Compute failed', cancelled: 'Cancelled', succeeded: 'Complete' }[String(job.status)] ?? String(job.status));
	const specificationName = (job: RecordValue) => specifications.find((item) => item.specification_id === job.specification_id)?.name ?? job.job_id;
	const selectedJob = () => jobs.find((job) => job.job_id === selectedJobId);
	const validationReport = (job: RecordValue | undefined) => job?.validation_report && typeof job.validation_report === 'object' ? job.validation_report as RecordValue : null;
	const reportMessages = (job: RecordValue | undefined, key: 'errors' | 'warnings') => { const value = validationReport(job)?.[key]; return Array.isArray(value) ? value.map(String) : []; };

	async function dispatch(specification: RecordValue) {
		if (!selectedEndpointId) { onfailure('No compute endpoint is configured.'); return; }
		dispatchingId = String(specification.specification_id);
		try {
			const check = await api.preflight(dispatchingId, selectedEndpointId);
			if (!check.ready) throw new Error(`Run cannot be dispatched: ${Array.isArray(check.reasons) ? check.reasons.join('; ') : 'preflight failed'}`);
			await api.dispatch(dispatchingId, selectedEndpointId);
			await onchanged(`Dispatched ${display(specification.name)} to ${display(selectedEndpoint()?.name)}.`);
		} catch (error) { onfailure(error instanceof Error ? error.message : 'Dispatch failed.'); }
		finally { dispatchingId = ''; }
	}

	async function reconcile(job: RecordValue) {
		try { await api.reconcile(String(job.job_id)); await onchanged('Job state refreshed.'); }
		catch (error) { onfailure(error instanceof Error ? error.message : 'Reconciliation failed.'); }
	}

	async function showDetails(job: RecordValue) {
		try { selectedJobId = String(job.job_id); jobEvents = (await api.jobEvents(selectedJobId)).events; }
		catch (error) { onfailure(error instanceof Error ? error.message : 'Could not load job events.'); }
	}

	async function reloadEvents() {
		try { jobEvents = (await api.jobEvents(selectedJobId)).events; }
		catch { /* Keep the last event stream visible during a transient restart. */ }
	}
</script>

<section class="panel queue-intro"><div><p class="eyebrow">REVIEW GATE</p><h2>Dispatch only the runs you want to execute.</h2><p>Readiness is checked immediately before Oracle Serve accepts an immutable specification.</p></div><label>Compute endpoint<select bind:value={selectedEndpointId} disabled={!computeEndpoints.length}>{#each computeEndpoints as endpoint}<option value={endpoint.endpoint_id}>{display(endpoint.name)} — {display(endpoint.status)}</option>{/each}</select></label></section>

<section class="panel compute-readiness"><div class="panel-head"><div><h2>Compute readiness</h2><p>Live capabilities and queue state from the selected endpoint.</p></div><span class="status {display(selectedEndpoint()?.status)}">{display(selectedEndpoint()?.status)}</span></div>{#if selectedEndpoint()}<div class="readiness-grid"><div><span>Endpoint</span><strong>{display(selectedEndpoint()!.name)}</strong><small>{display(selectedEndpoint()!.base_url)}</small></div><div><span>Queue</span><strong>{display(endpointQueue(selectedEndpoint())?.depth)} / {display(endpointQueue(selectedEndpoint())?.capacity)}</strong><small>waiting / capacity</small></div><div><span>Workers</span><strong>{endpointWorkers(selectedEndpoint()).length}</strong><small>{endpointWorkers(selectedEndpoint()).filter((worker) => worker.status === 'busy').length} busy</small></div></div>{#if selectedEndpoint()!.error}<div class="validation-message invalid"><strong>Endpoint unavailable</strong><span>{display(selectedEndpoint()!.error)}</span></div>{:else if endpointWorkers(selectedEndpoint()).length}<div class="worker-list">{#each endpointWorkers(selectedEndpoint()) as worker}<div><span class="worker-state {display(worker.status)}"></span><strong>{display(worker.name)}</strong><small>{display(worker.status)} · {workerGpuCount(worker)} GPU(s)</small></div>{/each}</div>{/if}{:else}<EmptyState title="No compute endpoint configured" text="Configure Oracle Serve before dispatching work." />{/if}</section>

<section class="panel"><div class="panel-head"><div><h2>Planned specifications</h2><p>Review fixed requests before assigning compute.</p></div><span class="count">{specifications.filter((item) => item.status === 'planned').length}</span></div>{#if specifications.length}<table><thead><tr><th>Specification</th><th>Experiment</th><th>Action</th><th>Status</th><th></th></tr></thead><tbody>{#each specifications as specification}<tr><td><strong>{display(specification.name)}</strong><small>{display(specification.specification_id)}</small></td><td>{display(specification.experiment_id)}</td><td>{display(specification.action)}</td><td><span class="status {display(specification.status)}">{display(specification.status)}</span></td><td>{#if specification.status === 'planned'}<button class="small" disabled={!selectedEndpointId || selectedEndpoint()?.status !== 'ready' || dispatchingId === specification.specification_id} on:click={() => dispatch(specification)}>{dispatchingId === specification.specification_id ? 'Checking…' : 'Preflight & dispatch'}</button>{/if}</td></tr>{/each}</tbody></table>{:else}<EmptyState title="No runs planned" text="Use the experiment workflow to create a controlled run matrix." action="Create experiment" onaction={oncreateexperiment} />{/if}</section>

<section class="panel"><div class="panel-head"><div><h2>Execution status</h2><p>Compute, validation, and catalog promotion are distinct stages.</p></div></div><div class="run-summary"><div><strong>{jobs.filter(isActiveJob).length}</strong><span>In progress</span></div><div><strong>{jobs.filter((job) => job.status === 'indexed').length}</strong><span>Indexed</span></div><div><strong>{jobs.filter((job) => ['failed', 'dispatch_failed', 'artifact_invalid'].includes(String(job.status))).length}</strong><span>Needs attention</span></div></div>{#if jobs.length}<table><thead><tr><th>Run</th><th>Stage</th><th>Worker</th><th>Result</th><th></th></tr></thead><tbody>{#each jobs as job}<tr class:attention={['failed', 'dispatch_failed', 'artifact_invalid'].includes(String(job.status))}><td><strong>{display(specificationName(job))}</strong><small>{display(job.action)} · {display(job.job_id)}</small></td><td><span class="status {display(job.status)}">{jobStage(job)}</span><small>{job.status === 'validating' ? 'Checking contract, inventory, and fingerprint' : display(job.remote_status)}</small></td><td>{display(job.worker_id)}<small>{display(job.started_at ?? job.submitted_at)}</small></td><td>{#if job.artifact_id}<button class="artifact-link" on:click={() => onviewartifact(job)}>{display(job.artifact_name ?? job.artifact_id)}</button><small>{display(job.artifact_lifecycle)} · {display(job.artifact_status)}</small>{:else if job.error}<span class="result-error" title={String(job.error)}>{display(job.error)}</span>{:else}<small>{display(job.output_path)}</small>{/if}</td><td><div class="row-actions">{#if isActiveJob(job)}<button class="secondary small" on:click={() => reconcile(job)}>Refresh</button>{/if}<button class="secondary small" on:click={() => showDetails(job)}>Details</button></div></td></tr>{/each}</tbody></table>{:else}<EmptyState title="Nothing has been dispatched" text="Approve a planned specification to send it to Oracle Serve." />{/if}
	{#if selectedJobId && selectedJob()}<div class="log-panel"><div><strong>{display(specificationName(selectedJob()!))}</strong><small>{selectedJobId}</small></div><button class="secondary small" on:click={() => { selectedJobId = ''; jobEvents = []; }}>Close</button><dl class="job-result"><div><dt>Stage</dt><dd>{jobStage(selectedJob()!)}</dd></div><div><dt>Output</dt><dd>{display(selectedJob()!.output_path)}</dd></div><div><dt>Started</dt><dd>{display(selectedJob()!.started_at)}</dd></div><div><dt>Finished</dt><dd>{display(selectedJob()!.completed_at)}</dd></div></dl>{#if reportMessages(selectedJob(), 'errors').length}<div class="validation-message invalid"><strong>Artifact validation failed</strong><ul>{#each reportMessages(selectedJob(), 'errors') as message}<li>{message}</li>{/each}</ul></div>{:else if selectedJob()!.validation_status === 'valid'}<div class="validation-message valid"><strong>Artifact verified and indexed</strong><span>Contract, inventory, fingerprint, completion, and sealing passed.</span>{#if reportMessages(selectedJob(), 'warnings').length}<ul>{#each reportMessages(selectedJob(), 'warnings') as message}<li>{message}</li>{/each}</ul>{/if}</div>{:else if selectedJob()!.error}<div class="validation-message invalid"><strong>Execution failed</strong><span>{display(selectedJob()!.error)}</span></div>{/if}{#if jobEvents.length}<ol>{#each jobEvents as event}<li><time>{display(event.timestamp)}</time><span class="event-{display(event.event_type)}">{display(event.message)}</span></li>{/each}</ol>{:else}<p class="empty">No events have been received yet.</p>{/if}</div>{/if}
</section>
