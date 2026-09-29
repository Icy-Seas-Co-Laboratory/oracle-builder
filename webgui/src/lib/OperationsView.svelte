<script lang="ts">
	import { onMount } from 'svelte';
	import { api, type RecordValue } from '$lib/api';

	export let onfailure: (message: string) => void = () => {};

	let deployments: RecordValue[] = [];
	let workers: RecordValue[] = [];
	let leases: RecordValue[] = [];
	let queuedRuns: RecordValue[] = [];
	let replicas: RecordValue[] = [];
	let logs: RecordValue[] = [];
	let loading = true;

	const records = (value: unknown): RecordValue[] => Array.isArray(value) ? value as RecordValue[] : [];
	const text = (value: unknown, fallback = '—') => value === undefined || value === null || value === '' ? fallback : String(value);
	const activeLease = (lease: RecordValue) => ['active', 'acknowledged', 'uploading'].includes(String(lease.status));
	const activeRun = (run: RecordValue) => ['ready', 'waiting_for_resources', 'leased', 'running'].includes(String(run.status));
	const failedDeployment = (deployment: RecordValue) => ['failed', 'unknown'].includes(String(deployment.state));
	const stalledWorker = (worker: RecordValue) => worker.state === 'offline' || worker.state === 'unhealthy';
	$: activeLeases = leases.filter(activeLease);
	$: backlog = queuedRuns.filter(activeRun);
	$: failedReplicas = replicas.filter((replica) => replica.status === 'failed');
	$: unhealthyDeployments = deployments.filter(failedDeployment);
	$: unhealthyWorkers = workers.filter(stalledWorker);
	$: status = unhealthyDeployments.length || unhealthyWorkers.length || failedReplicas.length ? 'needs-attention' : activeLeases.length || backlog.length ? 'working' : 'healthy';

	async function load() {
		loading = true;
		try {
			const [deploymentResult, workerResult, leaseResult, queueResult, replicaResult, logResult] = await Promise.all([
				api.workerDeployments(), api.registeredWorkers(), api.workerLeases(), api.queuedRuns(), api.artifactReplicas('failed'), api.serverLogs(80)
			]);
			deployments = records(deploymentResult.deployments);
			workers = records(workerResult.workers);
			leases = records(leaseResult.leases);
			queuedRuns = records(queueResult.queued_runs);
			replicas = records(replicaResult.replications);
			logs = records(logResult.logs);
		} catch (error) {
			onfailure(error instanceof Error ? error.message : 'Could not load demo operations state.');
		} finally { loading = false; }
	}

	onMount(() => { void load(); });
</script>

<section class="operations-intro panel">
	<div><p class="eyebrow">DEMO OPERATIONS</p><h2>One screen for the control-plane essentials.</h2><p>Deployments, worker heartbeats, queue pressure, artifact durability, and bounded service logs all come from the Orchestrator.</p></div>
	<div class="row-actions"><span class="status {status}">{status.replace('-', ' ')}</span><button class="secondary small" disabled={loading} on:click={load}>{loading ? 'Refreshing…' : 'Refresh'}</button></div>
</section>

<section class="operation-stats" aria-label="Demo operations summary">
	<article><span>Deployments</span><strong>{deployments.length}</strong><small>{unhealthyDeployments.length ? `${unhealthyDeployments.length} need attention` : 'All known'}</small></article>
	<article><span>Worker heartbeat</span><strong>{workers.length}</strong><small>{unhealthyWorkers.length ? `${unhealthyWorkers.length} unhealthy` : 'Registered workers'}</small></article>
	<article><span>Lease backlog</span><strong>{backlog.length}</strong><small>{activeLeases.length} active lease{activeLeases.length === 1 ? '' : 's'}</small></article>
	<article><span>Replica failures</span><strong>{failedReplicas.length}</strong><small>{failedReplicas.length ? 'Retry or inspect replica state' : 'No failed replicas'}</small></article>
</section>

<section class="panel"><div class="panel-head"><div><p class="eyebrow">LIFECYCLE</p><h2>Worker deployments</h2><p>Desired state is persisted; actual state is reconciled by the configured local-process provider.</p></div></div>{#if deployments.length}<div class="table-scroll"><table><thead><tr><th>Deployment</th><th>Pool</th><th>Desired</th><th>Actual</th><th>Last reconciled</th><th>Actionable detail</th></tr></thead><tbody>{#each deployments as deployment}<tr><td><strong>{text(deployment.name)}</strong><small>{text(deployment.profile_id)}</small></td><td>{text(deployment.pool_id)}</td><td><span class="status">{text(deployment.desired_state)}</span></td><td><span class="status {text(deployment.state)}">{text(deployment.state)}</span></td><td>{text(deployment.last_reconciled_at)}</td><td>{text(deployment.error, '—')}</td></tr>{/each}</tbody></table></div>{:else}<p class="empty">No managed worker deployment exists. Create one from the local demo profile.</p>{/if}</section>

<section class="panel"><div class="panel-head"><div><p class="eyebrow">HEARTBEATS</p><h2>Registered workers</h2><p>Workers update their durable heartbeat while they poll or hold a lease.</p></div></div>{#if workers.length}<div class="table-scroll"><table><thead><tr><th>Worker</th><th>State</th><th>Last seen</th><th>Active lease</th></tr></thead><tbody>{#each workers as worker}<tr><td><strong>{text(worker.name)}</strong><small>{text(worker.worker_id)}</small></td><td><span class="status {text(worker.state)}">{text(worker.state)}</span></td><td>{text(worker.last_seen_at)}</td><td>{leases.some((lease) => activeLease(lease) && String(lease.worker_id) === String(worker.worker_id)) ? 'Yes' : 'No'}</td></tr>{/each}</tbody></table></div>{:else}<p class="empty">No worker has registered yet.</p>{/if}</section>

<div class="operations-grid"><section class="panel"><div class="panel-head"><div><p class="eyebrow">QUEUE</p><h2>Lease backlog</h2><p>Only sealed, authorized work is eligible for a worker lease.</p></div></div>{#if backlog.length}<div class="compact-list">{#each backlog as run}<article><strong>{text(run.name, text(run.queued_run_id))}</strong><span class="status {text(run.status)}">{text(run.status)}</span><small>{text(run.failure_reason, 'Awaiting worker capacity')}</small></article>{/each}</div>{:else}<p class="empty">No queued or active work.</p>{/if}</section><section class="panel"><div class="panel-head"><div><p class="eyebrow">DURABILITY</p><h2>Failed replicas</h2><p>Failures do not make the local artifact unavailable, but should be resolved before a recovery-dependent demo.</p></div></div>{#if failedReplicas.length}<div class="compact-list">{#each failedReplicas as replica}<article><strong>{text(replica.artifact_ref_uri)}</strong><span class="status failed">failed</span><small>{text(replica.last_error)}</small></article>{/each}</div>{:else}<p class="empty">No failed replica records.</p>{/if}</section></div>

<section class="panel"><div class="panel-head"><div><p class="eyebrow">ACTIONABLE LOGS</p><h2>Recent bounded service output</h2><p>Use deployment and lease identifiers above to correlate the relevant service log.</p></div></div><div class="log-grid">{#each logs as log}<article><header><strong>{text(log.name, text(log.service))}</strong><span class:available={Boolean(log.available)}>{log.available ? 'available' : 'unavailable'}</span></header>{#if log.available}<pre>{text(log.text, 'No output yet.')}</pre>{:else}<p>{text(log.message)}</p>{/if}</article>{/each}</div></section>

<style>
	.operations-intro { display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; }.operations-intro p { max-width:48rem; margin-bottom:0; }.operations-intro .row-actions { display:flex; align-items:center; gap:.5rem; }.operation-stats { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.75rem; margin-bottom:1rem; }.operation-stats article { display:grid; gap:.16rem; padding:.9rem 1rem; border:1px solid var(--line); border-radius:10px; background:var(--surface); box-shadow:var(--shadow); }.operation-stats span,.operation-stats small { color:var(--muted); font-size:.69rem; }.operation-stats strong { color:var(--navy); font-size:1.45rem; }.status.healthy,.status.working,.status.available { color:#087568; background:#dff5ef; }.status.needs-attention,.status.failed,.status.unknown,.status.offline,.status.unhealthy { color:#9d3a3a; background:#fae7e7; }.operations-grid { display:grid; grid-template-columns:1fr 1fr; gap:1rem; }.compact-list { display:grid; gap:.55rem; }.compact-list article { display:grid; grid-template-columns:minmax(0,1fr) auto; gap:.2rem .6rem; padding:.65rem; border:1px solid var(--line); border-radius:7px; background:#f9fbfa; }.compact-list small { grid-column:1/-1; color:var(--muted); overflow-wrap:anywhere; }.log-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:.75rem; }.log-grid article { min-width:0; border:1px solid var(--line); border-radius:8px; overflow:hidden; }.log-grid header { display:flex; align-items:center; justify-content:space-between; gap:.5rem; padding:.55rem .65rem; background:#f7faf9; }.log-grid header span { color:#8a5555; font-size:.66rem; font-weight:700; }.log-grid header span.available { color:#087568; }.log-grid pre { max-height:15rem; margin:0; overflow:auto; padding:.65rem; color:#d7e8e5; background:#15282e; font-size:.67rem; white-space:pre-wrap; }.log-grid p { padding:.65rem; color:var(--muted); font-size:.75rem; } @media (max-width:850px) { .operations-intro { flex-direction:column; }.operation-stats,.operations-grid { grid-template-columns:1fr 1fr; } } @media (max-width:520px) { .operation-stats,.operations-grid { grid-template-columns:1fr; } }
</style>
