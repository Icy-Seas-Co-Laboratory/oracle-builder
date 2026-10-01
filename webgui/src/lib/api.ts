export type RecordValue = Record<string, unknown>;
export type ApiRequestOptions = Pick<RequestInit, 'signal'>;
export type UploadProgress = { transferredBytes: number; totalBytes: number };

export class ApiRequestError extends Error {
	constructor(message: string, readonly status: number) { super(message); }
}

async function request<T>(path: string, options: RequestInit = {}, timeoutMs = 15_000): Promise<T> {
	const controller = new AbortController();
	const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
	let response: Response;
	try {
		response = await fetch(`/api${path}`, {
			...options,
			signal: options.signal ? AbortSignal.any([options.signal, controller.signal]) : controller.signal,
			headers: { 'content-type': 'application/json', ...options.headers }
		});
	} catch (error) {
		if (controller.signal.aborted) throw new Error(`The service did not respond within ${Math.ceil(timeoutMs / 1_000)} seconds.`);
		throw error;
	} finally { window.clearTimeout(timeout); }
	if (!response.ok) {
		const body = await response.json().catch(() => ({}));
		throw new ApiRequestError(String(body.detail ?? `${response.status} ${response.statusText}`), response.status);
	}
	return response.json() as Promise<T>;
}

export const api = {
	datasets: (options?: ApiRequestOptions) => request<{ datasets: RecordValue[] }>('/v1/datasets', options),
	configurationSchema: () => request<RecordValue>('/v1/config-schema'),
	modelDefinitionTemplates: () => request<{ templates: RecordValue[] }>('/v1/model-definition-templates'),
	modelDefinitions: (options?: ApiRequestOptions) => request<{ definitions: RecordValue[] }>('/v1/model-definitions', options),
	modelDefinition: (id: string, revision?: number) => request<RecordValue>(`/v1/model-definitions/${encodeURIComponent(id)}${revision ? `?revision=${revision}` : ''}`),
	createModelDefinition: (body: RecordValue) => request<RecordValue>('/v1/model-definitions', { method: 'POST', body: JSON.stringify(body) }),
	updateModelDefinition: (id: string, body: RecordValue) => request<RecordValue>(`/v1/model-definitions/${encodeURIComponent(id)}`, { method: 'PATCH', body: JSON.stringify(body) }),
	deleteModelDefinition: (id: string) => request<RecordValue>(`/v1/model-definitions/${encodeURIComponent(id)}`, { method: 'DELETE' }),
	duplicateModelDefinition: (id: string, body: RecordValue = {}) => request<RecordValue>(`/v1/model-definitions/${encodeURIComponent(id)}:duplicate`, { method: 'POST', body: JSON.stringify(body) }),
	queueDefinition: (id: string, body: RecordValue) => request<RecordValue>(`/v1/model-definitions/${encodeURIComponent(id)}:queue`, { method: 'POST', body: JSON.stringify(body) }, 960_000),
	verifyQueuedRuns: (body: RecordValue) => request<RecordValue>('/v1/queued-runs:verify', { method: 'POST', body: JSON.stringify(body) }),
	validateAndQueueDefinition: (id: string, body: RecordValue) => request<RecordValue>(`/v1/model-definitions/${encodeURIComponent(id)}:validate-and-queue`, { method: 'POST', body: JSON.stringify(body) }, 960_000),
	validateAndQueueDefinitionSchedule: (id: string, body: RecordValue) => request<RecordValue>(`/v1/model-definitions/${encodeURIComponent(id)}:validate-and-queue/schedule`, { method: 'POST', body: JSON.stringify(body) }),
	authorizeQueuedRuns: (body: RecordValue) => request<RecordValue>('/v1/queued-runs:start', { method: 'POST', body: JSON.stringify(body) }),
	modelSetup: (architecture: string) => request<RecordValue>(`/v1/model-setups/${encodeURIComponent(architecture)}`),
	modelPreview: (body: RecordValue) => request<RecordValue>('/v1/model-previews', { method: 'POST', body: JSON.stringify(body) }),
	datasetDetail: (id: string) => request<RecordValue>(`/v1/datasets/${id}/detail`),
	datasetPreviews: (id: string, options: { offset?: number; limit?: number; split?: string; label?: string } = {}) => {
		const search = new URLSearchParams();
		for (const [key, value] of Object.entries(options)) if (value !== undefined && value !== '') search.set(key, String(value));
		return request<RecordValue>(`/v1/datasets/${id}/previews?${search}`);
	},
	ingestDataset: (path: string) => request<RecordValue>('/v1/datasets:ingest', { method: 'POST', body: JSON.stringify({ path }) }),
	artifacts: (options?: ApiRequestOptions) => request<{ artifacts: RecordValue[] }>('/v1/artifacts', options),
	artifactCatalog: () => request<{ artifacts: RecordValue[] }>('/v1/artifacts/catalog'),
	artifactCatalogQuery: (body: RecordValue) => request<RecordValue>('/v1/artifacts/catalog/query', { method: 'POST', body: JSON.stringify(body) }),
	artifactFilterSchema: () => request<RecordValue>('/v1/artifacts/filter-schema'),
	artifactTags: () => request<RecordValue>('/v1/artifact-tags'),
	createArtifactTag: (body: RecordValue) => request<RecordValue>('/v1/tags', { method: 'POST', body: JSON.stringify(body) }),
	assignArtifactTags: (id: string, tags: string[]) => request<RecordValue>(`/v1/artifacts/${encodeURIComponent(id)}/tags`, { method: 'PUT', body: JSON.stringify({ tags }) }),
	artifactArchitectureView: (id: string) => request<RecordValue>(`/v1/artifacts/${encodeURIComponent(id)}/architecture-view`),
	artifactDetail: (id: string) => request<RecordValue>(`/v1/artifacts/${id}/detail`),
	artifactHistory: (id: string) => request<RecordValue>(`/v1/artifacts/${id}/history`),
	artifactEvidence: (id: string) => request<RecordValue>(`/v1/artifacts/${id}/evidence`),
	recipes: () => request<{ recipes: RecordValue[] }>('/v1/recipes'),
	experiments: () => request<{ experiments: RecordValue[] }>('/v1/experiments'),
	experimentResults: (id: string) => request<RecordValue>(`/v1/experiments/${id}/results`),
	comparisons: () => request<{ comparisons: RecordValue[] }>('/v1/comparisons'),
	comparisonGroups: () => request<{ comparison_groups: RecordValue[] }>('/v1/comparison-groups'),
	comparisonGroup: (id: string) => request<RecordValue>(`/v1/comparison-groups/${id}`),
	specifications: (experimentId?: string) => request<{ specifications: RecordValue[] }>(`/v1/specifications${experimentId ? `?experiment_id=${encodeURIComponent(experimentId)}` : ''}`),
	// Job state is durable pull-worker state. The retired push-runtime refresh
	// query is intentionally ignored so callers cannot revive it by accident.
	jobs: (_refresh = false, options?: ApiRequestOptions) => request<{ jobs: RecordValue[] }>('/v1/jobs', options),
	operation: (id: string, options?: ApiRequestOptions) => request<RecordValue>(`/v1/operations/${encodeURIComponent(id)}`, options),
	serverLogs: (tailLines = 400) => request<RecordValue>(`/v1/system/logs?tail_lines=${Math.max(1, Math.min(2000, tailLines))}`),
	health: (options?: ApiRequestOptions) => request<RecordValue>('/health/ready', options),
	workerPools: () => request<{ pools: RecordValue[] }>('/v1/worker-pools'),
	createWorkerPool: (body: { name: string; allowed_actions: string[]; max_workers?: number }) => request<RecordValue>('/v1/worker-pools', { method: 'POST', body: JSON.stringify(body) }),
	workerDeployments: () => request<{ deployments: RecordValue[] }>('/v1/worker-deployments'),
	registeredWorkers: (poolId?: string) => request<{ workers: RecordValue[] }>(`/v1/workers${poolId ? `?pool_id=${encodeURIComponent(poolId)}` : ''}`),
	workerLeases: (workerId?: string) => request<{ leases: RecordValue[] }>(`/v1/worker-leases${workerId ? `?worker_id=${encodeURIComponent(workerId)}` : ''}`),
	executionRun: (runId: string) => request<{ run: RecordValue }>(`/v1/execution-runs/${encodeURIComponent(runId)}`),
	queuedRuns: () => request<{ queued_runs: RecordValue[] }>('/v1/queued-runs'),
	replaceQueuedRun: (id: string, body: RecordValue) => request<RecordValue>(`/v1/queued-runs/${encodeURIComponent(id)}:replace`, { method: 'POST', body: JSON.stringify(body) }, 960_000),
	artifactReplicas: (status?: 'pending' | 'replicated' | 'failed') => request<{ replications: RecordValue[] }>(`/v1/artifact-replicas${status ? `?status=${status}` : ''}`),
	jobEvents: (id: string) => request<{ events: RecordValue[] }>(`/v1/jobs/${id}/events`),
	jobCommands: (id: string) => request<{ commands: RecordValue[] }>(`/v1/jobs/${encodeURIComponent(id)}/commands`),
	issueJobCommand: (id: string, body: { action: 'pause' | 'yield' | 'stop_now' | 'restart' | 'resume'; reason?: string; command_id?: string }) => request<{ command: RecordValue }>(`/v1/jobs/${encodeURIComponent(id)}/commands`, { method: 'POST', body: JSON.stringify(body) }),
	workerHeartbeat: (id: string) => request<{ heartbeat: RecordValue | null }>(`/v1/workers/${encodeURIComponent(id)}/heartbeat`),
	jobTiming: (id: string) => request<RecordValue>(`/v1/jobs/${id}/timing`),
	jobOutputUpload: (id: string) => request<{ upload: RecordValue | null }>(`/v1/jobs/${id}/output-upload`),
	fileRoots: () => request<{ roots: RecordValue[] }>('/v1/files/roots'),
	files: (root: string, path = '') => request<RecordValue>(`/v1/files?root=${encodeURIComponent(root)}&path=${encodeURIComponent(path)}`),
	upload: async (kind: 'datasets' | 'configs' | 'models', file: File): Promise<RecordValue> => {
		const response = await fetch(`/api/v1/uploads/${kind}/${encodeURIComponent(file.name)}`, { method: 'POST', headers: { 'content-type': file.type || 'application/octet-stream' }, body: file });
		if (!response.ok) {
			const body = await response.json().catch(() => ({}));
			throw new Error(String(body.detail ?? `${response.status} ${response.statusText}`));
		}
		return response.json() as Promise<RecordValue>;
	},
	createUploadSession: (kind: 'datasets' | 'configs' | 'models', file: File) => request<RecordValue>('/v1/uploads/sessions', {
		method: 'POST', body: JSON.stringify({ kind, filename: file.name, size_bytes: file.size })
	}),
	uploadSession: (id: string) => request<RecordValue>(`/v1/uploads/sessions/${encodeURIComponent(id)}`),
	cancelUploadSession: (id: string) => request<RecordValue>(`/v1/uploads/sessions/${encodeURIComponent(id)}`, { method: 'DELETE' }),
	resumableUpload: async (
		kind: 'datasets' | 'configs' | 'models', file: File,
		onprogress?: (progress: UploadProgress) => void,
	): Promise<RecordValue> => {
		// Persist only the opaque session id. The server owns partial bytes and
		// verifies each declared range, so a page refresh can safely continue.
		const storageKey = `oracle-upload-v1:${kind}:${file.name}:${file.size}:${file.lastModified}`;
		let session: RecordValue | null = null;
		try {
			const existing = window.localStorage.getItem(storageKey);
			if (existing) {
				const candidate = await api.uploadSession(existing);
				if (candidate.status === 'uploading' && Number(candidate.size_bytes) === file.size) session = candidate;
			}
		} catch { window.localStorage.removeItem(storageKey); }
		if (!session) {
			session = await api.createUploadSession(kind, file);
			window.localStorage.setItem(storageKey, String(session.upload_id));
		}
		const uploadId = String(session.upload_id);
		const chunkSize = Number(session.chunk_size_bytes);
		let offset = Number(session.received_bytes ?? 0);
		onprogress?.({ transferredBytes: offset, totalBytes: file.size });
		try {
			while (offset < file.size) {
				const end = Math.min(offset + chunkSize, file.size);
				const response = await fetch(`/api/v1/uploads/sessions/${encodeURIComponent(uploadId)}/part`, {
					method: 'PUT',
					headers: {
						'content-type': file.type || 'application/octet-stream',
						'content-range': `bytes ${offset}-${end - 1}/${file.size}`
					},
					body: file.slice(offset, end)
				});
				if (!response.ok) {
					const body = await response.json().catch(() => ({}));
					throw new Error(String(body.detail ?? `${response.status} ${response.statusText}`));
				}
				offset = end;
				onprogress?.({ transferredBytes: offset, totalBytes: file.size });
			}
			const completed = await request<RecordValue>(`/v1/uploads/sessions/${encodeURIComponent(uploadId)}:complete`, { method: 'POST' });
			window.localStorage.removeItem(storageKey);
			return completed;
		} catch (error) {
			// Keep the session id so the user can retry after a network interruption.
			throw error;
		}
	},
	datasetDownloadUrl: (id: string) => `/api/v1/datasets/${encodeURIComponent(id)}:download`,
	artifactDownloadUrl: (id: string) => `/api/v1/artifacts/${encodeURIComponent(id)}:download`,
	// Batch inference is an Orchestrator-owned WorkUnit.  The browser never
	// addresses a worker or chooses a worker-local model/data path.
	inferenceRuns: () => request<{ inference_runs: RecordValue[] }>('/v1/inference-runs'),
	inferenceRun: (id: string) => request<RecordValue>(`/v1/inference-runs/${encodeURIComponent(id)}`),
	createInferenceRun: (body: RecordValue) => request<RecordValue>('/v1/inference-runs', { method: 'POST', body: JSON.stringify(body) }),
	startInferenceRun: (id: string) => request<RecordValue>(`/v1/inference-runs/${encodeURIComponent(id)}:start`, { method: 'POST' }),
	cancelInferenceRun: (id: string) => request<RecordValue>(`/v1/inference-runs/${encodeURIComponent(id)}:cancel`, { method: 'POST' }),
	inferenceDownloadUrl: (id: string) => `/api/v1/inference-runs/${encodeURIComponent(id)}:download`,
	createRecipe: (body: RecordValue) => request<RecordValue>('/v1/recipes', { method: 'POST', body: JSON.stringify(body) }),
	modelDrafts: () => request<RecordValue>('/v1/model-drafts'),
	modelDraft: (id: string) => request<RecordValue>(`/v1/model-drafts/${encodeURIComponent(id)}`),
	createModelDraft: (body: RecordValue) => request<RecordValue>('/v1/model-drafts', { method: 'POST', body: JSON.stringify(body) }),
	updateModelDraft: (id: string, body: RecordValue) => request<RecordValue>(`/v1/model-drafts/${encodeURIComponent(id)}`, { method: 'PATCH', body: JSON.stringify(body) }),
	cloneModelDraft: (id: string, body: RecordValue = {}) => request<RecordValue>(`/v1/model-drafts/${encodeURIComponent(id)}:clone`, { method: 'POST', body: JSON.stringify(body) }),
	validateModelDraft: (id: string) => request<RecordValue>(`/v1/model-drafts/${encodeURIComponent(id)}:validate`),
	previewModelDraft: (id: string, datasetId: string) => request<RecordValue>(`/v1/model-drafts/${encodeURIComponent(id)}:preview?dataset_id=${encodeURIComponent(datasetId)}`),
	trainingCatalogEntries: () => request<RecordValue>('/v1/training-catalog'),
	scanTrainingCatalog: (body: RecordValue = {}) => request<RecordValue>('/v1/training-catalog:scan', { method: 'POST', body: JSON.stringify(body) }),
	trainingCatalogDetail: (id: string) => request<RecordValue>(`/v1/training-catalog/${encodeURIComponent(id)}`),
	trainingCatalogPreviews: (id: string, options: RecordValue = {}) => {
		const search = new URLSearchParams(); for (const [key, value] of Object.entries(options)) if (value !== undefined && value !== '') search.set(key, String(value));
		return request<RecordValue>(`/v1/training-catalog/${encodeURIComponent(id)}/previews?${search}`);
	},
	freezeTrainingCatalogEntry: (id: string) => request<RecordValue>(`/v1/training-catalog/${encodeURIComponent(id)}:freeze`, { method: 'POST' }),
	trainingCatalogComparison: (ids: string[]) => request<RecordValue>('/v1/training-catalog:compare', { method: 'POST', body: JSON.stringify({ catalog_ids: ids }) }),
	createComparison: (body: RecordValue) => request<RecordValue>('/v1/comparisons', { method: 'POST', body: JSON.stringify(body) }),
	createComparisonGroup: (body: RecordValue) => request<RecordValue>('/v1/comparison-groups', { method: 'POST', body: JSON.stringify(body) }),
	scan: (root: string) => request<RecordValue>('/v1/catalog:scan', { method: 'POST', body: JSON.stringify({ root }) })
};
