export type RecordValue = Record<string, unknown>;

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
	const response = await fetch(`/api${path}`, {
		...options,
		headers: { 'content-type': 'application/json', ...options.headers }
	});
	if (!response.ok) {
		const body = await response.json().catch(() => ({}));
		throw new Error(String(body.detail ?? `${response.status} ${response.statusText}`));
	}
	return response.json() as Promise<T>;
}

export const api = {
	datasets: () => request<{ datasets: RecordValue[] }>('/v1/datasets'),
	ingestDataset: (path: string) => request<RecordValue>('/v1/datasets:ingest', { method: 'POST', body: JSON.stringify({ path }) }),
	artifacts: () => request<{ artifacts: RecordValue[] }>('/v1/artifacts'),
	artifactEvidence: (id: string) => request<RecordValue>(`/v1/artifacts/${id}/evidence`),
	recipes: () => request<{ recipes: RecordValue[] }>('/v1/recipes'),
	experiments: () => request<{ experiments: RecordValue[] }>('/v1/experiments'),
	experimentResults: (id: string) => request<RecordValue>(`/v1/experiments/${id}/results`),
	comparisons: () => request<{ comparisons: RecordValue[] }>('/v1/comparisons'),
	specifications: () => request<{ specifications: RecordValue[] }>('/v1/specifications'),
	jobs: (refresh = false) => request<{ jobs: RecordValue[] }>(`/v1/jobs${refresh ? '?refresh=true' : ''}`),
	health: () => request<RecordValue>('/health/ready'),
	computeEndpoints: (refresh = false) => request<{ endpoints: RecordValue[] }>(`/v1/compute/endpoints${refresh ? '?refresh=true' : ''}`),
	jobEvents: (id: string) => request<{ events: RecordValue[] }>(`/v1/jobs/${id}/events`),
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
	createRecipe: (body: RecordValue) => request<RecordValue>('/v1/recipes', { method: 'POST', body: JSON.stringify(body) }),
	createTrainingExperiment: (body: RecordValue) => request<RecordValue>('/v1/experiments:train', { method: 'POST', body: JSON.stringify(body) }),
	createModelImport: (body: RecordValue) => request<RecordValue>('/v1/model-imports', { method: 'POST', body: JSON.stringify(body) }),
	createComparison: (body: RecordValue) => request<RecordValue>('/v1/comparisons', { method: 'POST', body: JSON.stringify(body) }),
	preflight: (id: string, endpointId: string) => request<RecordValue>(`/v1/specifications/${id}/preflight?endpoint_id=${encodeURIComponent(endpointId)}`),
	dispatch: (id: string, endpoint_id: string) => request<RecordValue>(`/v1/specifications/${id}:dispatch`, { method: 'POST', body: JSON.stringify({ endpoint_id }) }),
	reconcile: (id: string) => request<RecordValue>(`/v1/jobs/${id}:reconcile`, { method: 'POST' }),
	scan: (root: string) => request<RecordValue>('/v1/catalog:scan', { method: 'POST', body: JSON.stringify({ root }) })
};
