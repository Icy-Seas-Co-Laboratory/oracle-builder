export type RecordValue = Record<string, unknown>;

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
	const controller = new AbortController();
	const timeout = window.setTimeout(() => controller.abort(), 15_000);
	let response: Response;
	try {
		response = await fetch(`/api${path}`, {
			...options,
			signal: options.signal ?? controller.signal,
			headers: { 'content-type': 'application/json', ...options.headers }
		});
	} catch (error) {
		if (controller.signal.aborted) throw new Error('The service did not respond within 15 seconds.');
		throw error;
	} finally { window.clearTimeout(timeout); }
	if (!response.ok) {
		const body = await response.json().catch(() => ({}));
		throw new Error(String(body.detail ?? `${response.status} ${response.statusText}`));
	}
	return response.json() as Promise<T>;
}

export const api = {
	datasets: () => request<{ datasets: RecordValue[] }>('/v1/datasets'),
	configurationSchema: () => request<RecordValue>('/v1/config-schema'),
	modelSetup: (architecture: string) => request<RecordValue>(`/v1/model-setups/${encodeURIComponent(architecture)}`),
	modelPreview: (body: RecordValue) => request<RecordValue>('/v1/model-previews', { method: 'POST', body: JSON.stringify(body) }),
	datasetDetail: (id: string) => request<RecordValue>(`/v1/datasets/${id}/detail`),
	datasetPreviews: (id: string, options: { offset?: number; limit?: number; split?: string; label?: string } = {}) => {
		const search = new URLSearchParams();
		for (const [key, value] of Object.entries(options)) if (value !== undefined && value !== '') search.set(key, String(value));
		return request<RecordValue>(`/v1/datasets/${id}/previews?${search}`);
	},
	ingestDataset: (path: string) => request<RecordValue>('/v1/datasets:ingest', { method: 'POST', body: JSON.stringify({ path }) }),
	artifacts: () => request<{ artifacts: RecordValue[] }>('/v1/artifacts'),
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
	modelDrafts: () => request<RecordValue>('/v1/model-drafts'),
	modelDraft: (id: string) => request<RecordValue>(`/v1/model-drafts/${encodeURIComponent(id)}`),
	createModelDraft: (body: RecordValue) => request<RecordValue>('/v1/model-drafts', { method: 'POST', body: JSON.stringify(body) }),
	updateModelDraft: (id: string, body: RecordValue) => request<RecordValue>(`/v1/model-drafts/${encodeURIComponent(id)}`, { method: 'PATCH', body: JSON.stringify(body) }),
	cloneModelDraft: (id: string, body: RecordValue = {}) => request<RecordValue>(`/v1/model-drafts/${encodeURIComponent(id)}:clone`, { method: 'POST', body: JSON.stringify(body) }),
	validateModelDraft: (id: string) => request<RecordValue>(`/v1/model-drafts/${encodeURIComponent(id)}:validate`),
	previewModelDraft: (id: string, datasetId: string) => request<RecordValue>(`/v1/model-drafts/${encodeURIComponent(id)}:preview?dataset_id=${encodeURIComponent(datasetId)}`),
	planTrainingFromDraft: (id: string, body: RecordValue) => request<RecordValue>(`/v1/model-drafts/${encodeURIComponent(id)}:plan-training`, { method: 'POST', body: JSON.stringify(body) }),
	trainingCatalogEntries: () => request<RecordValue>('/v1/training-catalog'),
	scanTrainingCatalog: (body: RecordValue = {}) => request<RecordValue>('/v1/training-catalog:scan', { method: 'POST', body: JSON.stringify(body) }),
	trainingCatalogDetail: (id: string) => request<RecordValue>(`/v1/training-catalog/${encodeURIComponent(id)}`),
	trainingCatalogPreviews: (id: string, options: RecordValue = {}) => {
		const search = new URLSearchParams(); for (const [key, value] of Object.entries(options)) if (value !== undefined && value !== '') search.set(key, String(value));
		return request<RecordValue>(`/v1/training-catalog/${encodeURIComponent(id)}/previews?${search}`);
	},
	freezeTrainingCatalogEntry: (id: string) => request<RecordValue>(`/v1/training-catalog/${encodeURIComponent(id)}:freeze`, { method: 'POST' }),
	trainingCatalogComparison: (ids: string[]) => request<RecordValue>('/v1/training-catalog:compare', { method: 'POST', body: JSON.stringify({ catalog_ids: ids }) }),
	createTrainingExperiment: (body: RecordValue) => request<RecordValue>('/v1/experiments:train', { method: 'POST', body: JSON.stringify(body) }),
	createModelImport: (body: RecordValue) => request<RecordValue>('/v1/model-imports', { method: 'POST', body: JSON.stringify(body) }),
	createComparison: (body: RecordValue) => request<RecordValue>('/v1/comparisons', { method: 'POST', body: JSON.stringify(body) }),
	createComparisonGroup: (body: RecordValue) => request<RecordValue>('/v1/comparison-groups', { method: 'POST', body: JSON.stringify(body) }),
	preflight: (id: string, endpointId: string) => request<RecordValue>(`/v1/specifications/${id}/preflight?endpoint_id=${encodeURIComponent(endpointId)}`),
	updatePlannedSpecification: (id: string, body: RecordValue) => request<RecordValue>(`/v1/specifications/${id}`, { method: 'PATCH', body: JSON.stringify(body) }),
	specificationDetail: (id: string) => request<RecordValue>(`/v1/specifications/${id}`),
	dispatch: (id: string, endpoint_id: string) => request<RecordValue>(`/v1/specifications/${id}:dispatch`, { method: 'POST', body: JSON.stringify({ endpoint_id }) }),
	reconcile: (id: string) => request<RecordValue>(`/v1/jobs/${id}:reconcile`, { method: 'POST' }),
	scan: (root: string) => request<RecordValue>('/v1/catalog:scan', { method: 'POST', body: JSON.stringify({ root }) })
};
