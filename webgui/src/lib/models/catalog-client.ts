import type { RecordValue } from '$lib/api';

export type CatalogQuery = {
	filters?: Record<string, unknown>;
	sort?: { field: string; direction: 'asc' | 'desc' };
	columns?: string[];
	offset?: number;
	limit?: number;
};

async function call<T>(path: string, options: RequestInit = {}): Promise<T> {
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

/** Control-plane client kept separate while the legacy API client remains in use. */
export const modelCatalogClient = {
	artifactCatalogQuery: (query: CatalogQuery) => call<{ artifacts: RecordValue[]; total?: number; columns?: string[] }>(
		'/v1/artifacts/catalog/query', { method: 'POST', body: JSON.stringify({ ...query, sort: query.sort?.field ?? 'updated_at', order: query.sort?.direction ?? 'desc' }) }
	),
	reindexArtifactCatalog: (artifactIds?: string[]) => call<{ refreshed: string[]; missing: string[]; skipped: RecordValue[]; artifact_files_changed: boolean }>(
		'/v1/artifacts/catalog:reindex', { method: 'POST', body: JSON.stringify(artifactIds?.length ? { artifact_ids: artifactIds } : {}) }
	),
	artifactFilterSchema: () => call<{ fields: RecordValue[]; default_columns?: string[] }>('/v1/artifacts/filter-schema'),
	artifactTags: () => call<{ tags: RecordValue[] }>('/v1/artifact-tags'),
	assignArtifactTags: (artifactIds: string[], tags: string[]) => call<RecordValue>('/v1/artifact-tags/assign', {
		method: 'POST', body: JSON.stringify({ artifact_ids: artifactIds, tags })
	}),
	artifactDetail: (id: string) => call<RecordValue>(`/v1/artifacts/${encodeURIComponent(id)}/detail`),
	artifactArchitectureView: (id: string) => call<RecordValue>(`/v1/artifacts/${encodeURIComponent(id)}/architecture-view`),
	comparisonGroups: () => call<{ comparison_groups: RecordValue[] }>('/v1/comparison-groups'),
	comparisonGroup: (id: string) => call<RecordValue>(`/v1/comparison-groups/${encodeURIComponent(id)}`),
	createComparisonGroup: (body: RecordValue) => call<RecordValue>('/v1/comparison-groups', { method: 'POST', body: JSON.stringify(body) })
};
