<script lang="ts">
	import { api, type RecordValue } from '$lib/api';

	export let artifactId: string;
	export let onclose: () => void;
	export let onfailure: (message: string) => void;

	type EvidenceTab = 'overview' | 'classes' | 'samples' | 'visuals';
	let evidence: RecordValue | null = null;
	let loading = true;
	let tab: EvidenceTab = 'overview';

	const asRecord = (value: unknown) => value && typeof value === 'object' ? value as RecordValue : null;
	const asRecords = (value: unknown) => Array.isArray(value) ? value as RecordValue[] : [];
	const display = (value: unknown) => value === null || value === undefined || value === '' ? '—' : String(value);
	const artifact = () => asRecord(evidence?.artifact);
	const classification = () => asRecord(evidence?.classification);
	const segmentation = () => asRecord(evidence?.segmentation);
	const availability = () => asRecord(evidence?.availability);
	const media = () => asRecords(evidence?.media);
	const matrix = () => asRecord(classification()?.confusion_matrix);
	const matrixRows = () => Array.isArray(matrix()?.matrix) ? matrix()!.matrix as number[][] : [];
	const classNames = () => Array.isArray(matrix()?.class_names) ? matrix()!.class_names as string[] : [];
	const perClass = () => asRecords(classification()?.per_class_metrics);
	const worstSamples = () => asRecords(segmentation()?.worst_samples);
	const metrics = () => Object.entries(asRecord(artifact()?.metrics) ?? {}).filter(([, value]) => Number.isFinite(Number(value)));
	const formatMetric = (value: unknown) => Number.isFinite(Number(value)) ? Number(value).toFixed(4) : '—';
	const rowMaximum = (row: number[]) => Math.max(...row.map(Number), 1);

	async function load() {
		loading = true;
		try { evidence = await api.artifactEvidence(artifactId); }
		catch (error) { onfailure(error instanceof Error ? error.message : 'Could not load artifact evidence.'); onclose(); }
		finally { loading = false; }
	}

	load();
</script>

<div class="drawer-backdrop" role="presentation" on:click|self={onclose}>
	<div class="evidence-drawer" role="dialog" aria-modal="true" aria-label="Artifact evidence">
		<header class="drawer-head">
			<div><p class="eyebrow">SEALED EVIDENCE</p><h2>{display(artifact()?.name ?? 'Loading artifact…')}</h2><p>{display(artifact()?.architecture)} · {display(artifact()?.task)} · {display(asRecord(artifact()?.protocol)?.split)} split</p></div>
			<button class="icon-button" aria-label="Close evidence" on:click={onclose}>×</button>
		</header>
		{#if loading}<div class="drawer-loading">Reading artifact evidence…</div>
		{:else if evidence}
			<nav class="evidence-tabs" aria-label="Evidence sections">
				<button class:active={tab === 'overview'} on:click={() => tab = 'overview'}>Overview</button>
				{#if artifact()?.task === 'classification'}<button class:active={tab === 'classes'} on:click={() => tab = 'classes'}>Class evidence</button>{/if}
				{#if artifact()?.task === 'segmentation'}<button class:active={tab === 'samples'} on:click={() => tab = 'samples'}>Sample evidence</button>{/if}
				<button class:active={tab === 'visuals'} on:click={() => tab = 'visuals'}>Visual evidence <span>{media().length}</span></button>
			</nav>

			<div class="drawer-body">
				{#if tab === 'overview'}
					<section class="evidence-summary">
						<div><span>Artifact</span><strong>{display(artifact()?.artifact_id)}</strong></div>
						<div><span>Dataset fingerprint</span><strong>{display(artifact()?.dataset_fingerprint_sha256)}</strong></div>
						<div><span>Evidence source</span><strong>Immutable artifact files</strong></div>
					</section>
					<section><div class="section-heading"><div><h3>Recorded metrics</h3><p>Values from the standard evaluation summary.</p></div></div><div class="evidence-metrics">{#each metrics() as [name, value]}<div><span>{name.replaceAll('_', ' ')}</span><strong>{formatMetric(value)}</strong></div>{/each}</div></section>
					<section><div class="section-heading"><div><h3>Available evidence</h3><p>Missing products are shown explicitly; the interface does not synthesize them.</p></div></div><div class="availability-grid">{#each [['confusion_matrix','Confusion matrix'],['per_class_metrics','Per-class metrics'],['sample_metrics','Sample metrics'],['overlays','Prediction overlays'],['activations','Activations / saliency']] as item}<div class:available={Boolean(availability()?.[item[0]])}><span>{availability()?.[item[0]] ? '✓' : '—'}</span><strong>{item[1]}</strong><small>{availability()?.[item[0]] ? 'Included' : 'Not included in artifact'}</small></div>{/each}</div></section>
				{:else if tab === 'classes'}
					{#if matrixRows().length}<section><div class="section-heading"><div><h3>Confusion matrix</h3><p>Rows are true classes; columns are predicted classes. Color is normalized within each row.</p></div></div><div class="matrix-scroll"><div class="confusion-matrix" style={`--matrix-size:${matrixRows().length}`} title="Confusion matrix">{#each matrixRows() as row, rowIndex}{#each row as value, columnIndex}<div class:diagonal={rowIndex === columnIndex} style={`--intensity:${Number(value) / rowMaximum(row)}`} title={`${classNames()[rowIndex] ?? rowIndex} → ${classNames()[columnIndex] ?? columnIndex}: ${value}`}>{matrixRows().length <= 15 ? value : ''}</div>{/each}{/each}</div></div></section>{/if}
					<section><div class="section-heading"><div><h3>Per-class performance</h3><p>Ordered as recorded in the artifact, typically weakest F1 first.</p></div></div>{#if perClass().length}<div class="table-scroll"><table><thead><tr><th>Class</th><th>Support</th><th>Precision</th><th>Recall</th><th>F1</th><th>Average precision</th></tr></thead><tbody>{#each perClass() as row}<tr><td><strong>{display(row.class_name)}</strong></td><td>{display(row.support)}</td><td>{formatMetric(row.precision)}</td><td>{formatMetric(row.recall)}</td><td>{formatMetric(row.f1_score)}</td><td>{formatMetric(row.average_precision)}</td></tr>{/each}</tbody></table></div>{:else}<p class="empty">This artifact does not include the standard per-class table.</p>{/if}</section>
				{:else if tab === 'samples'}
					<section><div class="section-heading"><div><h3>Lowest-performing samples</h3><p>Sample metrics sorted by Dice score; use these records to target review.</p></div></div>{#if worstSamples().length}<div class="table-scroll"><table><thead><tr><th>Sample UUID</th><th>Split</th><th>Dice</th><th>IoU</th><th>Precision</th><th>Recall</th><th>Pixel accuracy</th></tr></thead><tbody>{#each worstSamples() as row}<tr><td><code>{display(row.uuid)}</code></td><td>{display(row.split)}</td><td>{formatMetric(row.dice)}</td><td>{formatMetric(row.iou)}</td><td>{formatMetric(row.precision)}</td><td>{formatMetric(row.recall)}</td><td>{formatMetric(row.pixel_accuracy)}</td></tr>{/each}</tbody></table></div>{:else}<p class="empty">This artifact does not include standard segmentation sample metrics.</p>{/if}</section>
				{:else}
					<section><div class="section-heading"><div><h3>Artifact visualizations</h3><p>Figures, overlays, and activation products are served directly from the sealed artifact.</p></div></div>{#if media().length}<div class="evidence-gallery">{#each media() as item}<figure><a href={String(item.url)} target="_blank" rel="noreferrer"><img src={String(item.url)} alt={display(item.name)} loading="lazy" /></a><figcaption><span class="media-kind">{display(item.kind)}</span><strong>{display(item.name)}</strong><small>{display(item.path)}</small></figcaption></figure>{/each}</div>{:else}<div class="evidence-empty"><span>◇</span><h3>No visual evidence included</h3><p>The artifact has no standard figure, overlay, activation, saliency, or Grad-CAM image files. Oracle Builder will show them here when a run records them.</p></div>{/if}</section>
				{/if}
			</div>
		{/if}
	</div>
</div>
