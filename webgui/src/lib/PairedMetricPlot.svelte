<script lang="ts">
	import type { RecordValue } from '$lib/api';

	export let points: RecordValue[] = [];
	export let metric = 'loss';
	export let title = 'Loss';
	export let mode: 'epoch' | 'batch' = 'epoch';
	export let validationValue: number | null = null;
	export let validationLabel = 'Last completed epoch validation';

	const numeric = (value: unknown): number | null => typeof value === 'number' && Number.isFinite(value) ? value : null;
	const format = (value: number) => value.toLocaleString(undefined, { maximumSignificantDigits: 4 });
	$: samples = points.map((point, index) => ({
		x: numeric(point[mode]) ?? index + 1,
		train: numeric(point[metric]),
		validation: numeric(point[`val_${metric}`])
	}));
	$: series = (() => {
		const window: number[] = [];
		let total = 0;
		return samples.map((point) => {
			if (point.train !== null) {
				window.push(point.train);
				total += point.train;
				if (window.length > 7) total -= window.shift() as number;
			}
			return { x: point.x, primary: point.train === null ? null : mode === 'batch' ? total / window.length : point.train,
				secondary: mode === 'batch' ? point.train : null };
		});
	})();
	$: validationSeries = samples.filter((point): point is typeof point & { validation: number } => point.validation !== null);
	$: baseline = mode === 'batch' && !validationSeries.length ? numeric(validationValue) : null;
	$: values = [...series.flatMap((point) => [point.primary, point.secondary]), ...validationSeries.map((point) => point.validation), baseline].filter((value): value is number => value !== null);
	$: low = values.length ? Math.min(...values) : 0;
	$: high = values.length ? Math.max(...values) : 1;
	$: pad = (high - low) * .12 || Math.max(Math.abs(high) * .05, .01);
	$: xMin = series[0]?.x ?? 0;
	$: xMax = series.at(-1)?.x ?? 1;
	$: yAt = (value: number) => 158 - (value - low + pad) / (high - low + pad * 2) * 136;
	$: xAt = (value: number) => xMax === xMin ? 257 : 56 + (value - xMin) / (xMax - xMin) * 404;
	const line = (items: { x: number; primary: number | null; secondary: number | null }[], key: 'primary' | 'secondary', x: (value: number) => number, y: (value: number) => number) =>
		items.filter((point) => point[key] !== null).map((point, index) => `${index === 0 ? 'M' : 'L'}${x(point.x)},${y(point[key] as number)}`).join(' ');
	$: primaryPath = line(series, 'primary', xAt, yAt);
	$: secondaryPath = line(series, 'secondary', xAt, yAt);
	$: validationPath = validationSeries.map((point, index) => `${index === 0 ? 'M' : 'L'}${xAt(point.x)},${yAt(point.validation)}`).join(' ');
	$: primaryCount = series.filter((point) => point.primary !== null).length;
	$: secondaryCount = series.filter((point) => point.secondary !== null).length;
	$: validationCount = validationSeries.length;
</script>

<div class="paired-plot">
	{#if values.length}
		<svg viewBox="0 0 480 200" role="img" aria-label={`${title} by ${mode}. ${primaryCount} training samples; ${validationCount} validation samples${mode === 'batch' ? `; seven batch moving average${baseline === null ? '' : ` and ${validationLabel} reference`}` : ''}.`}>
			{#each [low, (low + high) / 2, high].filter((value, index, list) => list.indexOf(value) === index) as tick}
				<line class="grid" x1="56" x2="460" y1={yAt(tick)} y2={yAt(tick)} />
				<text x="47" y={yAt(tick) + 4} text-anchor="end">{format(tick)}</text>
			{/each}
			{#if secondaryCount}<path class:raw={mode === 'batch'} class="secondary" d={secondaryPath} />{/if}
			{#if validationCount}<path class="validation" d={validationPath} />{/if}
			{#if baseline !== null}<line class="validation-baseline" x1="56" x2="460" y1={yAt(baseline)} y2={yAt(baseline)}><title>{validationLabel}: {format(baseline)}</title></line>{/if}
			{#if primaryCount}<path class="primary" d={primaryPath} />{/if}
			{#if mode === 'epoch'}{#each series as point}{#if point.primary !== null}<circle class="train-point" cx={xAt(point.x)} cy={yAt(point.primary)} r="3"><title>Epoch {point.x} training: {format(point.primary)}</title></circle>{/if}{/each}{/if}
			{#each validationSeries as point}<circle class="validation-point" cx={xAt(point.x)} cy={yAt(point.validation)} r="3"><title>{mode === 'epoch' ? 'Epoch' : 'Batch'} {point.x} validation: {format(point.validation)}</title></circle>{/each}
			<text x="56" y="181">{xMin}</text><text x="460" y="181" text-anchor="end">{xMax}</text>
			<text x="258" y="193" text-anchor="middle">{mode === 'epoch' ? 'Epoch' : 'Batch'}</text>
		</svg>
		<div class="legend"><span><i class="primary-key"></i>{mode === 'epoch' ? 'Training' : 'Training · 7 batch average'}</span>{#if validationCount}<span><i class="secondary-key"></i>Validation · epoch end</span>{:else if mode === 'batch' && baseline !== null}<span><i class="secondary-key baseline-key"></i>{validationLabel}</span>{/if}{#if mode === 'batch' && secondaryCount}<span><i class="raw-key"></i>Raw training</span>{/if}<small>{primaryCount} {mode === 'epoch' ? 'completed epochs' : 'sampled batches'}</small></div>
	{:else}<p>No {mode} metrics yet.</p>{/if}
</div>

<style>
	.paired-plot { min-width:0; width:100%; }
	svg { display:block; width:100%; height:auto; overflow:visible; }
	.grid { stroke:#dce7e6; stroke-dasharray:3 5; }
	text { fill:#70858c; font-size:10px; font-family:inherit; font-weight:500; }
	.primary,.secondary,.validation { fill:none; stroke-linecap:round; stroke-linejoin:round; vector-effect:non-scaling-stroke; }
	.primary { stroke:var(--teal); stroke-width:2.8; }
	.secondary { stroke:#d08a35; stroke-width:2.4; }
	.validation { stroke:#d08a35; stroke-width:2.4; }
	.secondary.raw { stroke:#88aca4; stroke-width:1.3; opacity:.6; }
	.validation-baseline { stroke:#d08a35; stroke-width:2.2; stroke-dasharray:7 5; vector-effect:non-scaling-stroke; }
	.train-point { fill:var(--teal); }
	.validation-point { fill:#d08a35; }
	.legend { display:flex; flex-wrap:wrap; gap:.35rem .9rem; align-items:center; color:var(--muted); font-size:.66rem; }
	.legend span { display:inline-flex; align-items:center; gap:.3rem; white-space:nowrap; }
	.legend i { width:13px; height:3px; border-radius:2px; }
	.primary-key { background:var(--teal); }
	.secondary-key { background:#d08a35; }
	.baseline-key { background:repeating-linear-gradient(90deg,#d08a35 0 6px,transparent 6px 9px); }
	.raw-key { background:#88aca4; }
	.legend small { margin-left:auto; font-size:.63rem; }
	p { padding:2rem 0; color:var(--muted); font-size:.75rem; text-align:center; }
</style>
