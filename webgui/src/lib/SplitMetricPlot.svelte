<script lang="ts">
	import type { RecordValue } from '$lib/api';

	export let epochs: RecordValue[] = [];
	export let batches: RecordValue[] = [];
	export let metric = 'loss';
	export let title = 'Loss';
	export let currentEpoch: number | null = null;

	const numeric = (value: unknown): number | null => typeof value === 'number' && Number.isFinite(value) ? value : null;
	const format = (value: number) => value.toLocaleString(undefined, { maximumSignificantDigits: 4 });
	const plot = (points: { x: number; y: number }[], xAt: (x: number) => number, yAt: (y: number) => number) =>
		points.map((point, index) => `${index === 0 ? 'M' : 'L'}${xAt(point.x)},${yAt(point.y)}`).join(' ');
	$: epochTraining = epochs.flatMap((point) => {
		const x = numeric(point.epoch), y = numeric(point[metric]);
		return x === null || y === null ? [] : [{ x, y }];
	});
	$: epochValidation = epochs.flatMap((point) => {
		const x = numeric(point.epoch), y = numeric(point[`val_${metric}`]);
		return x === null || y === null ? [] : [{ x, y }];
	});
	$: batchTraining = batches.flatMap((point) => {
		const x = numeric(point.batch), y = numeric(point[metric]);
		return x === null || y === null ? [] : [{ x, y }];
	});
	$: batchSmoothed = (() => {
		let sum = 0;
		const window: number[] = [];
		return batchTraining.map((point) => {
			window.push(point.y);
			sum += point.y;
			if (window.length > 7) sum -= window.shift() as number;
			return { x: point.x, y: sum / window.length };
		});
	})();
	$: lastValidation = epochValidation.at(-1)?.y ?? null;
	$: values = [...epochTraining, ...epochValidation, ...batchTraining].map((point) => point.y);
	$: low = values.length ? Math.min(...values) : 0;
	$: high = values.length ? Math.max(...values) : 1;
	$: padding = (high - low) * .12 || Math.max(Math.abs(high) * .05, .01);
	$: yAt = (value: number) => 157 - (value - low + padding) / (high - low + padding * 2) * 130;
	$: epochMin = numeric(epochs[0]?.epoch) ?? 1;
	$: epochMax = numeric(epochs.at(-1)?.epoch) ?? epochMin;
	$: batchMin = batchTraining[0]?.x ?? 1;
	$: batchMax = batchTraining.at(-1)?.x ?? batchMin;
	$: epochX = (value: number) => epochMax === epochMin ? 320 : 70 + (value - epochMin) / (epochMax - epochMin) * 520;
	$: batchX = (value: number) => batchMax === batchMin ? 770 : 660 + (value - batchMin) / (batchMax - batchMin) * 224;
	$: epochTrainPath = plot(epochTraining, epochX, yAt);
	$: epochValidationPath = plot(epochValidation, epochX, yAt);
	$: batchTrainPath = plot(batchSmoothed, batchX, yAt);
	$: batchRawPath = plot(batchTraining, batchX, yAt);
</script>

<div class="split-plot">
	<svg viewBox="0 0 900 213" role="img" aria-label={`${title}: ${epochTraining.length} completed epoch training values, ${epochValidation.length} validation values, and ${batchTraining.length} training batches from ${currentEpoch === null ? 'the current epoch' : `epoch ${currentEpoch}`}. Training is blue; validation is orange. Both sections share one metric scale.`}>
		{#each [low, (low + high) / 2, high].filter((value, index, list) => list.indexOf(value) === index) as tick}
			<line class="grid" x1="70" x2="884" y1={yAt(tick)} y2={yAt(tick)} />
			<text class="y-tick" x="60" y={yAt(tick) + 4} text-anchor="end">{format(tick)}</text>
		{/each}
		<line class="axis-split" x1="625" x2="625" y1="20" y2="164" />
		<text class="section-label" x="330" y="16" text-anchor="middle">Completed epochs</text>
		<text class="section-label" x="772" y="16" text-anchor="middle">Current epoch {currentEpoch ?? ''} · batches</text>
		{#if epochTraining.length}<path class="training" d={epochTrainPath} />{/if}
		{#if epochValidation.length}<path class="validation" d={epochValidationPath} />{/if}
		{#if batchTraining.length}<path class="raw-training" d={batchRawPath} /><path class="training" d={batchTrainPath} />{/if}
		{#if lastValidation !== null}<line class="validation-reference" x1="660" x2="884" y1={yAt(lastValidation)} y2={yAt(lastValidation)}><title>Validation after epoch {epochMax}: {format(lastValidation)}</title></line>{/if}
		{#each epochTraining as point}<circle class="training-point" cx={epochX(point.x)} cy={yAt(point.y)} r="2.8"><title>Epoch {point.x} training: {format(point.y)}</title></circle>{/each}
		{#each epochValidation as point}<circle class="validation-point" cx={epochX(point.x)} cy={yAt(point.y)} r="2.8"><title>Epoch {point.x} validation: {format(point.y)}</title></circle>{/each}
		{#if !epochTraining.length && !epochValidation.length}<text class="empty" x="330" y="97" text-anchor="middle">Waiting for the first completed epoch</text>{/if}
		{#if !batchTraining.length}<text class="empty" x="772" y="97" text-anchor="middle">Waiting for batches</text>{/if}
		<text x="70" y="179">{epochTraining.length || epochValidation.length ? epochMin : ''}</text>
		<text x="590" y="179" text-anchor="end">{epochTraining.length || epochValidation.length ? epochMax : ''}</text>
		<text x="660" y="179">{batchTraining.length ? batchMin : ''}</text>
		<text x="884" y="179" text-anchor="end">{batchTraining.length ? batchMax : ''}</text>
		<text class="axis-title" x="330" y="197" text-anchor="middle">Epoch</text>
		<text class="axis-title" x="772" y="197" text-anchor="middle">Batch within current epoch</text>
	</svg>
	<div class="legend"><span><i class="training-key"></i>Training</span><span><i class="validation-key"></i>Validation · epoch end</span>{#if batchTraining.length}<span><i class="average-key"></i>Current epoch · 7 batch average</span>{/if}{#if lastValidation !== null}<span><i class="reference-key"></i>Latest validation reference</span>{/if}</div>
</div>

<style>
	.split-plot { min-width:0; width:100%; overflow-x:auto; }
	svg { display:block; width:100%; min-width:640px; height:auto; }
	.grid { stroke:#dce7e6; stroke-dasharray:3 5; }
	.axis-split { stroke:#9eb0b6; stroke-dasharray:4 5; }
	text { fill:#70858c; font-size:11px; font-family:inherit; font-weight:500; }
	.section-label { fill:#334850; font-size:11px; font-weight:700; }
	.axis-title { font-size:10px; }
	.empty { fill:#8a9ba2; font-size:10px; }
	.training,.validation,.raw-training { fill:none; stroke-linecap:round; stroke-linejoin:round; vector-effect:non-scaling-stroke; }
	.training { stroke:var(--teal); stroke-width:2.7; }
	.validation { stroke:#d08a35; stroke-width:2.5; }
	.raw-training { stroke:#88aca4; stroke-width:1; opacity:.5; }
	.validation-reference { stroke:#d08a35; stroke-width:2; stroke-dasharray:6 5; vector-effect:non-scaling-stroke; }
	.training-point { fill:var(--teal); }
	.validation-point { fill:#d08a35; }
	.legend { display:flex; flex-wrap:wrap; gap:.3rem .9rem; color:var(--muted); font-size:.68rem; }
	.legend span { display:inline-flex; align-items:center; gap:.3rem; }
	.legend i { width:14px; height:3px; border-radius:2px; }
	.training-key,.average-key { background:var(--teal); }
	.validation-key { background:#d08a35; }
	.reference-key { background:repeating-linear-gradient(90deg,#d08a35 0 6px,transparent 6px 9px); }
</style>
