<script lang="ts">
	import type { RecordValue } from '$lib/api';
	export let points: RecordValue[] = [];
	export let metric = 'loss';
	export let title = 'Loss';
	export let axis = 'Batch';
	export let expanded = false;
	let hovered: number | null = null;
	const numeric = (value: unknown): number | null => typeof value === 'number' && Number.isFinite(value) ? value : null;
	const format = (value: number) => value.toLocaleString(undefined, { maximumSignificantDigits: 4 });
	$: values = points.map((point, index) => ({ index, x: numeric(point[axis.toLowerCase()]) ?? index + 1, y: numeric(point[metric]) }));
	$: valid = values.filter((point): point is { index: number; x: number; y: number } => point.y !== null);
	$: low = valid.length ? Math.min(...valid.map((point) => point.y)) : 0;
	$: high = valid.length ? Math.max(...valid.map((point) => point.y)) : 1;
	$: pad = (high - low) * .12 || Math.max(Math.abs(high) * .05, .01);
	$: xMin = values[0]?.x ?? 0;
	$: xMax = values.at(-1)?.x ?? 1;
	$: yAt = (value: number) => 158 - (value - low + pad) / (high - low + pad * 2) * 136;
	$: xAt = (value: number) => xMax === xMin ? 257 : 56 + (value - xMin) / (xMax - xMin) * 404;
	$: path = values.map((point, index) => point.y === null ? '' : `${index === 0 || values[index - 1].y === null ? 'M' : 'L'}${xAt(point.x)},${yAt(point.y)}`).join(' ');
	$: inspected = valid.find((point) => point.index === hovered);
</script>

<div class="plot" class:expanded>
	{#if valid.length}
		<svg viewBox="0 0 480 200" role="img" aria-label={`${title} by ${axis.toLowerCase()}. ${valid.length} samples; minimum ${format(low)}, maximum ${format(high)}.`}>
			{#each [low, (low + high) / 2, high].filter((value, index, list) => list.indexOf(value) === index) as tick}
				<line x1="56" x2="460" y1={yAt(tick)} y2={yAt(tick)} class="grid" />
				<text x="47" y={yAt(tick) + 4} text-anchor="end">{format(tick)}</text>
			{/each}
			<path d={path} />
			{#each valid as point}
				<circle role="img" aria-label={`${axis} ${point.x}: ${format(point.y)}`} cx={xAt(point.x)} cy={yAt(point.y)} r={expanded ? 3 : 2} class:highlighted={hovered === point.index} on:mouseenter={() => hovered = point.index} on:mouseleave={() => hovered = null}>
					<title>{axis} {point.x}: {format(point.y)}</title>
				</circle>
			{/each}
			<text x="56" y="181">{xMin}</text><text x="460" y="181" text-anchor="end">{xMax}</text>
			<text x="258" y="193" text-anchor="middle">{axis}</text>
		</svg>
		<div class="plot-caption">{#if inspected}{axis} {inspected.x} · <strong>{format(inspected.y)}</strong>{:else}<span>Min {format(low)} · Max {format(high)}</span><span>{valid.length} samples</span>{/if}</div>
	{:else}<p>No {title.toLowerCase()} samples yet.</p>{/if}
</div>

<style>
	.plot { min-width:0; width:100%; }
	svg { display:block; width:100%; height:auto; overflow:visible; }
	.grid { stroke:#dce7e6; stroke-dasharray:3 5; }
	text { fill:#70858c; font-size:10px; font-family:inherit; font-weight:500; }
	path { fill:none; stroke:var(--teal); stroke-width:2.3; stroke-linejoin:round; stroke-linecap:round; vector-effect:non-scaling-stroke; }
	circle { fill:var(--teal); stroke:var(--surface); stroke-width:1.5; }
	circle.highlighted { fill:#c18529; r:6; }
	.plot-caption { display:flex; justify-content:space-between; gap:.5rem; min-height:1.3rem; color:var(--muted); font-size:.68rem; font-weight:500; }
	.plot-caption strong { color:var(--teal); }
	p { padding:2rem 0; font-size:.8rem; text-align:center; }
	.expanded { padding:.5rem 0; }
</style>
