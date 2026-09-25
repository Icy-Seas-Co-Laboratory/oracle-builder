<script lang="ts">
	import { onDestroy, onMount } from 'svelte';
	import { api, type RecordValue } from '$lib/api';

	export let job: RecordValue;
	export let runName = '';
	export let onclose: () => void;

	type Point = { epoch: number; value: number };
	const activeStates = new Set(['queued', 'submitted', 'running', 'paused', 'training', 'validating', 'sealing', 'indexing']);
	const stages = ['Preparing', 'Training', 'Validation', 'Sealing'];
	const records = (value: unknown): RecordValue[] => Array.isArray(value) ? value as RecordValue[] : [];
	const record = (value: unknown): RecordValue => value && typeof value === 'object' && !Array.isArray(value) ? value as RecordValue : {};
	const number = (value: unknown): number | null => typeof value === 'number' && Number.isFinite(value) ? value : typeof value === 'string' && value.trim() !== '' && Number.isFinite(Number(value)) ? Number(value) : null;
	const text = (value: unknown, fallback = '—') => value == null || value === '' ? fallback : String(value);
	const label = (value: string) => value.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
	const formatNumber = (value: unknown, digits = 4) => { const parsed = number(value); return parsed == null ? '—' : parsed.toLocaleString(undefined, { maximumFractionDigits: digits }); };
	const formatDuration = (value: unknown) => {
		const seconds = number(value); if (seconds == null || seconds < 0) return '—';
		const hours = Math.floor(seconds / 3600), minutes = Math.floor((seconds % 3600) / 60), remainder = Math.floor(seconds % 60);
		return hours ? `${hours}h ${minutes}m` : minutes ? `${minutes}m ${remainder}s` : `${remainder}s`;
	};

	let status: RecordValue | null = null;
	let jobEvents: RecordValue[] = [];
	let loading = true;
	let error = '';
	let lastUpdated = 0;
	let timer: ReturnType<typeof setTimeout> | undefined;
	let disposed = false;
	let browserLiveTrace: RecordValue[] = [];
	let controlBusy = false;
	let controlError = '';

	const endpoint = api as typeof api & { jobTrainingStatus: (id: string) => Promise<RecordValue> };
	$: jobId = String(job?.job_id ?? job?.id ?? '');
	$: jobStatus = String(status?.lifecycle ?? status?.job_status ?? status?.status ?? job?.status ?? '').toLowerCase();
	$: active = activeStates.has(jobStatus);
	$: progress = record(status?.progress);
	$: currentStage = String(status?.stage ?? status?.phase ?? progress.stage ?? jobStatus ?? 'Preparing');
	$: normalizedStage = currentStage.toLowerCase();
	$: stageIndex = normalizedStage.includes('valid') ? 2 : normalizedStage.includes('seal') || normalizedStage.includes('index') ? 3 : normalizedStage.includes('train') || active ? 1 : 0;
	$: epoch = number(status?.epoch ?? progress.epoch ?? status?.current_epoch);
	$: totalEpochs = number(status?.total_epochs ?? progress.total_epochs ?? status?.epochs);
	$: batch = number(status?.batch ?? progress.batch ?? status?.current_batch);
	$: totalBatches = number(status?.total_batches ?? progress.total_batches ?? status?.batches_per_epoch);
	$: percent = number(status?.percent ?? progress.percent ?? progress.progress) ?? (batch != null && totalBatches ? 100 * batch / totalBatches : epoch != null && totalEpochs ? 100 * epoch / totalEpochs : 0);
	$: history = records(status?.history ?? status?.metric_history ?? status?.epoch_metrics ?? status?.metrics);
	$: events = (records(status?.events ?? status?.activity).length ? records(status?.events ?? status?.activity) : jobEvents).slice(-8).reverse();
	$: config = record(status?.config ?? status?.configuration ?? status?.run_config);
	$: latest = record(status?.latest_metrics ?? status?.metrics_latest ?? history[history.length - 1]);
	$: liveMetrics = record(record(status?.metrics).current_batch ?? status?.current_batch_metrics);
	$: completedMetrics = { ...record(history[history.length - 1]), ...record(record(status?.metrics).last_completed_epoch ?? latest) };
	$: serverLiveTrace = records(record(status?.metrics).current_epoch_history ?? status?.current_epoch_history);
	$: liveTrace = serverLiveTrace.length ? serverLiveTrace : browserLiveTrace;
	$: watchlist = record(status?.watchlist ?? status?.monitoring);
	$: configuredPrimaryKey = String(watchlist.primary_metric ?? status?.primary_metric ?? status?.monitor_metric ?? 'auto');
	$: availableCompletedMetricKeys = new Set([...Object.keys(completedMetrics), ...history.flatMap((row) => Object.keys(row))]);
	$: automaticPrimaryKey = ['val_macro_f1', 'val_f1', 'val_accuracy', 'val_dice', 'val_iou', 'val_loss', 'macro_f1', 'accuracy', 'loss'].find((key) => availableCompletedMetricKeys.has(key)) ?? '';
	$: primaryKey = configuredPrimaryKey && configuredPrimaryKey !== 'auto' ? configuredPrimaryKey : automaticPrimaryKey;
	$: liveCards = metricCards(liveMetrics, 5);
	$: completedCards = metricCards(completedMetrics, 4);
	$: cards = completedCards;
	$: snapshotUpdatedAt = Date.parse(String(status?.updated_at ?? '')) || lastUpdated;
	$: stale = Boolean(status) && (Boolean(status?.stale) || Date.now() - snapshotUpdatedAt > 8_000);
	$: controls = records(status?.controls).map(String);
	$: canPause = controls.includes('pause');
	$: canResume = controls.includes('resume');
	$: canCancel = controls.includes('cancel');

	function metricCards(source: RecordValue, limit = 4): [string, unknown][] {
		const preferred = ['loss', 'accuracy', 'macro_f1', 'val_loss', primaryKey, 'val_accuracy', 'val_macro_f1', 'learning_rate'];
		const selected: [string, unknown][] = [];
		for (const key of preferred) if (key && source[key] != null && !selected.some(([existing]) => existing === key)) selected.push([key, source[key]]);
		for (const [key, value] of Object.entries(source)) {
			if (selected.length >= limit) break;
			if (number(value) != null && !['epoch', 'step', 'batch'].includes(key) && !selected.some(([existing]) => existing === key)) selected.push([key, value]);
		}
		return selected.slice(0, limit);
	}
	function points(key: string): Point[] {
		return history.map((row, index) => ({ epoch: number(row.epoch) ?? index + 1, value: number(row[key]) ?? NaN })).filter((point) => Number.isFinite(point.value));
	}
	function chartPoints(values: Point[]) {
		if (!values.length) return '';
		const low = Math.min(...values.map((item) => item.value)), high = Math.max(...values.map((item) => item.value)), spread = high - low || 1;
		return values.map((item, index) => `${values.length === 1 ? 50 : index / (values.length - 1) * 100},${94 - ((item.value - low) / spread) * 82}`).join(' ');
	}
	function chartRange(values: Point[]) {
		if (!values.length) return 'Waiting for completed epochs';
		const numeric = values.map((item) => item.value);
		return `${formatNumber(Math.min(...numeric))} – ${formatNumber(Math.max(...numeric))}`;
	}
	function metricDirection(key: string): 'higher' | 'lower' | 'neutral' {
		if (/(^|_)(loss|error|mae|mse|rmse|mape|nll|perplexity)(_|$)/i.test(key)) return 'lower';
		if (/(^|_)(learning_rate|lr)(_|$)/i.test(key)) return 'neutral';
		return 'higher';
	}
	function metricTone(key: string, value: unknown, baseline: unknown): 'improving' | 'attention' | 'steady' | 'unknown' {
		const current = number(value), prior = number(baseline);
		if (current == null || prior == null || metricDirection(key) === 'neutral') return 'unknown';
		const tolerance = Math.max(Math.abs(prior) * 0.002, 0.0001);
		if (Math.abs(current - prior) <= tolerance) return 'steady';
		const improving = metricDirection(key) === 'lower' ? current < prior : current > prior;
		return improving ? 'improving' : 'attention';
	}
	function metricTrendSymbol(value: unknown, baseline: unknown) {
		const current = number(value), prior = number(baseline);
		return current == null || prior == null ? '' : current > prior ? '↑' : current < prior ? '↓' : '→';
	}
	function metricTrendIndicator(key: string, value: unknown, baseline: unknown) {
		return metricTone(key, value, baseline) === 'steady' ? '→' : metricTrendSymbol(value, baseline);
	}
	function metricTrendText(key: string, value: unknown, baseline: unknown, reference: string) {
		const tone = metricTone(key, value, baseline);
		if (tone === 'unknown') return reference === 'batch' ? 'current batch / running' : `epoch ${Math.max(0, (epoch ?? 1) - 1)}`;
		if (tone === 'steady') return `steady vs prior ${reference}`;
		return `${metricTrendSymbol(value, baseline)} ${tone === 'improving' ? 'improving' : 'needs attention'} vs prior ${reference}`;
	}
	function priorLiveValue(key: string) {
		const values = tracePoints(key);
		return values.length > 1 ? values[values.length - 2].value : null;
	}
	function priorCompletedValue(key: string) {
		return history.length > 1 ? number(history[history.length - 2][key]) : null;
	}
	function tracePoints(key: string): Point[] {
		return liveTrace.map((row, index) => ({ epoch: number(row.batch) ?? index + 1, value: number(row[key]) ?? NaN })).filter((point) => Number.isFinite(point.value));
	}
	function liveTrendKeys(): string[] {
		const keys = new Set(liveTrace.flatMap((row) => Object.keys(row)));
		return ['loss', 'accuracy', 'macro_f1'].filter((key) => keys.has(key));
	}
	function allMetricKeys(): string[] {
		const available = new Set([...Object.keys(liveMetrics), ...Object.keys(completedMetrics), ...history.flatMap((row) => Object.keys(row))]);
		const excluded = new Set(['epoch', 'batch', 'step', 'size', 'learning_rate']);
		const preferred = ['loss', 'accuracy', 'macro_f1', 'precision', 'recall', 'val_loss', 'val_accuracy', 'val_macro_f1', 'val_precision', 'val_recall'];
		return [...preferred.filter((key) => available.has(key)), ...[...available].filter((key) => !excluded.has(key) && !preferred.includes(key)).sort()].slice(0, 16);
	}
	function chartKeys(): string[] {
		const available = new Set(history.flatMap((row) => Object.keys(row)).filter((key) => number(history.find((row) => row[key] != null)?.[key]) != null));
		const preferred = ['loss', 'accuracy', 'macro_f1', 'val_loss', 'val_accuracy', 'val_macro_f1'];
		return [...preferred.filter((key) => available.has(key)), ...[...available].filter((key) => !['epoch', ...preferred].includes(key)).sort()].slice(0, 6);
	}
	function bestMetricValue(key: string): number | null {
		const values = points(key).map((point) => point.value);
		if (!values.length) return number(completedMetrics[key]);
		return metricDirection(key) === 'lower' ? Math.min(...values) : Math.max(...values);
	}
	function validationGap(): { metric: string; value: number } | null {
		const candidates = [primaryKey, 'val_macro_f1', 'val_accuracy', 'val_loss'];
		for (const validation of candidates.filter((key, index, keys) => key.startsWith('val_') && keys.indexOf(key) === index)) {
			const training = validation.slice(4);
			const trainValue = number(completedMetrics[training]), validationValue = number(completedMetrics[validation]);
			if (trainValue == null || validationValue == null) continue;
			return { metric: training, value: metricDirection(validation) === 'lower' ? validationValue - trainValue : trainValue - validationValue };
		}
		return null;
	}
	function healthSignals(): { tone: 'good' | 'attention' | 'pending'; label: string; detail: string }[] {
		const signals: { tone: 'good' | 'attention' | 'pending'; label: string; detail: string }[] = [];
		const targetMetric = String(watchlist.target_metric ?? primaryKey);
		const targetValue = number(watchlist.target_value);
		const completedValue = number(completedMetrics[targetMetric]);
		if (watchlist.target_enabled) {
			if (targetValue == null || completedValue == null) signals.push({ tone: 'pending', label: 'Target', detail: `Waiting for ${label(targetMetric)}` });
			else {
				const passed = metricDirection(targetMetric) === 'lower' ? completedValue <= targetValue : completedValue >= targetValue;
				signals.push({ tone: passed ? 'good' : 'attention', label: 'Target', detail: `${label(targetMetric)} ${passed ? 'meets' : 'has not met'} ${formatNumber(targetValue)}` });
			}
		}
		const validationLoss = number(completedMetrics.val_loss), bestValidationLoss = bestMetricValue('val_loss'), permittedRise = number(watchlist.max_validation_loss_increase);
		if (watchlist.max_validation_loss_increase_enabled) {
			if (validationLoss == null || bestValidationLoss == null || permittedRise == null) signals.push({ tone: 'pending', label: 'Validation loss', detail: 'Waiting for validation evidence' });
			else {
				const rise = validationLoss - bestValidationLoss, passed = rise <= permittedRise;
				signals.push({ tone: passed ? 'good' : 'attention', label: 'Validation loss', detail: passed ? `within +${formatNumber(permittedRise)}` : `+${formatNumber(rise)} above best` });
			}
		}
		const gap = validationGap(), maxGap = number(watchlist.max_generalization_gap);
		if (watchlist.max_generalization_gap_enabled) {
			if (!gap || maxGap == null) signals.push({ tone: 'pending', label: 'Generalization gap', detail: 'Waiting for paired validation metric' });
			else signals.push({ tone: gap.value <= maxGap ? 'good' : 'attention', label: 'Generalization gap', detail: `${formatNumber(gap.value)} ${gap.value <= maxGap ? 'within' : 'over'} ${formatNumber(maxGap)}` });
		}
		return signals;
	}
	function healthTone() { return healthSignals().some((signal) => signal.tone === 'attention') ? 'attention' : healthSignals().some((signal) => signal.tone === 'pending') ? 'pending' : 'good'; }
	function etaConfidence() { return history.length >= 3 ? 'Established' : history.length ? 'Calibrating' : 'Pending'; }
	function completedPairs(): { training: string; validation: string | null }[] {
		const available = new Set(history.flatMap((row) => Object.keys(row)).filter((key) => key !== 'epoch'));
		const bases = ['loss', 'accuracy', 'macro_f1', 'precision', 'recall', 'dice', 'iou'];
		const pairs = bases.filter((base) => available.has(base) || available.has(`val_${base}`)).map((base) => ({ training: base, validation: available.has(`val_${base}`) ? `val_${base}` : null }));
		return [...pairs, ...[...available].filter((key) => !key.startsWith('val_') && !bases.includes(key)).sort().map((key) => ({ training: key, validation: available.has(`val_${key}`) ? `val_${key}` : null }))].slice(0, 6);
	}
	function pairedChartPoints(primary: Point[], secondary: Point[]) {
		const values = [...primary, ...secondary];
		if (!values.length) return { primary: '', secondary: '' };
		const low = Math.min(...values.map((item) => item.value)), high = Math.max(...values.map((item) => item.value)), spread = high - low || 1;
		const render = (series: Point[]) => series.map((item, index) => `${series.length === 1 ? 50 : index / (series.length - 1) * 100},${94 - ((item.value - low) / spread) * 82}`).join(' ');
		return { primary: render(primary), secondary: render(secondary) };
	}
	function stageDone(index: number) { return index < stageIndex; }
	function stageActive(index: number) { return index === stageIndex; }
	function close(event?: KeyboardEvent) { if (!event || event.key === 'Escape') onclose?.(); }
	function backdrop(event: MouseEvent) { if (event.target === event.currentTarget) onclose?.(); }
	async function control(action: 'pause' | 'resume' | 'cancel') {
		if (!jobId || controlBusy) return;
		if (action === 'cancel' && !window.confirm('Cancel this training run? The active process will stop and this run will not be resumable from the live dashboard.')) return;
		controlBusy = true; controlError = '';
		try {
			if (action === 'pause') await api.pauseJob(jobId);
			else if (action === 'resume') await api.resumeJob(jobId);
			else await api.cancelJob(jobId);
			await load();
		} catch (caught) { controlError = caught instanceof Error ? caught.message : `Could not ${action} this run.`; }
		finally { controlBusy = false; }
	}
	function rowsFromHistory(value: unknown): RecordValue[] {
		const series = record(value);
		const entries = Object.entries(series).filter(([, values]) => Array.isArray(values));
		const length = Math.max(0, ...entries.map(([, values]) => (values as unknown[]).length));
		return Array.from({ length }, (_, index) => Object.fromEntries([
			['epoch', index + 1],
			...entries.map(([name, values]) => [name, (values as unknown[])[index]])
		]));
	}
	function normalizedStatus(result: RecordValue): RecordValue {
		const outer = record(result.status ?? result.training_status ?? result);
		const snapshot = record(outer.snapshot);
		const progressData = record(snapshot.progress);
		const timing = record(snapshot.timing);
		const metrics = record(snapshot.metrics);
		return {
			...outer,
			...snapshot,
			progress: progressData,
			batch: progressData.completed_batches ?? progressData.batch,
			elapsed_seconds: timing.elapsed_seconds ?? snapshot.elapsed_seconds,
			batches_per_second: timing.batches_per_second ?? snapshot.batches_per_second,
			total_eta_seconds: timing.total_eta_seconds ?? snapshot.total_eta_seconds,
			latest_metrics: metrics.last_completed_epoch ?? snapshot.latest_metrics,
			current_batch_metrics: metrics.current_batch ?? snapshot.current_batch_metrics,
			current_epoch_history: metrics.current_epoch_history ?? snapshot.current_epoch_history,
			history: rowsFromHistory(metrics.history ?? snapshot.history)
		};
	}
	function retainBrowserLiveSample(next: RecordValue) {
		const nextProgress = record(next.progress);
		const nextBatch = number(next.batch ?? nextProgress.completed_batches ?? nextProgress.batch);
		const nextMetrics = record(record(next.metrics).current_batch ?? next.current_batch_metrics);
		if (nextBatch == null || !Object.keys(nextMetrics).length) return;
		const previousBatch = number(browserLiveTrace[browserLiveTrace.length - 1]?.batch);
		if (previousBatch === nextBatch) return;
		const priorEpochTrace = previousBatch != null && nextBatch < previousBatch ? [] : browserLiveTrace;
		browserLiveTrace = [...priorEpochTrace, { batch: nextBatch, ...nextMetrics }].slice(-360);
	}
	async function load() {
		if (!jobId || disposed) return;
		try {
			const [result, eventResult] = await Promise.all([
				endpoint.jobTrainingStatus(jobId),
				api.jobEvents(jobId).catch(() => ({ events: [] as RecordValue[] }))
			]);
			if (disposed) return;
			// Oracle Serve returns the callback document under snapshot; retain outer
			// availability/lifecycle fields so an unavailable worker is still intelligible.
			const nextStatus = normalizedStatus(result);
			status = nextStatus;
			retainBrowserLiveSample(nextStatus);
			jobEvents = records(eventResult.events);
			lastUpdated = Date.now(); error = '';
		} catch (caught) {
			if (!disposed) error = caught instanceof Error ? caught.message : 'Live training status is unavailable.';
		} finally {
			if (!disposed) { loading = false; if (active) timer = setTimeout(() => void load(), 1500); }
		}
	}
	onMount(() => { void load(); });
	onDestroy(() => { disposed = true; if (timer) clearTimeout(timer); });
</script>

<svelte:window on:keydown={close} />

<div class="training-status-backdrop" role="presentation" on:click={backdrop}>
	<div class="training-status-modal" role="dialog" aria-modal="true" aria-labelledby="training-status-title" tabindex="-1">
		<header class="modal-header">
			<div><p class="eyebrow">LIVE TRAINING RUN</p><h2 id="training-status-title">{runName || text(job?.name ?? job?.specification_name, 'Training run')}</h2><p><span class:active class="live-status">{label(jobStatus || 'starting')}</span> · {text(status?.worker_id ?? job?.worker_id, 'Awaiting worker')} · elapsed {formatDuration(status?.elapsed_seconds ?? status?.runtime_seconds)}</p></div>
			<div class="modal-actions">{#if canPause}<button class="secondary small" disabled={controlBusy} on:click={() => control('pause')}>{controlBusy ? 'Working…' : 'Pause'}</button>{/if}{#if canResume}<button class="secondary small" disabled={controlBusy} on:click={() => control('resume')}>{controlBusy ? 'Working…' : 'Resume'}</button>{/if}{#if canCancel}<button class="quiet-button small danger" disabled={controlBusy} on:click={() => control('cancel')}>{controlBusy ? 'Working…' : 'Cancel run'}</button>{/if}<button class="secondary small close-button" on:click={() => onclose?.()} aria-label="Close training dashboard">Close</button></div>
		</header>

		{#if loading && !status}<div class="dashboard-message" aria-live="polite">Connecting to the live training status…</div>
		{:else if error && !status}<div class="dashboard-message error" role="alert"><strong>Live status unavailable</strong><span>{error}</span></div>
		{:else if status?.available === false}<div class="dashboard-message error" role="status"><strong>Live training status is not available</strong><span>{text(status.message, 'The worker has not published a training snapshot yet.')}</span></div>
		{:else if status}
			<div class="stage-rail" aria-label={`Current stage: ${currentStage}`}>
				{#each stages as stage, index}<div class:done={stageDone(index)} class:current={stageActive(index)}><i></i><span>{stage}</span></div>{/each}
			</div>
			<div class="run-at-glance">
			<section class="progress-panel">
				<div><strong>{epoch != null ? `Epoch ${epoch}${totalEpochs ? ` / ${totalEpochs}` : ''}` : label(currentStage)}</strong><span>{batch != null ? `${batch}${totalBatches ? ` / ${totalBatches}` : ''} batches` : 'Awaiting batch progress'}</span></div>
				<div class="progress-track" aria-label={`${Math.round(percent)} percent complete`}><i style={`width: ${Math.max(0, Math.min(100, percent))}%`}></i></div>
				<div class="progress-summary"><strong>{Math.round(percent)}%</strong><span>ETA {formatDuration(status?.eta_seconds ?? progress.eta_seconds ?? status?.total_eta_seconds)}</span></div>
			</section>
			<div class="operational-line"><span>{formatNumber(status?.batches_per_second ?? status?.throughput_batches_per_second, 2)} batch/s</span><span>{formatNumber(status?.samples_per_second ?? status?.throughput_samples_per_second, 0)} samples/s</span><span>LR {formatNumber(status?.learning_rate ?? latest.learning_rate, 7)}</span><span class:stale>Last update {stale ? 'more than 8s ago' : 'just now'}</span></div>
			{#if status?.external_analysis}<div class="analysis-indicator"><strong>Post-epoch analysis</strong><span>{text(record(status.external_analysis).phase)} · {text(record(status.external_analysis).completed_batches, '0')}{#if record(status.external_analysis).total_batches} / {text(record(status.external_analysis).total_batches)} batches{/if}</span></div>{/if}
			<section class={`run-health ${healthTone()}`} aria-label="Run health summary"><header><div><p class="eyebrow">RUN HEALTH</p><h3>{healthTone() === 'attention' ? 'Watchlist needs attention' : healthTone() === 'pending' ? 'Evidence is still calibrating' : 'Training is on track'}</h3></div><span class="health-state">{healthTone() === 'attention' ? 'WATCH' : healthTone() === 'pending' ? 'LEARNING' : 'ON TRACK'}</span></header><div class="health-grid"><article><span>Primary metric</span><strong>{primaryKey ? formatNumber(completedMetrics[primaryKey]) : '—'}</strong><small>{primaryKey ? label(primaryKey) : 'Waiting for a completed epoch'}</small></article><article><span>Best validation</span><strong>{primaryKey ? formatNumber(bestMetricValue(primaryKey)) : '—'}</strong><small>{primaryKey ? `best ${label(primaryKey)}` : 'No baseline yet'}</small></article><article><span>Generalization gap</span><strong>{validationGap() ? formatNumber(validationGap()?.value) : '—'}</strong><small>{validationGap() ? label(validationGap()?.metric ?? '') : 'Waiting for paired metric'}</small></article><article><span>ETA confidence</span><strong>{etaConfidence()}</strong><small>{history.length ? `${history.length} completed ${history.length === 1 ? 'epoch' : 'epochs'}` : 'Needs completed epochs'}</small></article></div>{#if healthSignals().length}<div class="watchlist-signals">{#each healthSignals() as signal}<span class={signal.tone}><strong>{signal.label}</strong>{signal.detail}</span>{/each}</div>{:else}<p class="watchlist-empty">No thresholds are enabled. Add sealed targets in Construction → Live monitoring.</p>{/if}</section>
			</div>

			<div class="metric-overview-grid">
			<section class="live-epoch-panel"><header><div><p class="eyebrow">CURRENT EPOCH SIGNAL</p><h3>Batch {batch ?? '—'} · live training values</h3><p>These are transient batch-level values, useful for spotting instability before validation completes.</p></div><div class="epoch-timing"><span>Epoch elapsed<strong>{formatDuration(record(status?.timing).epoch_elapsed_seconds)}</strong></span><span>Epoch ETA<strong>{formatDuration(record(status?.timing).epoch_eta_seconds)}</strong></span><span>Run ETA<strong>{formatDuration(status?.total_eta_seconds)}</strong></span></div></header>{#if liveCards.length}<div class="metric-cards live">{#each liveCards as [key, value]}<article class={metricTone(key, value, priorLiveValue(key))}><span>{label(key)}</span><strong>{formatNumber(value, 4)}</strong><small class={`metric-trend ${metricTone(key, value, priorLiveValue(key))}`}>{metricTrendText(key, value, priorLiveValue(key), 'batch')}</small></article>{/each}</div>{:else}<p class="no-live-signal">Waiting for the first optimizer update to publish live metrics.</p>{/if}</section>

			{#if completedCards.length}<section class="completed-summary"><header><div><p class="eyebrow">LAST COMPLETED EPOCH</p><h3>Validated epoch evidence</h3></div><p>Green means better than the preceding completed epoch; amber calls out a reversal.</p></header><div class="metric-cards compact">{#each completedCards as [key, value]}<article class={metricTone(key, value, priorCompletedValue(key))}><span>{label(key)}</span><strong>{formatNumber(value, key.includes('learning') ? 7 : 4)}</strong>{#if key === primaryKey && status?.best_metric != null}<small>best {formatNumber(status.best_metric)}</small>{:else}<small class={`metric-trend ${metricTone(key, value, priorCompletedValue(key))}`}>{metricTrendText(key, value, priorCompletedValue(key), 'epoch')}</small>{/if}</article>{/each}</div></section>{/if}
			</div>

			<div class="dashboard-grid">
				<section class="charts"><header><div><h3>Current epoch trend</h3><p>Live values across the current epoch.</p></div><span>{liveTrace.length ? `${liveTrace.length} observations` : 'Current snapshot'}</span></header><div class="chart-grid">{#each liveTrendKeys() as key}<article class="metric-chart"><div><strong>{label(key)}</strong><small>{chartRange(tracePoints(key))}</small></div>{#if tracePoints(key).length > 1}<svg viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label={`${label(key)} across current epoch batches`}><line x1="0" y1="94" x2="100" y2="94"/><line x1="0" y1="50" x2="100" y2="50"/><polyline points={chartPoints(tracePoints(key))}/></svg>{:else if liveMetrics[key] != null}<div class="chart-single"><strong>{formatNumber(liveMetrics[key])}</strong><span>Current batch</span></div>{/if}</article>{:else}<p class="no-chart">Current-epoch trends begin after the next status update.</p>{/each}</div></section>
				<details class="activity dashboard-details"><summary><span>Activity</span><small>{events.length ? `${events.length} recent events` : 'No events yet'}</small></summary>{#if events.length}<ol>{#each events as event}<li><i class={String(event.level ?? event.event_type ?? '')}></i><div><strong>{text(event.message ?? event.title)}</strong><small>{text(event.timestamp ?? event.created_at)}</small></div></li>{/each}</ol>{:else}<p>No training events received yet.</p>{/if}</details>
			</div>
			<details class="completed-trends dashboard-details"><summary><span><strong>Completed epoch trends</strong><small>Paired training / validation</small></span><small>{history.length} completed {history.length === 1 ? 'epoch' : 'epochs'}</small></summary><div class="chart-grid">{#each completedPairs() as pair}{@const training = points(pair.training)}{@const validation = pair.validation ? points(pair.validation) : []}{@const lines = pairedChartPoints(training, validation)}<article class="metric-chart paired"><div><strong>{label(pair.training)}</strong><small>{pair.validation ? 'Train / validation' : chartRange(training)}</small></div>{#if training.length > 1 || validation.length > 1}<div class="chart-legend"><span class="train">Train</span>{#if pair.validation}<span class="validation">Validation</span>{/if}</div><svg viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label={`${label(pair.training)}${pair.validation ? ` and ${label(pair.validation)}` : ''} by completed epoch`}><line x1="0" y1="94" x2="100" y2="94"/><line x1="0" y1="50" x2="100" y2="50"/>{#if lines.primary}<polyline class="train-line" points={lines.primary}/>{/if}{#if lines.secondary}<polyline class="validation-line" points={lines.secondary}/>{/if}</svg>{:else if training.length || validation.length}<div class="chart-single paired-values"><span><strong>{training.length ? formatNumber(training[0].value) : '—'}</strong>Train</span>{#if pair.validation}<span><strong>{validation.length ? formatNumber(validation[0].value) : '—'}</strong>Validation</span>{/if}</div>{:else}<p>Not recorded.</p>{/if}</article>{:else}<p class="no-chart">Completed epoch trends appear after the first validation pass.</p>{/each}</div></details>
			{#if allMetricKeys().length}<details class="metric-matrix dashboard-details"><summary><span><strong>Metric ledger</strong><small>Every reported metric</small></span><small>{allMetricKeys().length} metrics</small></summary><div class="metric-table-scroll"><table><thead><tr><th>Metric</th><th>Current batch</th><th>Last completed epoch</th></tr></thead><tbody>{#each allMetricKeys() as key}<tr><th>{label(key)}</th><td class:missing={liveMetrics[key] == null} class={metricTone(key, liveMetrics[key], priorLiveValue(key))}>{formatNumber(liveMetrics[key], key.includes('learning') ? 7 : 5)}{#if metricTone(key, liveMetrics[key], priorLiveValue(key)) !== 'unknown'}<small>{metricTrendIndicator(key, liveMetrics[key], priorLiveValue(key))}</small>{/if}</td><td class:missing={completedMetrics[key] == null} class={metricTone(key, completedMetrics[key], priorCompletedValue(key))}>{formatNumber(completedMetrics[key], key.includes('learning') ? 7 : 5)}{#if metricTone(key, completedMetrics[key], priorCompletedValue(key)) !== 'unknown'}<small>{metricTrendIndicator(key, completedMetrics[key], priorCompletedValue(key))}</small>{/if}</td></tr>{/each}</tbody></table></div></details>{/if}
			<details class="configuration dashboard-details"><summary><span><strong>Run configuration</strong><small>Sealed execution evidence</small></span><small>Show details</small></summary><div>{#each Object.entries(config).slice(0, 8) as [key, value]}<span>{label(key)}<strong>{typeof value === 'object' ? JSON.stringify(value) : text(value)}</strong></span>{:else}<span>Job ID<strong>{jobId}</strong></span><span>Output<strong>{text(job?.output_path)}</strong></span>{/each}</div></details>
			{#if error}<p class="refresh-error" role="status">Could not refresh: {error}. Showing the most recent status.</p>{/if}
			{#if controlError}<p class="refresh-error" role="alert">{controlError}</p>{/if}
		{/if}
	</div>
</div>

<style>
	.training-status-backdrop{position:fixed;z-index:50;inset:0;display:grid;place-items:center;padding:2rem;background:rgb(15 23 42 / .58);backdrop-filter:blur(3px)}.training-status-modal{width:min(1220px,96vw);max-height:92vh;overflow:auto;padding:1.4rem;border:1px solid #cbd5e1;border-radius:12px;background:#fff;color:var(--ink,#172033);box-shadow:0 26px 70px rgb(15 23 42 / .34)}.modal-header{display:flex;justify-content:space-between;gap:1rem;padding-bottom:1rem;border-bottom:1px solid var(--line,#e2e8f0)}.modal-header h2{margin:.12rem 0 .35rem;font-size:1.35rem}.modal-header p{margin:0;color:var(--muted,#64748b);font-size:.78rem}.close-button{align-self:start}.live-status{display:inline-flex;align-items:center;gap:.3rem;font-weight:750;text-transform:capitalize}.live-status.active:before{width:.45rem;height:.45rem;border-radius:50%;background:#16a34a;content:''}.dashboard-message{padding:3rem 1rem;text-align:center;color:var(--muted,#64748b)}.dashboard-message.error{color:#b91c1c}.dashboard-message strong,.dashboard-message span{display:block}.stage-rail{display:grid;grid-template-columns:repeat(4,1fr);margin:1.25rem 0}.stage-rail div{position:relative;display:grid;justify-items:center;gap:.4rem;color:var(--muted,#64748b);font-size:.7rem;font-weight:750;text-transform:uppercase}.stage-rail div:before{position:absolute;z-index:0;top:.35rem;left:-50%;width:100%;height:2px;background:#dbe3ed;content:''}.stage-rail div:first-child:before{display:none}.stage-rail i{z-index:1;width:.75rem;height:.75rem;border:2px solid #cbd5e1;border-radius:50%;background:#fff}.stage-rail .done,.stage-rail .current{color:var(--teal,#0f766e)}.stage-rail .done:before,.stage-rail .current:before{background:#5eead4}.stage-rail .done i{border-color:#0f766e;background:#0f766e}.stage-rail .current i{border-color:#0f766e;box-shadow:0 0 0 4px #ccfbf1}.progress-panel{display:grid;grid-template-columns:minmax(10rem,1fr) minmax(10rem,2fr) auto;align-items:center;gap:1rem;padding:1rem;background:#f8fafc;border:1px solid var(--line,#e2e8f0);border-radius:8px}.progress-panel div:first-child strong,.progress-panel div:first-child span,.progress-summary strong,.progress-summary span{display:block}.progress-panel span,.operational-line,.metric-cards span,.metric-cards small,.configuration span{color:var(--muted,#64748b);font-size:.7rem}.progress-track{height:.55rem;overflow:hidden;border-radius:99px;background:#dbeafe}.progress-track i{display:block;height:100%;border-radius:inherit;background:linear-gradient(90deg,#0f766e,#2dd4bf)}.progress-summary{text-align:right}.progress-summary strong{font-size:1rem}.operational-line{display:flex;flex-wrap:wrap;gap:.5rem 1rem;padding:.65rem 0}.operational-line .stale{margin-left:auto;color:#b45309}.analysis-indicator{display:flex;flex-wrap:wrap;gap:.3rem .55rem;margin:0 0 .75rem;padding:.65rem .8rem;color:#92400e;background:#fffbeb;border:1px solid #fde68a;border-radius:7px;font-size:.72rem}.analysis-indicator span{color:#a16207}.live-epoch-panel,.completed-summary,.metric-matrix,.completed-trends,.run-health{margin-top:.75rem;padding:1rem;border:1px solid var(--line,#e2e8f0);border-radius:8px}.live-epoch-panel{border-color:#99f6e4;background:linear-gradient(135deg,#f0fdfa,#fff 58%)}.live-epoch-panel header,.completed-summary>header,.completed-trends>header,.charts>header,.run-health>header{display:flex;align-items:flex-start;justify-content:space-between;gap:1rem}.live-epoch-panel h3,.completed-summary h3,.completed-trends h3,.charts h3,.run-health h3{margin:.12rem 0 .25rem;font-size:.92rem}.live-epoch-panel p:not(.eyebrow),.completed-summary>header>p,.completed-trends p,.charts p{margin:0;color:var(--muted,#64748b);font-size:.72rem}.completed-summary>header>p{max-width:25rem;text-align:right}.epoch-timing{display:grid;grid-template-columns:repeat(3,max-content);gap:.3rem .85rem}.epoch-timing span{display:grid;gap:.12rem;color:var(--muted,#64748b);font-size:.63rem}.epoch-timing strong{color:var(--ink,#172033);font-size:.78rem}.run-health{border-color:#cbd5e1;background:#f8fafc}.run-health.good{border-color:#86efac;background:#f0fdf4}.run-health.attention{border-color:#fcd34d;background:#fffbeb}.health-state{padding:.22rem .48rem;border-radius:999px;background:#e2e8f0;color:#475569;font-size:.65rem;font-weight:800;letter-spacing:.05em}.run-health.good .health-state{background:#dcfce7;color:#166534}.run-health.attention .health-state{background:#fef3c7;color:#92400e}.health-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.65rem;margin-top:.75rem}.health-grid article{padding:.7rem;border:1px solid rgb(148 163 184 / .28);border-radius:7px;background:rgb(255 255 255 / .75)}.health-grid span,.health-grid strong,.health-grid small{display:block}.health-grid span,.health-grid small{color:var(--muted,#64748b);font-size:.67rem}.health-grid strong{margin:.2rem 0;font-size:1rem}.watchlist-signals{display:flex;flex-wrap:wrap;gap:.45rem;margin-top:.7rem}.watchlist-signals span{padding:.35rem .5rem;border-radius:6px;background:#eef2ff;color:#4338ca;font-size:.68rem}.watchlist-signals span.good{background:#dcfce7;color:#166534}.watchlist-signals span.attention{background:#fef3c7;color:#92400e}.watchlist-signals strong{display:inline;margin-right:.25rem}.watchlist-empty{margin:.7rem 0 0;color:var(--muted,#64748b);font-size:.7rem}.metric-cards{display:grid;grid-template-columns:repeat(4,1fr);gap:.65rem;margin:.75rem 0 0}.metric-cards.live{grid-template-columns:repeat(auto-fit,minmax(8rem,1fr))}.metric-cards.compact{grid-template-columns:repeat(auto-fit,minmax(10rem,1fr));margin:.75rem 0 0}.metric-cards article{padding:.75rem;border:1px solid var(--line,#e2e8f0);border-radius:8px;background:#fff}.metric-cards.live article{border-color:#ccfbf1;background:rgb(255 255 255 / .72)}.metric-cards article.improving{border-color:#86efac;background:#f0fdf4}.metric-cards article.attention{border-color:#fcd34d;background:#fffbeb}.metric-cards span,.metric-cards strong,.metric-cards small{display:block}.metric-cards strong{margin:.22rem 0;font-size:1.12rem}.metric-trend.improving{color:#15803d}.metric-trend.attention{color:#b45309;font-weight:700}.metric-trend.steady{color:#64748b}.no-live-signal{margin:.7rem 0 0;color:var(--muted,#64748b);font-size:.76rem}.completed-trends>header>span,.charts>header>span{padding:.22rem .45rem;border-radius:99px;background:#eef2ff;color:#4338ca;font-size:.67rem;font-weight:750;white-space:nowrap}.metric-table-scroll{margin-top:.7rem;overflow:auto}.metric-matrix table{width:100%;border-collapse:collapse;font-size:.75rem}.metric-matrix th,.metric-matrix td{padding:.55rem .65rem;border-top:1px solid #e2e8f0;text-align:left;white-space:nowrap}.metric-matrix thead th{border-top:0;color:var(--muted,#64748b);font-size:.65rem;font-weight:750;text-transform:uppercase;letter-spacing:.04em}.metric-matrix tbody th{font-weight:700}.metric-matrix .missing{color:var(--muted,#64748b)}.metric-matrix td.improving{color:#15803d;font-weight:750}.metric-matrix td.attention{color:#b45309;font-weight:750}.metric-matrix td small{margin-left:.22rem;font-size:.72rem}.dashboard-details{padding:1rem;border:1px solid var(--line,#e2e8f0);border-radius:8px;background:#fff}.dashboard-details summary{display:flex;align-items:center;justify-content:space-between;gap:.75rem;cursor:pointer;list-style:none}.dashboard-details summary::-webkit-details-marker{display:none}.dashboard-details summary:after{color:var(--muted,#64748b);content:'›';font-size:1.2rem}.dashboard-details[open] summary:after{transform:rotate(90deg)}.dashboard-details summary span{display:grid;gap:.12rem}.dashboard-details summary strong,.dashboard-details summary>span:first-child{font-size:.84rem}.dashboard-details summary small{color:var(--muted,#64748b);font-size:.68rem}.dashboard-grid{display:grid;grid-template-columns:minmax(0,1.7fr) minmax(15rem,.8fr);gap:.75rem;margin-top:.75rem}.dashboard-grid>section,.dashboard-grid>.dashboard-details{padding:1rem;border:1px solid var(--line,#e2e8f0);border-radius:8px}.dashboard-grid h3{margin:0 0 .25rem;font-size:.86rem}.chart-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.7rem;margin-top:.7rem}.metric-chart{min-height:8.6rem;padding:.65rem;background:#f8fafc;border-radius:6px}.metric-chart>div{display:flex;justify-content:space-between;gap:.5rem;font-size:.72rem}.metric-chart small{color:var(--muted,#64748b)}.metric-chart svg{width:100%;height:6.2rem;margin-top:.35rem;overflow:visible}.metric-chart line{stroke:#dbe3ed;stroke-width:.7}.metric-chart polyline{fill:none;stroke:#0f766e;stroke-width:2.2;vector-effect:non-scaling-stroke}.metric-chart .train-line{stroke:#0f766e}.metric-chart .validation-line{stroke:#4f46e5}.chart-legend{display:flex!important;gap:.55rem;margin-top:.4rem}.chart-legend span{font-size:.62rem}.chart-legend .train{color:#0f766e}.chart-legend .validation{color:#4f46e5}.chart-single{display:grid!important;place-content:center;min-height:5.8rem;text-align:center}.chart-single strong{font-size:1.35rem}.chart-single span{color:var(--muted,#64748b);font-size:.68rem}.paired-values{grid-template-columns:repeat(2,1fr);gap:.65rem}.paired-values span{display:grid;gap:.15rem}.paired-values strong{color:var(--ink,#172033);font-size:1rem}.metric-chart p,.no-chart,.activity>p{margin:2rem 0;text-align:center;color:var(--muted,#64748b);font-size:.76rem}.activity ol{display:grid;gap:.7rem;margin:.75rem 0 0;padding:0;list-style:none}.activity li{display:grid;grid-template-columns:.55rem 1fr;gap:.5rem}.activity li>i{width:.45rem;height:.45rem;margin-top:.2rem;border-radius:50%;background:#94a3b8}.activity li>i.warning{background:#f59e0b}.activity li>i.error{background:#dc2626}.activity strong,.activity small{display:block;font-size:.72rem}.activity small{margin-top:.18rem;color:var(--muted,#64748b)}.configuration{margin-top:.75rem}.configuration>div{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:.5rem;margin-top:.75rem}.configuration span{min-width:0;display:grid;gap:.2rem}.configuration strong{overflow:hidden;color:var(--ink,#172033);font-size:.72rem;white-space:nowrap;text-overflow:ellipsis}.refresh-error{margin:.65rem 0 0;color:#b45309;font-size:.72rem}@media(max-width:760px){.training-status-backdrop{padding:.5rem}.training-status-modal{max-height:96vh;padding:1rem}.metric-cards,.configuration>div,.health-grid{grid-template-columns:repeat(2,1fr)}.dashboard-grid{grid-template-columns:1fr}.progress-panel{grid-template-columns:1fr}.progress-summary{text-align:left}.operational-line .stale{margin-left:0}.chart-grid{grid-template-columns:1fr}.live-epoch-panel header,.completed-summary>header,.completed-trends>header,.charts>header,.run-health>header{flex-direction:column}.completed-summary>header>p{text-align:left}.epoch-timing{grid-template-columns:repeat(3,1fr);width:100%}}@media(max-width:460px){.modal-header{align-items:flex-start;flex-direction:column}.metric-cards,.health-grid,.configuration>div{grid-template-columns:1fr}.stage-rail{font-size:.56rem}.epoch-timing{grid-template-columns:1fr}}

	/* Dense desktop reading mode: keep the live signal and validated evidence together. */
	.training-status-modal { width: min(1320px, 96vw); padding: 1.1rem 1.25rem; }
	.modal-header { padding-bottom: .7rem; }
	.stage-rail { margin: .65rem 0 .5rem; }
	.stage-rail div { gap: .25rem; }
	.run-at-glance { display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(25rem, .85fr); gap: .65rem; align-items: stretch; }
	.run-at-glance .progress-panel { align-self: start; padding: .75rem .85rem; }
	.run-at-glance .operational-line { grid-column: 1; padding: .45rem 0 0; }
	.run-at-glance .analysis-indicator { grid-column: 1; margin: .1rem 0 0; }
	.run-at-glance .run-health { grid-column: 2; grid-row: 1 / span 3; margin-top: 0; padding: .75rem .85rem; }
	.run-health header { align-items: center; }
	.run-health h3 { font-size: .82rem; }
	.health-grid { gap: .4rem; margin-top: .45rem; }
	.health-grid article { padding: .45rem .5rem; }
	.health-grid strong { margin: .12rem 0; font-size: .9rem; }
	.watchlist-signals { gap: .3rem; margin-top: .45rem; }
	.watchlist-empty { margin: .45rem 0 0; }
	.metric-overview-grid { display: grid; grid-template-columns: minmax(0, 1.18fr) minmax(0, .82fr); gap: .65rem; margin-top: .65rem; }
	.metric-overview-grid .live-epoch-panel, .metric-overview-grid .completed-summary { margin-top: 0; padding: .8rem; }
	.metric-overview-grid .live-epoch-panel header { align-items: center; }
	.metric-overview-grid .completed-summary > header { min-height: 2.4rem; }
	.metric-overview-grid .completed-summary > header > p { display: none; }
	.metric-overview-grid .metric-cards { gap: .45rem; margin-top: .5rem; }
	.metric-overview-grid .metric-cards.live { grid-template-columns: repeat(auto-fit, minmax(7rem, 1fr)); }
	.metric-overview-grid .metric-cards.compact { grid-template-columns: repeat(2, minmax(0, 1fr)); }
	.metric-overview-grid .metric-cards article { padding: .55rem; }
	.metric-overview-grid .metric-cards strong { margin: .12rem 0; font-size: 1rem; }
	.epoch-timing { gap: .2rem .55rem; }
	.dashboard-grid { margin-top: .65rem; }
	.dashboard-grid > section, .dashboard-grid > .dashboard-details { padding: .8rem; }
	.metric-chart { min-height: 6.7rem; padding: .5rem; }
	.metric-chart svg { height: 4.5rem; }
	.completed-trends { margin-top: .65rem; }
	.completed-trends .chart-grid { margin-top: .6rem; }
	.configuration { margin-top: .65rem; }
	@media (max-width: 980px) {
		.run-at-glance, .metric-overview-grid { grid-template-columns: 1fr; }
		.run-at-glance .run-health, .run-at-glance .operational-line, .run-at-glance .analysis-indicator { grid-column: auto; grid-row: auto; }
	}
	.modal-actions { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: .4rem; align-self: start; }
	.danger { color: #b91c1c; border-color: #fecaca; }
	.danger:hover:not(:disabled) { color: #991b1b; background: #fef2f2; }
	@media (max-width: 460px) { .modal-actions { justify-content: flex-start; } }
</style>
