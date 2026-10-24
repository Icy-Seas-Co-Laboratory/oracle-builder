export type TelemetryPoint = Record<string, unknown>;

export function phaseId(snapshot: TelemetryPoint, fallback: 'self_supervised' | 'supervised' = 'supervised'): 'self_supervised' | 'supervised' {
	const explicit = snapshot.phase_id;
	if (explicit === 'self_supervised' || explicit === 'supervised') return explicit;
	const name = String(snapshot.phase ?? '').toLowerCase();
	return name ? (name.includes('ssl') || name.includes('self_supervised') ? 'self_supervised' : 'supervised') : fallback;
}

export function phaseHistory(points: TelemetryPoint[], phase: 'self_supervised' | 'supervised'): TelemetryPoint[] {
	return points.filter((point) => phaseId(point) === phase);
}

export const sslMetricKeys = ['loss', 'simclr_loss', 'contrastive_accuracy', 'reconstruction_mse', 'foreground_mse', 'variance_loss', 'covariance_loss', 'representation_std', 'encoder_variance_loss', 'encoder_covariance_loss', 'encoder_representation_std', 'projector_representation_std', 'collapse_indicator'];
export const supervisedMetricKeys = ['loss', 'accuracy', 'macro_f1'];

export function visibleMetricKeys(points: TelemetryPoint[], phase: 'self_supervised' | 'supervised'): string[] {
	points = phaseHistory(points, phase);
	const candidates = phase === 'self_supervised' ? sslMetricKeys : supervisedMetricKeys;
	const numeric = (key: string) => points.some((point) => typeof point[key] === 'number' && Number.isFinite(point[key]));
	const known = candidates.filter(numeric);
	const dynamic = phase === 'self_supervised'
		? [...new Set(points.flatMap((point) => Object.keys(point)))].filter((key) => !known.includes(key) && !['epoch', 'batch', 'elapsed_seconds'].includes(key) && numeric(key)).sort()
		: [];
	return [...known, ...dynamic];
}

export function collectPhaseEpochs(units: {
	snapshot: TelemetryPoint; startEpoch?: number; stopEpoch?: number;
	fallbackPhase?: 'self_supervised' | 'supervised';
}[], phase: 'self_supervised' | 'supervised'): TelemetryPoint[] {
	const byEpoch = new Map<number, TelemetryPoint>();
	for (const unit of units) {
		if (phaseId(unit.snapshot, unit.fallbackPhase) !== phase) continue;
		for (const point of Array.isArray(unit.snapshot.history) ? unit.snapshot.history as TelemetryPoint[] : []) {
			const reported = point.epoch;
			if (typeof reported !== 'number' || !Number.isFinite(reported)) continue;
			const start = unit.startEpoch ?? 0;
			const epoch = unit.stopEpoch !== undefined && reported <= unit.stopEpoch - start ? start + reported : reported;
			byEpoch.set(epoch, { ...point, epoch });
		}
	}
	return [...byEpoch.values()].sort((a, b) => Number(a.epoch) - Number(b.epoch));
}
