/** Default y-axis bounds for recorded training metrics. */
export function metricBounds(metric: string, values: number[]): { low: number; high: number } {
	const key = metric.toLowerCase().replace(/^val_/, '');
	const finite = values.filter(Number.isFinite);
	const min = finite.length ? Math.min(...finite) : 0;
	const max = finite.length ? Math.max(...finite) : 1;

	// Loss names can contain score words (for example dice_loss), so classify
	// objectives before bounded classification scores.
	if (/(^|_)(loss|error|mse|mae|rmse|crossentropy|cross_entropy|learning_rate|duration|seconds)(_|$)/.test(key)) {
		const low = Math.min(0, min);
		const high = Math.max(0, max);
		const span = high - low || Math.max(Math.abs(high), 1);
		return { low: low < 0 ? low - span * 0.08 : 0, high: high + span * 0.1 };
	}
	if (/(^|_)(accuracy|f1|precision|recall|auc|auroc|dice|iou|jaccard|specificity|sensitivity)(_|$)/.test(key)) {
		return { low: 0, high: 1 };
	}
	if (key === 'cosine_similarity' || key === 'correlation') return { low: -1, high: 1 };

	const padding = (max - min) * 0.1 || Math.max(Math.abs(max) * 0.05, 0.01);
	return { low: min - padding, high: max + padding };
}
