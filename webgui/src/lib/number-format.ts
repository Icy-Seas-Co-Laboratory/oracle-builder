/**
 * Formats measurements for people without changing the stored value.
 *
 * Four decimal places are enough to compare model metrics in the interface.
 * Very small non-zero values use scientific notation so a learning rate or
 * probability is not displayed as zero.
 */
export function formatNumber(value: unknown, maximumFractionDigits = 4): string {
	if (value === null || value === undefined || value === '') return '—';
	const numeric = typeof value === 'number' ? value : Number(value);
	if (!Number.isFinite(numeric)) return '—';
	if (numeric !== 0 && Math.abs(numeric) < 10 ** -maximumFractionDigits) {
		return numeric.toExponential(Math.max(0, maximumFractionDigits - 1));
	}
	return numeric.toLocaleString(undefined, { maximumFractionDigits });
}
