type RecordValue = Record<string, unknown>;
const object = (value: unknown): RecordValue => value && typeof value === 'object' && !Array.isArray(value) ? value as RecordValue : {};

/** Numeric rankings require an identified population for every member. */
export function exactPosthocComparison(members: RecordValue[]): boolean {
	if (members.length < 2) return false;
	const evidence = members.map((member) => {
		const report = object(member.posthoc);
		return {
			signature: report.evaluation_signature,
			fingerprint: object(report.training_dataset).fingerprint_sha256,
			vocabulary: object(report.class_names)
		};
	});
	if (!evidence.every((item) => typeof item.signature === 'string' && item.signature.length > 0 && typeof item.fingerprint === 'string' && item.fingerprint.length > 0 && Object.keys(item.vocabulary).length > 0)) return false;
	const first = evidence[0];
	const vocabulary = (value: RecordValue) => JSON.stringify(Object.entries(value).sort(([a], [b]) => a.localeCompare(b)));
	return evidence.every((item) => item.signature === first.signature && item.fingerprint === first.fingerprint && vocabulary(item.vocabulary) === vocabulary(first.vocabulary));
}

export function lowerIsBetter(key: string): boolean {
	return /(^|_)(loss|error|brier|log_loss)(_|$)/.test(key);
}
