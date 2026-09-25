import type { RecordValue } from '$lib/api';
import type { ConfigFieldDefinition } from './types';

export const isRecord = (value: unknown): value is RecordValue => Boolean(value) && typeof value === 'object' && !Array.isArray(value);
export const copyConfig = <T,>(value: T): T => JSON.parse(JSON.stringify(value));

export function getPath(source: RecordValue, path: string, fallback: unknown = undefined): unknown {
	const result = path.split('.').reduce<unknown>((current, key) => {
		if (Array.isArray(current) && /^\d+$/.test(key)) return current[Number(key)];
		return isRecord(current) ? current[key] : undefined;
	}, source);
	return result === undefined ? fallback : result;
}

export function setPath(target: RecordValue, path: string, value: unknown): void {
	const keys = path.split('.');
	let current: RecordValue = target;
	for (const key of keys.slice(0, -1)) {
		if (!isRecord(current[key])) current[key] = {};
		current = current[key] as RecordValue;
	}
	current[keys.at(-1)!] = value;
}

export function configWithValue(config: RecordValue, path: string, value: unknown): RecordValue {
	const next = copyConfig(config);
	setPath(next, path, value);
	return next;
}

/** Whether a catalog condition is satisfied by the current resolved config. */
export function conditionMatches(config: RecordValue, condition?: Record<string, unknown>): boolean {
	if (!condition) return true;
	return Object.entries(condition).every(([path, expected]) => {
		const actual = getPath(config, path);
		// Collection controls (such as V2 input channels) satisfy a condition
		// when they contain one of the catalogued applicable values.
		if (Array.isArray(actual)) {
			const expectedValues = Array.isArray(expected) ? expected : [expected];
			return actual.some((value) => expectedValues.includes(value));
		}
		return Array.isArray(expected) ? expected.includes(actual) : actual === expected;
	});
}

export function fieldApplies(field: ConfigFieldDefinition, config: RecordValue): boolean {
	const task = String(getPath(config, 'run.task', 'classification'));
	return (!field.applies_to?.length || field.applies_to.includes(task)) && conditionMatches(config, field.visible_when);
}

export function humanize(value: unknown): string {
	return String(value).replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());
}
