import type { RecordValue } from '$lib/api';

/** Public, server-authored configuration catalog contract. */
export type ConfigFieldDefinition = {
	path: string;
	type?: string;
	default?: unknown;
	label?: string;
	help?: string;
	choices?: unknown[];
	choice_options?: Array<{ value: unknown; label: string }>;
	control?: string;
	minimum?: number;
	maximum?: number;
	step?: number;
	required_choices?: unknown[];
	allowed_values_by?: Record<string, unknown[]>;
	allowed_values_path?: string;
	exposure?: 'standard' | 'advanced' | 'expert' | 'internal' | string;
	editable?: boolean;
	applies_to?: string[];
	visible_when?: Record<string, unknown>;
};

export type ConfigurationSchema = {
	schema_version?: number;
	fingerprint?: string;
	defaults?: RecordValue;
	fields?: ConfigFieldDefinition[];
	groups?: string[];
	group_definitions?: Array<{ id: string; label: string }>;
};

export type ConfigChange = { path: string; value: unknown };
