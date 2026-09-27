import { env } from '$env/dynamic/private';
import { error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const proxy: RequestHandler = async ({ request, params, url, fetch }) => {
	if (!env.ORCHESTRATOR_URL) throw error(500, 'ORCHESTRATOR_URL is not configured');
	const target = new URL(`/${params.path ?? ''}`, env.ORCHESTRATOR_URL);
	target.search = url.search;
	const headers = new Headers(request.headers);
	headers.delete('host');
	// Preserve streaming bodies (notably multi-gigabyte dataset uploads) instead
	// of materializing a second full copy in the SvelteKit process.
	const body = ['GET', 'HEAD'].includes(request.method) ? undefined : request.body;
	// Node's Undici implementation requires an explicit duplex mode whenever a
	// ReadableStream is forwarded. Without it every POST/PATCH is rejected by
	// the development proxy before the orchestrator can return useful feedback.
	const options: RequestInit & { duplex?: 'half' } = { method: request.method, headers, body };
	if (body) options.duplex = 'half';
	const response = await fetch(target, options);
	return new Response(response.body, { status: response.status, headers: response.headers });
};

export const GET = proxy;
export const POST = proxy;
export const PATCH = proxy;
export const PUT = proxy;
