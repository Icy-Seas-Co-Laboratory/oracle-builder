import { env } from '$env/dynamic/private';
import { error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const proxy: RequestHandler = async ({ request, params, url, fetch }) => {
	if (!env.ORCHESTRATOR_URL) throw error(500, 'ORCHESTRATOR_URL is not configured');
	const target = new URL(`/${params.path ?? ''}`, env.ORCHESTRATOR_URL);
	target.search = url.search;
	const headers = new Headers(request.headers);
	headers.delete('host');
	const body = ['GET', 'HEAD'].includes(request.method) ? undefined : await request.arrayBuffer();
	const response = await fetch(target, { method: request.method, headers, body });
	return new Response(response.body, { status: response.status, headers: response.headers });
};

export const GET = proxy;
export const POST = proxy;
