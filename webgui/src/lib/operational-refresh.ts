/**
 * A single operational refresh loop for the workspace.  Server-sent events are
 * opportunistic: older orchestrators simply return an error and polling remains
 * the source of truth.
 */
export type RefreshReason = 'initial' | 'poll' | 'event' | 'visible' | 'manual' | 'mutation';
export type RefreshCallback = (signal: AbortSignal, reason: RefreshReason) => Promise<void>;

type Listener = (reason: RefreshReason) => void;
const listeners = new Set<Listener>();

export function subscribeOperationalRefresh(listener: Listener): () => void {
	listeners.add(listener);
	return () => listeners.delete(listener);
}

export function publishOperationalRefresh(reason: RefreshReason) {
	for (const listener of listeners) listener(reason);
}

function isAbort(error: unknown): boolean {
	return error instanceof DOMException && error.name === 'AbortError';
}

export class OperationalRefreshCoordinator {
	private timer: ReturnType<typeof setTimeout> | undefined;
	private reconnectTimer: ReturnType<typeof setTimeout> | undefined;
	private eventSource: EventSource | undefined;
	private controller: AbortController | undefined;
	private inFlight: Promise<void> | undefined;
	private stopped = true;

	constructor(private readonly refresh: RefreshCallback, private readonly hasActiveWork: () => boolean) {}

	start() {
		if (!this.stopped) return;
		this.stopped = false;
		document.addEventListener('visibilitychange', this.onVisibilityChange);
		this.connectEvents();
		void this.trigger('initial');
	}

	stop() {
		this.stopped = true;
		this.controller?.abort();
		this.controller = undefined;
		this.eventSource?.close();
		this.eventSource = undefined;
		if (this.timer) clearTimeout(this.timer);
		if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
		this.timer = undefined;
		this.reconnectTimer = undefined;
		document.removeEventListener('visibilitychange', this.onVisibilityChange);
	}

	trigger(reason: RefreshReason, supersede = false): Promise<void> {
		if (this.stopped) return Promise.resolve();
		if (this.inFlight && !supersede) return this.inFlight;
		if (supersede) this.controller?.abort();
		this.controller = new AbortController();
		const signal = this.controller.signal;
		const task = this.refresh(signal, reason)
			.catch((error: unknown) => {
				// Cancellation is expected when a newer manual/event refresh wins.
				if (!isAbort(error)) throw error;
			})
			.finally(() => {
				if (this.controller?.signal === signal) this.inFlight = undefined;
				if (!this.stopped) this.schedule();
			});
		this.inFlight = task;
		return task;
	}

	private schedule() {
		if (this.timer) clearTimeout(this.timer);
		const delay = document.visibilityState === 'hidden' ? 60_000 : this.hasActiveWork() ? 5_000 : 20_000;
		this.timer = setTimeout(() => void this.trigger('poll'), delay);
	}

	private readonly onVisibilityChange = () => {
		if (document.visibilityState === 'visible') void this.trigger('visible', true);
		else this.schedule();
	};

	private connectEvents() {
		if (this.stopped || this.eventSource || typeof EventSource === 'undefined') return;
		try {
			const source = new EventSource('/api/v1/events');
			this.eventSource = source;
			const refreshFromEvent = () => {
				// A burst of events maps to one in-flight refresh. Subscribers are
				// notified after that refresh commits a coherent snapshot.
				void this.trigger('event');
			};
			source.onmessage = refreshFromEvent;
			// The durable event API uses named events, while older deployments may
			// send ordinary `message` events.
			// Validation emits named stage events as well as lifecycle events.  Listen
			// to each known stage so a long calibration/preflight updates its UI
			// immediately; polling remains the compatibility and recovery path.
			for (const eventType of [
				'queued', 'started', 'completed', 'failed', 'validating', 'preflight',
				'calibrating', 'compute_preflight', 'queued_run_created'
			]) source.addEventListener(eventType, refreshFromEvent);
			source.onerror = () => {
				source.close();
				if (this.eventSource === source) this.eventSource = undefined;
				// The endpoint is intentionally optional during rolling upgrades.
				if (!this.stopped) this.reconnectTimer = setTimeout(() => this.connectEvents(), 30_000);
			};
		} catch {
			this.reconnectTimer = setTimeout(() => this.connectEvents(), 30_000);
		}
	}
}
