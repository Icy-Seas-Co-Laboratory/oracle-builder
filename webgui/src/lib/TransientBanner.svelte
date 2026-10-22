<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { fade } from 'svelte/transition';

	export let message: string;
	export let title: string;
	export let tone: 'error' | 'notice' = 'error';
	export let ondismiss: () => void;

	let mounted = false;
	let reducedMotion = false;
	let timer: ReturnType<typeof setTimeout> | undefined;

	function clearTimer() {
		if (timer !== undefined) clearTimeout(timer);
		timer = undefined;
	}
	function expireAfterOnset(_message: string) {
		clearTimer();
		timer = setTimeout(ondismiss, 8000);
	}
	// A changed message gets its own lifetime; identical polling errors do not
	// keep extending the currently displayed warning.
	$: if (mounted) expireAfterOnset(message);
	onMount(() => {
		reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
		mounted = true;
	});
	onDestroy(clearTimer);
</script>

<div class="banner {tone}" role={tone === 'error' ? 'alert' : 'status'} out:fade={{ duration: reducedMotion ? 0 : 300 }}>
	<strong>{title}</strong><span>{message}</span><button type="button" aria-label="Dismiss" on:click={ondismiss}>×</button>
</div>
