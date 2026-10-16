<script lang="ts">
	export let name: 'edit' | 'verify' | 'start' | 'pause' | 'resume' | 'restart' | 'stop' | 'delete';
</script>

<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">
	{#if name === 'edit'}
		<path d="M12 20h9M4 17.5V20h2.5L19 7.5 16.5 5 4 17.5Z" />
	{:else if name === 'verify'}
		<path d="M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20Z" /><path d="m7.5 12 3 3 6-6" />
	{:else if name === 'start' || name === 'resume'}
		<path d="m8 5 11 7-11 7V5Z" />
	{:else if name === 'pause'}
		<path d="M8 5v14M16 5v14" />
	{:else if name === 'restart'}
		<path d="M20 11a8 8 0 1 0-2.3 6.1M20 4v7h-7" />
	{:else if name === 'stop'}
		<rect x="5" y="5" width="14" height="14" rx="2" />
	{:else}
		<path d="M4 7h16M10 3h4M6 7l1 14h10l1-14M10 11v6M14 11v6" />
	{/if}
</svg>
