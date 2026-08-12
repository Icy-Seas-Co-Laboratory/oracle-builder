<script lang="ts">
	export let title: string;
	export let text: string;
	export let action = '';
	export let onaction: (() => void) | undefined = undefined;
</script>

<div class="empty-state">
	<div>◌</div>
	<strong>{title}</strong>
	<p>{text}</p>
	{#if action && onaction}<button class="secondary small" on:click={onaction}>{action}</button>{/if}
</div>
