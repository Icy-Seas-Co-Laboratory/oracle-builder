<script lang="ts">
	export let loading = false;
	export let ready = false;
</script>

<div class="side-foot">
	<span class:pulse={ready} class:offline={!ready}></span>
	{ready ? 'Services ready' : loading ? 'Checking services' : 'Service attention needed'}
</div>
