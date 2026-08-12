<script lang="ts">
	import { onMount } from 'svelte';
	import { api, type RecordValue } from '$lib/api';

	export let title = 'Choose a file';
	export let extensions: string[] = [];
	export let allowDirectory = false;
	export let onselect: (path: string) => void;
	export let onclose: () => void;

	let roots: RecordValue[] = [];
	let rootId = 'workspace';
	let listing: RecordValue | null = null;
	let failure = '';
	let loading = true;

	const asArray = (value: unknown) => Array.isArray(value) ? value as RecordValue[] : [];
	const display = (value: unknown) => value === null || value === undefined ? '' : String(value);
	const matches = (entry: RecordValue) => entry.kind === 'directory' || !extensions.length || extensions.some((extension) => display(entry.name).toLowerCase().endsWith(extension));

	async function browse(path = '') {
		loading = true; failure = '';
		try { listing = await api.files(rootId, path); }
		catch (error) { failure = error instanceof Error ? error.message : 'Could not read this directory.'; }
		finally { loading = false; }
	}

	function changeRoot() { browse(); }
	function choose(entry: RecordValue) {
		if (entry.kind === 'directory') { browse(display(entry.path)); return; }
		const root = listing?.root as RecordValue | undefined;
		onselect(`${display(root?.path).replace(/\/$/, '')}/${display(entry.path)}`);
	}

	function chooseDirectory() {
		const root = listing?.root as RecordValue | undefined;
		const path = display(listing?.path);
		onselect(path ? `${display(root?.path).replace(/\/$/, '')}/${path}` : display(root?.path));
	}

	onMount(async () => {
		try {
			roots = (await api.fileRoots()).roots;
			if (roots.length) rootId = display(roots[0].id);
			await browse();
		} catch (error) { failure = error instanceof Error ? error.message : 'Could not initialize file explorer.'; loading = false; }
	});
</script>

<div class="explorer-backdrop" role="presentation" on:click={onclose}>
	<dialog class="explorer" open aria-label={title} on:click|stopPropagation>
		<header><div><p>FILE EXPLORER</p><h2>{title}</h2></div><button class="close" aria-label="Close file explorer" on:click={onclose}>×</button></header>
		<div class="explorer-tools"><label>Location<select bind:value={rootId} on:change={changeRoot}>{#each roots as root}<option value={display(root.id)}>{display(root.id)} — {display(root.path)}</option>{/each}</select></label>{#if listing?.parent !== null && listing}<button class="secondary small" on:click={() => browse(display(listing?.parent))}>Up one folder</button>{/if}</div>
		<div class="crumb">{display((listing?.root as RecordValue | undefined)?.path)}<span>/{display(listing?.path)}</span></div>
		{#if failure}<p class="explorer-error">{failure}</p>{/if}
		<div class="entries" aria-busy={loading}>{#if loading}<p>Reading directory…</p>{:else if !asArray(listing?.entries).filter(matches).length}<p>No matching files in this folder.</p>{:else}{#each asArray(listing?.entries).filter(matches) as entry}<button class="entry" on:click={() => choose(entry)}><span>{entry.kind === 'directory' ? '▸' : '□'}</span><strong>{display(entry.name)}</strong><small>{entry.kind === 'directory' ? 'Folder' : `${display(entry.size_bytes)} bytes`}</small></button>{/each}{/if}</div>
		<footer><span>{extensions.length ? `Showing ${extensions.join(', ')} files` : 'Showing files and folders'}</span><div>{#if allowDirectory}<button class="secondary small" on:click={chooseDirectory}>Use this folder</button>{/if}<button class="secondary" on:click={onclose}>Cancel</button></div></footer>
	</dialog>
</div>
