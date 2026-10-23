"""Worker-local reuse of immutable raw embeddings across classifier fits."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterator, Mapping

from .cache import open_embedding_cache


def _complete(root: Path) -> bool:
    try:
        for split in ('train', 'validation', 'test'):
            cache = open_embedding_cache(root / split / 'fused_embedding')
            if cache.shape[0] < 1 or len(list(cache.iter_records())) != cache.shape[0]:
                return False
        return True
    except (OSError, ValueError, KeyError, RuntimeError):
        return False


@contextmanager
def reusable_representation_cache(
    run_dir: str | Path, cache_root: str | Path | None, identity: Mapping[str, Any],
) -> Iterator[None]:
    """Copy complete caches under a lock; publish only after a successful fit.

    Identity must contain the sealed source model, dataset fingerprint and exact
    split assignments. Only raw embeddings are shared; fitted preprocessing and
    classifiers always belong to their own output artifact.
    """
    if cache_root is None:
        yield
        return
    import fcntl

    root = Path(cache_root)
    root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()
    cached = root / key
    local = Path(run_dir) / 'intermediates' / 'representations'
    with (root / f'{key}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if _complete(cached):
            shutil.copytree(cached, local, dirs_exist_ok=True)
            consumer_path = Path(run_dir) / 'artifact.json'
            consumer = json.loads(consumer_path.read_text()) if consumer_path.is_file() else {}
            for manifest_path in local.glob('*/fused_embedding/manifest.json'):
                manifest = json.loads(manifest_path.read_text())
                manifest.setdefault('provenance', {})['cache_reuse'] = {
                    'cache_key': key,
                    'consumer_artifact_id': consumer.get('artifact_id'),
                    'consumer_run_id': consumer.get('run_id'),
                    'identity': dict(identity),
                }
                manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
            # Warm-cache fits no longer need the extraction lock and can run
            # in parallel with other classifiers using these immutable rows.
            fcntl.flock(lock, fcntl.LOCK_UN)
            yield
            return
        yield
        if not _complete(cached) and _complete(local):
            temporary = Path(tempfile.mkdtemp(prefix=f'.{key}-', dir=root))
            try:
                shutil.copytree(local, temporary, dirs_exist_ok=True)
                if cached.exists():
                    shutil.rmtree(cached)
                temporary.replace(cached)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
