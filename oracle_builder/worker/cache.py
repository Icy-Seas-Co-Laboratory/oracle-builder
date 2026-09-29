"""Verified immutable artifact cache for pull workers.

The runtime supplies delivery bytes; this cache never treats a reference as a
path or trust a manifest carried only by those bytes.  Every reuse verifies the
content digest pinned in the authenticated delivery grant.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import time
from contextlib import contextmanager
import fcntl
from typing import Any, Callable, Mapping

from oracle_data_contracts.artifacts import (
    ArtifactIntegrityError,
    ArtifactRef,
    copy_verified_directory,
    directory_content_digest,
)


class ArtifactCacheError(RuntimeError):
    pass


class WorkerArtifactCache:
    def __init__(self, root: str | Path, *, max_entries: int = 32) -> None:
        self.root = Path(root).expanduser().resolve()
        self.max_entries = max(1, int(max_entries))
        self.entries = self.root / "entries"; self.quarantine = self.root / "quarantine"
        self.entries.mkdir(parents=True, exist_ok=True); self.quarantine.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _expected(delivery: Mapping[str, Any]) -> str:
        digest = delivery.get("content_manifest_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ArtifactCacheError("delivery grant lacks a pinned directory content digest")
        return digest.lower()

    def _key(self, ref: ArtifactRef, digest: str) -> str:
        return hashlib.sha256(f"{ref.uri}\0{digest}".encode()).hexdigest()

    def _valid(self, path: Path, expected: str) -> bool:
        payload = path / "payload"
        if not payload.is_dir() or payload.is_symlink():
            return False
        try:
            return directory_content_digest(payload) == expected
        except ArtifactIntegrityError:
            # A corrupted cache can contain links, sockets, or malformed
            # inventory entries. Treat every such shape as an invalid entry
            # so the caller quarantines it before a later retry downloads a
            # trusted snapshot.
            return False

    def _quarantine(self, path: Path) -> None:
        if path.exists():
            target = self.quarantine / f"{path.name}-{time.time_ns()}"
            os.replace(path, target)

    def _lock(self, key: str, *, blocking: bool):
        """Acquire a persistent inode lock; kernel releases it on process death."""
        path = self.entries / f".{key}.lock"
        handle = path.open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            return handle
        except BlockingIOError:
            handle.close(); return None

    @staticmethod
    def _unlock(handle) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()

    def materialize(
        self, ref: ArtifactRef, delivery: Mapping[str, Any],
        download: Callable[[Path], Path],
    ) -> Path:
        """Return a verified payload directory, fetching through ``download`` on miss.

        ``download(destination)`` must create and return a materialized artifact
        root whose payload lives at ``root/payload``. It receives only a
        worker-owned temporary path.
        """
        expected = self._expected(delivery); key = self._key(ref, expected); entry = self.entries / key
        lock = self._lock(key, blocking=True)
        if lock is None:
            raise ArtifactCacheError("timed out waiting for artifact cache materialization")
        try:
            return self._materialize_locked(key, entry, expected, download)
        finally:
            self._unlock(lock)

    def _materialize_locked(self, key: str, entry: Path, expected: str, download: Callable[[Path], Path]) -> Path:
            if entry.exists():
                if self._valid(entry, expected):
                    os.utime(entry, None); return entry / "payload"
                self._quarantine(entry)
                raise ArtifactCacheError("cached artifact content digest mismatch; entry quarantined")
            temporary = self.entries / f".{key}.{time.time_ns()}.tmp"; temporary.mkdir()
            try:
                materialized = Path(download(temporary)).resolve()
                if not materialized.is_relative_to(temporary) or not (materialized / "payload").is_dir():
                    raise ArtifactCacheError("delivery callback returned an unsafe or incomplete materialization")
                if directory_content_digest(materialized / "payload") != expected:
                    raise ArtifactCacheError("downloaded artifact content digest does not match its authenticated grant")
                os.replace(materialized, entry)
            except Exception:
                shutil.rmtree(temporary, ignore_errors=True); raise
            finally:
                if temporary.exists(): shutil.rmtree(temporary, ignore_errors=True)
            self._evict(); return entry / "payload"

    @contextmanager
    def acquire(self, ref: ArtifactRef, delivery: Mapping[str, Any], download: Callable[[Path], Path]):
        """Pin one cached payload while a caller copies/uses it."""
        expected = self._expected(delivery); key = self._key(ref, expected)
        lock = self._lock(key, blocking=True)
        if lock is None: raise ArtifactCacheError("could not acquire artifact cache entry")
        try:
            yield self._materialize_locked(key, self.entries / key, expected, download)
        finally:
            self._unlock(lock)

    def materialize_to(self, ref: ArtifactRef, delivery: Mapping[str, Any], download: Callable[[Path], Path], destination: str | Path) -> Path:
        with self.acquire(ref, delivery, download) as payload:
            manifest = copy_verified_directory(payload, Path(destination))
            if manifest.digest_sha256 != self._expected(delivery):
                raise ArtifactCacheError("verified cache copy does not match its authenticated grant")
            return Path(destination)

    def _evict(self) -> None:
        candidates = sorted((path for path in self.entries.iterdir() if path.is_dir() and not path.name.startswith(".")), key=lambda path: path.stat().st_mtime)
        for path in candidates[:-self.max_entries]:
            lock = self._lock(path.name, blocking=False)
            if lock is None: continue
            try:
                shutil.rmtree(path, ignore_errors=True)
            finally:
                self._unlock(lock)
