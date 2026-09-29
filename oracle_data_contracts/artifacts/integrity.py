"""Content identities and verified snapshots for directory artifacts.

The helpers in this module deliberately describe *bytes on disk*.  They do
not calculate a dataset's semantic fingerprint: a semantic identity answers a
scientific question (which examples and labels are present), whereas this
manifest answers a delivery question (which files did the consumer receive).
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping


DIRECTORY_MANIFEST_SCHEMA = "oracle.directory-content-manifest"
DIRECTORY_MANIFEST_VERSION = 1
DIRECTORY_DIGEST_SCHEME = "sha256:oracle-directory-content-manifest-v1"
_CHUNK_SIZE = 1024 * 1024


class ArtifactIntegrityError(ValueError):
    """Raised when filesystem content differs from a pinned identity."""


def sha256_file(path: str | Path) -> str:
    """Return the raw SHA-256 of one regular, non-symlink file."""
    value = Path(path)
    if value.is_symlink() or not value.is_file():
        raise ArtifactIntegrityError("expected a regular, non-symlink file")
    digest = sha256()
    with value.open("rb") as stream:
        while chunk := stream.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _relative(root: Path, path: Path) -> str:
    relative = path.relative_to(root).as_posix()
    if not relative or relative == "." or relative.startswith("../"):
        raise ArtifactIntegrityError("artifact inventory has an invalid relative path")
    return relative


def _entry(root: Path, path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ArtifactIntegrityError(f"artifact contains a symlink: {_relative(root, path)}")
    stat = path.stat(follow_symlinks=False)
    relative = _relative(root, path)
    if path.is_dir():
        return {"path": relative, "type": "directory"}
    if path.is_file():
        return {"path": relative, "type": "file", "size_bytes": stat.st_size, "sha256": sha256_file(path)}
    raise ArtifactIntegrityError(f"artifact contains a non-regular file: {relative}")


def _normalise_entries(entries: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    paths: set[str] = set()
    for raw in entries:
        item = dict(raw)
        path = item.get("path")
        entry_type = item.get("type")
        if not isinstance(path, str) or not path or path.startswith("/") or "\\" in path:
            raise ArtifactIntegrityError("directory manifest contains an invalid relative path")
        parts = path.split("/")
        if any(part in {"", ".", ".."} for part in parts) or path in paths:
            raise ArtifactIntegrityError("directory manifest contains duplicate or unsafe paths")
        paths.add(path)
        if entry_type == "directory":
            if set(item) != {"path", "type"}:
                raise ArtifactIntegrityError("directory manifest directory entry is malformed")
            values.append({"path": path, "type": "directory"})
        elif entry_type == "file":
            if set(item) != {"path", "type", "size_bytes", "sha256"}:
                raise ArtifactIntegrityError("directory manifest file entry is malformed")
            size = item["size_bytes"]
            digest = item["sha256"]
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ArtifactIntegrityError("directory manifest file size is invalid")
            if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ArtifactIntegrityError("directory manifest file digest is invalid")
            values.append({"path": path, "type": "file", "size_bytes": size, "sha256": digest})
        else:
            raise ArtifactIntegrityError("directory manifest entry type is invalid")
    return sorted(values, key=lambda item: item["path"])


@dataclass(frozen=True, slots=True)
class DirectoryContentManifest:
    """Canonical, versioned inventory of a directory's exact content bytes."""

    entries: tuple[Mapping[str, Any], ...]
    digest_sha256: str

    @classmethod
    def create(cls, root: str | Path) -> "DirectoryContentManifest":
        directory = Path(root)
        if directory.is_symlink() or not directory.is_dir():
            raise ArtifactIntegrityError("artifact root must be a directory, not a symlink")
        entries = _normalise_entries(_entry(directory, path) for path in directory.rglob("*"))
        body = {
            "schema": DIRECTORY_MANIFEST_SCHEMA,
            "version": DIRECTORY_MANIFEST_VERSION,
            "digest_scheme": DIRECTORY_DIGEST_SCHEME,
            "entries": entries,
        }
        return cls(tuple(entries), sha256(_canonical_json(body)).hexdigest())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DirectoryContentManifest":
        if value.get("schema") != DIRECTORY_MANIFEST_SCHEMA or value.get("version") != DIRECTORY_MANIFEST_VERSION:
            raise ArtifactIntegrityError("unsupported directory content manifest")
        if value.get("digest_scheme") != DIRECTORY_DIGEST_SCHEME or not isinstance(value.get("entries"), list):
            raise ArtifactIntegrityError("directory content manifest is malformed")
        entries = _normalise_entries(value["entries"])
        body = {
            "schema": DIRECTORY_MANIFEST_SCHEMA,
            "version": DIRECTORY_MANIFEST_VERSION,
            "digest_scheme": DIRECTORY_DIGEST_SCHEME,
            "entries": entries,
        }
        digest = sha256(_canonical_json(body)).hexdigest()
        supplied = value.get("digest_sha256")
        if supplied is not None and supplied != digest:
            raise ArtifactIntegrityError("directory content manifest digest is invalid")
        return cls(tuple(entries), digest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DIRECTORY_MANIFEST_SCHEMA,
            "version": DIRECTORY_MANIFEST_VERSION,
            "digest_scheme": DIRECTORY_DIGEST_SCHEME,
            "entries": [dict(entry) for entry in self.entries],
            "digest_sha256": self.digest_sha256,
        }

    def verify(self, root: str | Path) -> None:
        actual = self.create(root)
        if actual.digest_sha256 != self.digest_sha256:
            expected = {entry["path"]: entry for entry in self.entries}
            observed = {entry["path"]: entry for entry in actual.entries}
            missing = sorted(set(expected) - set(observed))
            extra = sorted(set(observed) - set(expected))
            changed = sorted(
                path for path in set(expected).intersection(observed)
                if expected[path] != observed[path]
            )
            detail = []
            if missing:
                detail.append(f"missing={','.join(missing[:5])}")
            if extra:
                detail.append(f"extra={','.join(extra[:5])}")
            if changed:
                detail.append(f"changed={','.join(changed[:5])}")
            raise ArtifactIntegrityError("artifact content does not match pinned directory manifest" + (f" ({'; '.join(detail)})" if detail else ""))


def verify_directory(root: str | Path, expected: DirectoryContentManifest | Mapping[str, Any]) -> DirectoryContentManifest:
    """Verify ``root`` against a parsed/serialized expected manifest."""
    manifest = expected if isinstance(expected, DirectoryContentManifest) else DirectoryContentManifest.from_dict(expected)
    manifest.verify(root)
    return manifest


def directory_content_digest(root: str | Path) -> str:
    """Compute the versioned digest of the exact directory bytes on disk.

    Workers use this after safe extraction and compare it with the digest
    supplied by an authenticated materialization grant.  The digest is never
    accepted from the archive itself as the expected identity.
    """
    return DirectoryContentManifest.create(root).digest_sha256


def copy_verified_directory(
    source: str | Path,
    destination: str | Path,
    *,
    expected: DirectoryContentManifest | Mapping[str, Any] | None = None,
) -> DirectoryContentManifest:
    """Copy an exact, verified private snapshot.

    When an expected manifest is supplied, source and destination must both
    match it.  Otherwise the source is inventoried first and the copied bytes
    must match that captured inventory.  This detects source mutation during
    copying and never follows links.
    """
    source_path = Path(source)
    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"verified snapshot destination already exists: {target}")
    manifest = expected if isinstance(expected, DirectoryContentManifest) else (
        DirectoryContentManifest.from_dict(expected) if expected is not None else DirectoryContentManifest.create(source_path)
    )
    manifest.verify(source_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Copy each verified entry explicitly; shutil.copytree would make it
        # too easy to silently accept a late symlink or an unexpected extra.
        target.mkdir()
        for entry in manifest.entries:
            relative = Path(str(entry["path"]))
            origin = source_path / relative
            output = target / relative
            if entry["type"] == "directory":
                output.mkdir()
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                if origin.is_symlink() or not origin.is_file():
                    raise ArtifactIntegrityError(f"artifact changed while copying: {entry['path']}")
                with origin.open("rb") as reader, output.open("xb") as writer:
                    shutil.copyfileobj(reader, writer, length=_CHUNK_SIZE)
        manifest.verify(target)
        return manifest
    except Exception:
        # The destination is caller-owned scratch; leave no partial snapshot
        # that a later caller could mistake for validated bytes.
        if target.exists():
            shutil.rmtree(target)
        raise
