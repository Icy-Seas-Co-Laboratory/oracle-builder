"""Filesystem-backed artifact storage for the orchestrator control plane.

This is intentionally a small compatibility layer.  It keeps artifacts as
ordinary directories so they remain easy to inspect, copy, snapshot, and
recover, while ensuring workers can use location-free :class:`ArtifactRef`
values rather than receiving host paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
from threading import RLock
from typing import Any, Callable, Iterator, Mapping, Protocol
import uuid
from secrets import token_urlsafe

from oracle_data_contracts.artifacts import (
    ArtifactIntegrityError,
    ArtifactRef,
    DirectoryContentManifest,
    copy_verified_directory,
    sha256_file,
)


_STAGE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
Validator = Callable[[Path], Mapping[str, Any]]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    """Stable bytes used for content-manifest fingerprints."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stage_token(value: str, name: str) -> str:
    if not isinstance(value, str) or not _STAGE_TOKEN.fullmatch(value):
        raise ValueError(f"{name} must be a non-path token")
    return value


def artifact_ref_for_file(
    kind: str,
    artifact_id: str,
    source: str | Path,
    *,
    revision: str | None = None,
) -> ArtifactRef:
    """Return the SHA-pinned reference required to ingest a portable file.

    The caller supplies stable domain identity (for example a dataset id and
    revision); this helper deliberately derives only the content fingerprint.
    It is intended for the orchestrator at submission time, never for a
    worker that has received an already sealed work unit.
    """
    raw_path = Path(source).expanduser()
    if raw_path.is_symlink() or not raw_path.is_file():
        raise ValueError("portable file artifact source must be a regular, non-symlink file")
    path = raw_path.resolve()
    digest = sha256()
    with path.open("rb") as payload:
        while chunk := payload.read(1024 * 1024):
            digest.update(chunk)
    return ArtifactRef(kind, artifact_id, revision=revision, fingerprint_sha256=digest.hexdigest())


@dataclass(frozen=True, slots=True)
class StagingArea:
    """An opaque, job-scoped candidate area; only the store exposes its path."""

    job_id: str
    attempt_id: str
    path: Path


@dataclass(frozen=True, slots=True)
class ValidationResult:
    valid: bool
    report: dict[str, Any]


@dataclass(frozen=True, slots=True)
class MaterializationGrant:
    """A short-lived, single-use authority to obtain one artifact.

    This is deliberately a *delivery* contract, not an artifact location.
    ``token`` is an opaque bearer value and is excluded from :meth:`to_dict`
    so normal logging, persistence, and status responses cannot accidentally
    retain it.  The orchestrator sends :meth:`delivery_dict` only to the
    worker that owns the matching lease.
    """

    grant_id: str
    lease_id: str
    job_id: str
    ref: ArtifactRef
    expires_at: str
    token: str
    content_manifest_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return safe metadata suitable for persistence and observability."""
        return {
            "grant_id": self.grant_id,
            "lease_id": self.lease_id,
            "job_id": self.job_id,
            "ref": self.ref.to_dict(),
            "expires_at": self.expires_at,
            # This is a directory-content identity, deliberately distinct
            # from ArtifactRef.fingerprint_sha256, which legacy callers may
            # use as either a raw payload or semantic fingerprint.
            **({"content_manifest_sha256": self.content_manifest_sha256}
               if self.content_manifest_sha256 is not None else {}),
        }

    def delivery_dict(self) -> dict[str, Any]:
        """Return the one response shape that includes the bearer token."""
        return {**self.to_dict(), "token": self.token}


class ArtifactStore(Protocol):
    """Storage authority used by the orchestrator.

    The protocol separates a worker-writable staging area from immutable,
    published artifacts.  ``seal`` here records the storage transition; an
    artifact-specific validator remains responsible for format-level sealing
    such as the existing run ``artifact.json`` checksum seal.
    """

    def begin_staging(self, job_id: str, attempt_id: str) -> StagingArea: ...
    def staging(self, job_id: str, attempt_id: str) -> StagingArea: ...
    def staging_path(self, staging: StagingArea) -> Path: ...
    def validate(self, staging: StagingArea, validator: Validator) -> ValidationResult: ...
    def seal(self, staging: StagingArea) -> None: ...
    def publish(self, staging: StagingArea, ref: ArtifactRef) -> Path: ...
    def published_ref(self, job_id: str, attempt_id: str) -> ArtifactRef | None: ...
    def resolve(self, ref: ArtifactRef) -> Path: ...
    def materialize(self, ref: ArtifactRef, destination: str | Path) -> Path: ...
    def ingest_file(self, ref: ArtifactRef, source: str | Path) -> Path: ...
    def issue_materialization_grant(
        self, ref: ArtifactRef, *, lease_id: str, job_id: str, ttl_seconds: int = 600
    ) -> MaterializationGrant: ...
    def materialize_grant(
        self,
        grant: MaterializationGrant | str,
        destination: str | Path,
        *,
        lease_id: str,
        job_id: str,
    ) -> Path: ...
    def stream_grant_archive(
        self,
        grant: MaterializationGrant | str,
        *,
        lease_id: str,
        job_id: str,
        expected_grant_id: str | None = None,
        chunk_size: int = 64 * 1024,
    ) -> Iterator[bytes]: ...


class S3ObjectClient(Protocol):
    """The small S3-compatible surface used by :class:`S3ReplicatedArtifactStore`.

    ``boto3.client("s3")`` implements this protocol.  Keeping the dependency
    at this boundary lets local-only installations stay dependency-free and
    makes MinIO or another compatible client straightforward to test.
    """

    def upload_file(self, filename: str, bucket: str, key: str) -> None: ...

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> Any: ...

    def get_object(self, *, Bucket: str, Key: str) -> Mapping[str, Any]: ...

    def download_file(self, bucket: str, key: str, filename: str) -> None: ...


class S3ReplicatedArtifactStore:
    """Folder-backed store with an immutable S3-compatible replica.

    Local folders remain the primary storage contract in this first object
    store slice: validation, atomic rename, catalog scanning, and grant
    delivery all continue to operate on ordinary directories.  After a local
    artifact is sealed and published, its complete contents are uploaded to
    an S3-compatible bucket and a final marker is written last.  Consumers
    can therefore distinguish a complete replica from a partial upload.

    This is deliberately replication rather than a second, incompatible path
    abstraction.  It preserves easy copy/backup/model inspection today while
    allowing S3, MinIO, or a managed object store to provide off-host
    durability.  A failed replication leaves the local artifact available
    and can be safely retried with :meth:`replicate`.
    """

    _MARKER = ".oracle-artifact.json"

    def __init__(
        self,
        local: LocalArtifactStore,
        *,
        bucket: str,
        client: S3ObjectClient,
        prefix: str = "oracle-builder",
    ) -> None:
        if not isinstance(local, LocalArtifactStore):
            raise TypeError("local must be a LocalArtifactStore")
        bucket = bucket.strip()
        if not bucket or any(char.isspace() for char in bucket):
            raise ValueError("bucket must be a non-empty S3 bucket name")
        normalized_prefix = prefix.strip().strip("/")
        if ".." in normalized_prefix.split("/"):
            raise ValueError("prefix must not contain parent segments")
        self.local = local
        self.bucket = bucket
        self.prefix = normalized_prefix
        self.client = client
        # This is diagnostic only.  The durable replication ledger belongs to
        # the Orchestrator, which can resume work after this process exits.
        self._last_results: dict[str, dict[str, Any]] = {}

    def _key(self, ref: ArtifactRef, relative: Path | None = None) -> str:
        parts = [part for part in (self.prefix, ref.kind, ref.artifact_id, ref.revision) if part]
        if relative is not None:
            parts.extend(relative.as_posix().split("/"))
        return "/".join(parts)

    def _manifest(self, ref: ArtifactRef) -> dict[str, Any]:
        """Build the deterministic content manifest used as replica proof."""
        source = self.local.resolve(ref)
        files = [path for path in sorted(source.rglob("*"), key=lambda item: item.as_posix()) if path.is_file()]
        manifest_files: list[dict[str, Any]] = []
        for path in files:
            if path.is_symlink():
                raise ValueError("artifact contains a symlink and cannot be replicated")
            relative = path.relative_to(source)
            manifest_files.append({
                "path": relative.as_posix(), "sha256": sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            })
        body = {"schema_version": 1, "artifact_ref": ref.to_dict(), "files": manifest_files}
        body["manifest_sha256"] = sha256(_json_bytes(body)).hexdigest()
        return body

    def replicate(self, ref: ArtifactRef) -> dict[str, Any]:
        """Upload one already-published artifact, writing completion last."""
        source = self.local.resolve(ref)
        manifest = self._manifest(ref)
        for entry in manifest["files"]:
            relative = Path(str(entry["path"]))
            path = source / relative
            relative = path.relative_to(source)
            self.client.upload_file(str(path), self.bucket, self._key(ref, relative))
        marker = {**manifest, "replicated_at": _now()}
        self.client.put_object(
            Bucket=self.bucket,
            Key=self._key(ref, Path(self._MARKER)),
            Body=(json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"),
        )
        result = {"bucket": self.bucket, "prefix": self._key(ref), "files": len(manifest["files"]),
                  "marker": self._key(ref, Path(self._MARKER)), "manifest_sha256": manifest["manifest_sha256"]}
        self._last_results[ref.uri] = {"status": "replicated", **result}
        return result

    def replication_status(self, ref: ArtifactRef) -> dict[str, Any]:
        """Return process-local diagnostics; use the Orchestrator ledger for truth."""
        return dict(self._last_results.get(ref.uri, {"status": "pending"}))

    def _remote_marker(self, ref: ArtifactRef) -> dict[str, Any]:
        response = self.client.get_object(Bucket=self.bucket, Key=self._key(ref, Path(self._MARKER)))
        body = response.get("Body")
        raw = body.read() if hasattr(body, "read") else body
        if not isinstance(raw, bytes):
            raise ValueError("S3 completion marker was not bytes")
        marker = json.loads(raw.decode("utf-8"))
        if not isinstance(marker, dict) or marker.get("artifact_ref") != ref.to_dict() or not isinstance(marker.get("files"), list):
            raise ValueError("S3 completion marker does not match artifact")
        expected = {key: marker[key] for key in ("schema_version", "artifact_ref", "files")}
        if marker.get("manifest_sha256") != sha256(_json_bytes(expected)).hexdigest():
            raise ValueError("S3 completion marker manifest digest is invalid")
        return marker

    def verify_replica(self, ref: ArtifactRef) -> dict[str, Any]:
        """Compare the immutable remote completion marker with local content."""
        marker = self._remote_marker(ref)
        local = self._manifest(ref)
        if marker["manifest_sha256"] != local["manifest_sha256"]:
            raise ValueError("S3 replica content does not match the local artifact")
        result = {"bucket": self.bucket, "marker": self._key(ref, Path(self._MARKER)),
                  "files": len(local["files"]), "manifest_sha256": local["manifest_sha256"]}
        self._last_results[ref.uri] = {"status": "replicated", **result}
        return result

    def restore(self, ref: ArtifactRef) -> Path:
        """Restore a missing canonical local artifact from a verified replica.

        Existing local data is never overwritten.  All downloaded files are
        checked against the signed-by-content marker before one local rename.
        """
        destination = self.local._canonical_path(ref)  # Store-owned canonical location, not caller input.
        if destination.exists():
            raise FileExistsError(f"local artifact already exists: {ref.uri}")
        marker = self._remote_marker(ref)
        staging_root = self.local.root / ".restore-staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix="restore-", dir=staging_root))
        try:
            for entry in marker["files"]:
                relative = Path(str(entry.get("path", "")))
                if not relative.parts or relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("S3 completion marker contains an unsafe path")
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                self.client.download_file(self.bucket, self._key(ref, relative), str(target))
                if not target.is_file() or target.stat().st_size != entry.get("size_bytes") or sha256(target.read_bytes()).hexdigest() != entry.get("sha256"):
                    raise ValueError("downloaded S3 object did not match completion marker")
            expected = {key: marker[key] for key in ("schema_version", "artifact_ref", "files")}
            if sha256(_json_bytes(expected)).hexdigest() != marker["manifest_sha256"]:
                raise ValueError("S3 completion marker manifest digest is invalid")
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, destination)
            self._last_results[ref.uri] = {"status": "replicated", "restored_at": _now(),
                                           "manifest_sha256": marker["manifest_sha256"]}
            return destination
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    # The regular ArtifactStore operations deliberately delegate to the
    # folder authority.  This keeps its atomic staging/grant state machine
    # authoritative and avoids an object store becoming a hidden path API.
    def begin_staging(self, job_id: str, attempt_id: str) -> StagingArea:
        return self.local.begin_staging(job_id, attempt_id)

    def staging(self, job_id: str, attempt_id: str) -> StagingArea:
        return self.local.staging(job_id, attempt_id)

    def staging_path(self, staging: StagingArea) -> Path:
        return self.local.staging_path(staging)

    def validate(self, staging: StagingArea, validator: Validator) -> ValidationResult:
        return self.local.validate(staging, validator)

    def seal(self, staging: StagingArea) -> None:
        self.local.seal(staging)

    def publish(self, staging: StagingArea, ref: ArtifactRef) -> Path:
        published = self.local.publish(staging, ref)
        try:
            self.replicate(ref)
        except Exception as exc:
            self._last_results[ref.uri] = {"status": "failed", "error": str(exc)}
        return published

    def published_ref(self, job_id: str, attempt_id: str) -> ArtifactRef | None:
        return self.local.published_ref(job_id, attempt_id)

    def resolve(self, ref: ArtifactRef) -> Path:
        return self.local.resolve(ref)

    def materialize(self, ref: ArtifactRef, destination: str | Path) -> Path:
        return self.local.materialize(ref, destination)

    def ingest_file(self, ref: ArtifactRef, source: str | Path) -> Path:
        published = self.local.ingest_file(ref, source)
        try:
            self.replicate(ref)
        except Exception as exc:
            self._last_results[ref.uri] = {"status": "failed", "error": str(exc)}
        return published

    def register_existing(self, ref: ArtifactRef, path: str | Path) -> Path:
        published = self.local.register_existing(ref, path)
        try:
            self.replicate(ref)
        except Exception as exc:
            self._last_results[ref.uri] = {"status": "failed", "error": str(exc)}
        return published

    def issue_materialization_grant(self, ref: ArtifactRef, *, lease_id: str, job_id: str,
                                    ttl_seconds: int = 600) -> MaterializationGrant:
        return self.local.issue_materialization_grant(ref, lease_id=lease_id, job_id=job_id, ttl_seconds=ttl_seconds)

    def materialize_grant(self, grant: MaterializationGrant | str, destination: str | Path, *,
                          lease_id: str, job_id: str) -> Path:
        return self.local.materialize_grant(grant, destination, lease_id=lease_id, job_id=job_id)

    def stream_grant_archive(self, grant: MaterializationGrant | str, *, lease_id: str, job_id: str,
                             expected_grant_id: str | None = None, chunk_size: int = 64 * 1024) -> Iterator[bytes]:
        return self.local.stream_grant_archive(
            grant, lease_id=lease_id, job_id=job_id, expected_grant_id=expected_grant_id, chunk_size=chunk_size,
        )


class LocalArtifactStore:
    """Portable-folder implementation of :class:`ArtifactStore`.

    Published directories live below ``<root>/<kind>/<artifact-id>/`` (with a
    revision subdirectory when supplied).  Existing folders can be registered
    in place, which makes adoption non-destructive during the migration.
    Staging state is kept outside candidates so published artifacts never gain
    storage-control files.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._staging_root = self.root / "staging"
        self._state_root = self.root / ".staging-state"
        self._registry_path = self.root / ".artifact-locations.json"
        # The authority records only a SHA-256 digest of each bearer token.
        # A process-local lock makes issue/consume atomic for this filesystem
        # adapter; a distributed store must provide the same atomic transition.
        self._grants_path = self.root / ".materialization-grants.json"
        self._grant_lock = RLock()

    def _canonical_path(self, ref: ArtifactRef) -> Path:
        path = self.root / ref.kind / ref.artifact_id
        return path / ref.revision if ref.revision is not None else path

    def _state_path(self, staging: StagingArea) -> Path:
        return self._state_root / staging.job_id / f"{staging.attempt_id}.json"

    def _checked_staging(self, staging: StagingArea) -> tuple[Path, dict[str, Any]]:
        if not isinstance(staging, StagingArea):
            raise TypeError("staging must be a StagingArea returned by this store")
        raw_path = self._staging_root / staging.job_id / staging.attempt_id
        # A worker must not turn its staging area (or its job parent) into a
        # symlink and thereby make a local store write outside its namespace.
        if raw_path.is_symlink() or raw_path.parent.is_symlink():
            raise ValueError("staging area may not be a symlink")
        expected = raw_path.resolve()
        if staging.path.resolve() != expected or not expected.is_dir():
            raise ValueError("staging area does not belong to this store or no longer exists")
        state_path = self._state_path(staging)
        if not state_path.is_file():
            raise ValueError("staging area has no storage state")
        return expected, _read_json(state_path)

    def _write_state(self, staging: StagingArea, state: Mapping[str, Any]) -> None:
        _write_json(self._state_path(staging), state)

    def _registry(self) -> dict[str, str]:
        if not self._registry_path.exists():
            return {}
        value = _read_json(self._registry_path)
        locations = value.get("locations", {})
        if not isinstance(locations, dict):
            raise ValueError("artifact location registry is malformed")
        return {str(key): str(path) for key, path in locations.items()}

    def _write_registry(self, locations: Mapping[str, str]) -> None:
        _write_json(self._registry_path, {"schema_version": 1, "locations": dict(locations)})

    def _grants(self) -> dict[str, dict[str, Any]]:
        if not self._grants_path.exists():
            return {}
        value = _read_json(self._grants_path)
        grants = value.get("grants", {})
        if not isinstance(grants, dict):
            raise ValueError("materialization grant registry is malformed")
        return {str(key): dict(record) for key, record in grants.items() if isinstance(record, dict)}

    def _write_grants(self, grants: Mapping[str, Mapping[str, Any]]) -> None:
        _write_json(self._grants_path, {"schema_version": 1, "grants": dict(grants)})

    @staticmethod
    def _expiration(ttl_seconds: int) -> str:
        if not isinstance(ttl_seconds, int) or isinstance(ttl_seconds, bool) or ttl_seconds < 1:
            raise ValueError("ttl_seconds must be a positive integer")
        return datetime.fromtimestamp(datetime.now(timezone.utc).timestamp() + ttl_seconds, timezone.utc).isoformat()

    @staticmethod
    def _is_expired(expires_at: str) -> bool:
        try:
            expiry = datetime.fromisoformat(expires_at)
        except (TypeError, ValueError) as error:
            raise ValueError("materialization grant has an invalid expiry") from error
        if expiry.tzinfo is None:
            raise ValueError("materialization grant expiry must include a timezone")
        return expiry <= datetime.now(timezone.utc)

    @staticmethod
    def _copy_directory(source: Path, target: Path) -> None:
        # Capture a content identity before copying, then require the private
        # copy to match it.  This closes the source-mutation window even for a
        # legacy artifact that has not yet been backfilled with a pinned
        # manifest.
        copy_verified_directory(source, target)

    @staticmethod
    def _manifest_from_state(state: Mapping[str, Any]) -> DirectoryContentManifest | None:
        raw = state.get("content_manifest")
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ValueError("artifact content manifest is malformed")
        try:
            return DirectoryContentManifest.from_dict(raw)
        except ArtifactIntegrityError as error:
            raise ValueError("artifact content manifest is invalid") from error

    def begin_staging(self, job_id: str, attempt_id: str) -> StagingArea:
        job_id, attempt_id = _stage_token(job_id, "job_id"), _stage_token(attempt_id, "attempt_id")
        path = self._staging_root / job_id / attempt_id
        state_path = self._state_root / job_id / f"{attempt_id}.json"
        if path.exists() or state_path.exists():
            raise FileExistsError(f"Staging area already exists for {job_id}/{attempt_id}")
        path.mkdir(parents=True)
        staging = StagingArea(job_id=job_id, attempt_id=attempt_id, path=path)
        self._write_state(staging, {"schema_version": 1, "status": "staging", "created_at": _now()})
        return staging

    def staging(self, job_id: str, attempt_id: str) -> StagingArea:
        """Reopen an existing control-plane staging handle without a path API.

        This is for orchestrator restart recovery only; workers never receive
        this object or its local path.
        """
        job_id, attempt_id = _stage_token(job_id, "job_id"), _stage_token(attempt_id, "attempt_id")
        staging = StagingArea(job_id, attempt_id, self._staging_root / job_id / attempt_id)
        self._checked_staging(staging)
        return staging

    def staging_path(self, staging: StagingArea) -> Path:
        path, state = self._checked_staging(staging)
        if state["status"] != "staging":
            raise ValueError(f"staging area is {state['status']}, not writable")
        return path

    def validate(self, staging: StagingArea, validator: Validator) -> ValidationResult:
        path, state = self._checked_staging(staging)
        if state["status"] not in {"staging", "validated"}:
            raise ValueError(f"cannot validate staging area in {state['status']} state")
        report = dict(validator(path))
        # A validator must explicitly assert validity; accidental empty reports
        # must never authorize publication.
        result = ValidationResult(valid=report.get("valid") is True, report=report)
        state.update({"status": "validated" if result.valid else "staging", "validated_at": _now(), "validation": report})
        self._write_state(staging, state)
        return result

    def seal(self, staging: StagingArea) -> None:
        path, state = self._checked_staging(staging)
        if state["status"] != "validated" or state.get("validation", {}).get("valid") is not True:
            raise ValueError("only a successfully validated staging area can be sealed")
        # Seal a storage-level byte inventory separately from any domain
        # manifest (such as a run's semantic fingerprint).  Publication and
        # every materialization will recheck these exact bytes.
        manifest = DirectoryContentManifest.create(path)
        state.update({
            "status": "sealed", "sealed_at": _now(),
            "content_manifest": manifest.to_dict(),
        })
        self._write_state(staging, state)

    def publish(self, staging: StagingArea, ref: ArtifactRef) -> Path:
        path, state = self._checked_staging(staging)
        if state["status"] != "sealed":
            raise ValueError("only a sealed staging area can be published")
        destination = self._canonical_path(ref)
        if destination.exists() or ref.uri in self._registry():
            raise FileExistsError(f"artifact is already published: {ref.uri}")
        expected_manifest = self._manifest_from_state(state)
        if expected_manifest is None:
            raise ValueError("sealed artifact has no content manifest")
        try:
            expected_manifest.verify(path)
        except ArtifactIntegrityError as error:
            raise ValueError("sealed artifact bytes changed before publication") from error
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Both paths are under root, making rename atomic on the local filesystem.
        os.replace(path, destination)
        try:
            expected_manifest.verify(destination)
        except ArtifactIntegrityError as error:
            # The immutable publication is intentionally retained for
            # diagnosis, but never becomes a valid delivery source.
            state.update({"status": "integrity_failed", "integrity_failed_at": _now()})
            self._write_state(staging, state)
            raise ValueError("published artifact bytes do not match its sealed manifest") from error
        state.update({"status": "published", "published_at": _now(), "artifact_ref": ref.to_dict()})
        self._write_state(staging, state)
        return destination

    def published_ref(self, job_id: str, attempt_id: str) -> ArtifactRef | None:
        """Recover a prior publication from the store-owned staging ledger.

        This is used only by control-plane startup reconciliation after a
        process interruption between the atomic filesystem publish and the
        corresponding database commit.  It exposes an identity, never a path.
        """
        job_id, attempt_id = _stage_token(job_id, "job_id"), _stage_token(attempt_id, "attempt_id")
        state_path = self._state_root / job_id / f"{attempt_id}.json"
        if not state_path.is_file():
            return None
        state = _read_json(state_path)
        if state.get("status") != "published" or not isinstance(state.get("artifact_ref"), Mapping):
            return None
        return ArtifactRef.from_dict(state["artifact_ref"])

    def register_existing(self, ref: ArtifactRef, path: str | Path) -> Path:
        """Register a legacy artifact without moving or copying its folder.

        This exists solely for offline migration/catalog recovery.  New
        work must use :meth:`ingest_file` (or staged publication) so a
        worker can be given a location-free, immutable artifact.
        """
        location = Path(path).expanduser().resolve()
        if not location.is_dir():
            raise NotADirectoryError(location)
        locations = self._registry()
        if ref.uri in locations:
            raise FileExistsError(f"artifact is already registered: {ref.uri}")
        if self._canonical_path(ref).exists():
            raise FileExistsError(f"artifact is already published: {ref.uri}")
        locations[ref.uri] = str(location)
        self._write_registry(locations)
        return location

    def ingest_file(self, ref: ArtifactRef, source: str | Path) -> Path:
        """Copy one immutable input file into this store and publish it.

        File inputs are represented as an artifact *directory* containing a
        stable ``payload`` name and a small manifest.  This lets every worker
        consume ``<materialized-artifact>/payload`` without learning either
        the submitter's filename or any control-plane filesystem path.

        A new portable input must be SHA-pinned.  The fingerprint is checked
        against the copied bytes before publication, making a mismatch fail
        before any worker can receive a grant.
        """
        if ref.fingerprint_sha256 is None:
            raise ValueError("portable file artifacts must include fingerprint_sha256")
        raw_source = Path(source).expanduser()
        if raw_source.is_symlink() or not raw_source.is_file():
            raise ValueError("portable file artifact source must be a regular, non-symlink file")
        source_path = raw_source.resolve()
        existing = self._canonical_path(ref)
        if existing.exists():
            # Content-addressed file inputs are safely reusable across plans.
            # A collision is never silently accepted: only the exact sealed
            # payload pinned by this reference may satisfy a later request.
            payload = existing / "payload"
            manifest = existing / "manifest.json"
            if payload.is_file() and manifest.is_file() and sha256_file(payload) == ref.fingerprint_sha256:
                return existing
            raise FileExistsError(f"published artifact conflicts with {ref.uri}")
        if ref.uri in self._registry():
            raise FileExistsError(f"legacy registration conflicts with {ref.uri}")

        # Generated identifiers are internal staging handles, never user file
        # names.  `payload` is copied with a descriptor opened only after the
        # source has been checked so the immutable digest is authoritative.
        staging = self.begin_staging(f"ingest-{uuid.uuid4().hex}", "file")
        try:
            target = self.staging_path(staging) / "payload"
            digest = sha256()
            size = 0
            with source_path.open("rb") as input_file, target.open("xb") as output_file:
                while chunk := input_file.read(1024 * 1024):
                    digest.update(chunk)
                    size += len(chunk)
                    output_file.write(chunk)
            actual = digest.hexdigest()
            if actual != ref.fingerprint_sha256:
                raise ValueError("portable file artifact fingerprint does not match source bytes")
            _write_json(self.staging_path(staging) / "manifest.json", {
                "schema_version": 2,
                "digest_kind": "raw_payload_sha256",
                "kind": ref.kind,
                "source_name": source_path.name,
                "sha256": actual,
                "size_bytes": size,
            })
            validation = self.validate(staging, lambda path: {
                "valid": (path / "payload").is_file()
                and (path / "manifest.json").is_file()
                and sha256_file(path / "payload") == ref.fingerprint_sha256,
            })
            if not validation.valid:  # Defensive: keeps the publication protocol explicit.
                raise ValueError("portable file artifact did not validate")
            self.seal(staging)
            return self.publish(staging, ref)
        except Exception:
            # Staging records are intentionally retained for crash diagnosis;
            # no failed candidate is resolvable or publishable.
            raise

    def resolve(self, ref: ArtifactRef) -> Path:
        registered = self._registry().get(ref.uri)
        location = Path(registered).expanduser().resolve() if registered else self._canonical_path(ref)
        if not location.is_dir():
            raise FileNotFoundError(f"artifact is not available: {ref.uri}")
        return location

    def materialize(self, ref: ArtifactRef, destination: str | Path) -> Path:
        """Copy a published directory into worker scratch without exposing roots."""
        source = self.resolve(ref)
        target = Path(destination).expanduser().resolve()
        if target.exists():
            raise FileExistsError(f"materialization destination already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = self._published_manifest(ref)
        try:
            copy_verified_directory(source, target, expected=expected)
        except ArtifactIntegrityError as error:
            raise ValueError(f"artifact integrity verification failed for {ref.uri}: {error}") from error
        self._verify_portable_file_payload(ref, target)
        return target

    def _published_manifest(self, ref: ArtifactRef) -> DirectoryContentManifest | None:
        """Return sealed byte identity for a new artifact, if available.

        Registered historical folders intentionally return ``None``: their
        old ``fingerprint_sha256`` may be a semantic dataset identity and must
        not be reinterpreted as a raw directory hash.  They still get a
        copy-time snapshot verification until an explicit backfill pins one.
        """
        if ref.uri in self._registry():
            return None
        for state_path in self._state_root.glob("*/*.json"):
            try:
                state = _read_json(state_path)
            except (OSError, json.JSONDecodeError):
                continue
            if state.get("artifact_ref") == ref.to_dict():
                return self._manifest_from_state(state)
        return None

    @staticmethod
    def _verify_portable_file_payload(ref: ArtifactRef, directory: Path) -> None:
        """Validate only explicitly typed raw-payload legacy envelopes."""
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            return
        try:
            manifest = _read_json(manifest_path)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("portable file artifact manifest is unreadable") from error
        if manifest.get("digest_kind") != "raw_payload_sha256":
            return
        payload = directory / "payload"
        expected = manifest.get("sha256")
        if not isinstance(expected, str) or sha256_file(payload) != expected:
            raise ValueError("portable file artifact payload does not match its raw digest")
        if ref.fingerprint_sha256 is not None and expected != ref.fingerprint_sha256:
            raise ValueError("portable file artifact raw digest does not match its pinned reference")

    def issue_materialization_grant(
        self, ref: ArtifactRef, *, lease_id: str, job_id: str, ttl_seconds: int = 600
    ) -> MaterializationGrant:
        """Issue a single-use artifact grant bound to one durable work lease.

        Resolution happens before issue so a worker never receives a usable
        token for an unavailable artifact.  No resolved local path is stored
        in the grant registry or returned in the contract.
        """
        lease_id = _stage_token(lease_id, "lease_id")
        job_id = _stage_token(job_id, "job_id")
        self.resolve(ref)
        expected_manifest = self._published_manifest(ref)
        grant_id = uuid.uuid4().hex
        token = token_urlsafe(32)
        expires_at = self._expiration(ttl_seconds)
        record = {
            "schema_version": 1,
            "lease_id": lease_id,
            "job_id": job_id,
            "ref": ref.to_dict(),
            **({"content_manifest_sha256": expected_manifest.digest_sha256} if expected_manifest else {}),
            "token_sha256": sha256(token.encode("utf-8")).hexdigest(),
            "expires_at": expires_at,
            "issued_at": _now(),
            "status": "issued",
        }
        with self._grant_lock:
            grants = self._grants()
            grants[grant_id] = record
            self._write_grants(grants)
        return MaterializationGrant(
            grant_id, lease_id, job_id, ref, expires_at, token,
            expected_manifest.digest_sha256 if expected_manifest else None,
        )

    def _grant_identity(self, grant: MaterializationGrant | str) -> tuple[str, str]:
        if isinstance(grant, MaterializationGrant):
            return grant.grant_id, grant.token
        if not isinstance(grant, str) or not grant:
            raise ValueError("materialization grant must include a bearer token")
        digest = sha256(grant.encode("utf-8")).hexdigest()
        with self._grant_lock:
            matches = [grant_id for grant_id, record in self._grants().items() if record.get("token_sha256") == digest]
        if len(matches) != 1:
            raise PermissionError("materialization grant is unknown")
        return matches[0], grant

    def materialize_grant(
        self,
        grant: MaterializationGrant | str,
        destination: str | Path,
        *,
        lease_id: str,
        job_id: str,
    ) -> Path:
        """Consume a scoped grant and copy its artifact to worker scratch.

        The matching lease and job must be supplied by the lease handler, not
        trusted from a worker payload.  A failed copy restores the grant to
        ``issued`` so a transient local-storage failure can be retried.
        """
        lease_id = _stage_token(lease_id, "lease_id")
        job_id = _stage_token(job_id, "job_id")
        grant_id, token = self._grant_identity(grant)
        digest = sha256(token.encode("utf-8")).hexdigest()
        with self._grant_lock:
            grants = self._grants()
            record = grants.get(grant_id)
            if record is None or record.get("token_sha256") != digest:
                raise PermissionError("materialization grant is invalid")
            if record.get("lease_id") != lease_id or record.get("job_id") != job_id:
                raise PermissionError("materialization grant is outside this lease scope")
            if self._is_expired(str(record.get("expires_at", ""))):
                record["status"] = "expired"
                grants[grant_id] = record
                self._write_grants(grants)
                raise PermissionError("materialization grant has expired")
            if record.get("status") != "issued":
                raise PermissionError("materialization grant has already been used")
            record["status"] = "materializing"
            record["materializing_at"] = _now()
            grants[grant_id] = record
            self._write_grants(grants)
        try:
            ref = ArtifactRef.from_dict(record["ref"])
            materialized = self.materialize(ref, destination)
        except Exception:
            with self._grant_lock:
                grants = self._grants()
                current = grants.get(grant_id)
                if current is not None and current.get("status") == "materializing":
                    current["status"] = "issued"
                    current.pop("materializing_at", None)
                    grants[grant_id] = current
                    self._write_grants(grants)
            raise
        with self._grant_lock:
            grants = self._grants()
            current = grants.get(grant_id)
            if current is None or current.get("status") != "materializing":
                raise RuntimeError("materialization grant state changed unexpectedly")
            current.update({"status": "used", "used_at": _now()})
            grants[grant_id] = current
            self._write_grants(grants)
        return materialized

    def _claim_grant(
        self, grant: MaterializationGrant | str, *, lease_id: str, job_id: str,
        expected_grant_id: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Atomically reserve a grant for an irreversible delivery attempt."""
        lease_id = _stage_token(lease_id, "lease_id")
        job_id = _stage_token(job_id, "job_id")
        grant_id, token = self._grant_identity(grant)
        if expected_grant_id is not None and grant_id != expected_grant_id:
            raise PermissionError("materialization grant does not match delivery request")
        digest = sha256(token.encode("utf-8")).hexdigest()
        with self._grant_lock:
            grants = self._grants()
            record = grants.get(grant_id)
            if record is None or record.get("token_sha256") != digest:
                raise PermissionError("materialization grant is invalid")
            if record.get("lease_id") != lease_id or record.get("job_id") != job_id:
                raise PermissionError("materialization grant is outside this lease scope")
            if self._is_expired(str(record.get("expires_at", ""))):
                record["status"] = "expired"
                grants[grant_id] = record
                self._write_grants(grants)
                raise PermissionError("materialization grant has expired")
            if record.get("status") != "issued":
                raise PermissionError("materialization grant has already been used")
            record["status"] = "materializing"
            record["materializing_at"] = _now()
            grants[grant_id] = record
            self._write_grants(grants)
        return grant_id, record

    def _restore_claimed_grant(self, grant_id: str) -> None:
        with self._grant_lock:
            grants = self._grants()
            current = grants.get(grant_id)
            if current is not None and current.get("status") == "materializing":
                current["status"] = "issued"
                current.pop("materializing_at", None)
                grants[grant_id] = current
                self._write_grants(grants)

    def _consume_claimed_grant(self, grant_id: str) -> None:
        with self._grant_lock:
            grants = self._grants()
            current = grants.get(grant_id)
            if current is None or current.get("status") != "materializing":
                raise RuntimeError("materialization grant state changed unexpectedly")
            current.update({"status": "used", "used_at": _now()})
            grants[grant_id] = current
            self._write_grants(grants)

    @staticmethod
    def _write_archive(source: Path, output: Any) -> None:
        """Write a deterministic, symlink-free gzip tar archive.

        Artifacts are directory contracts.  The single top-level ``artifact``
        directory prevents a malicious filename from being extracted beside a
        worker's selected destination.  We enumerate and add entries manually
        rather than relying on recursive ``tar.add`` so links and special
        files are rejected at the delivery boundary.
        """
        entries = [source, *sorted(source.rglob("*"), key=lambda path: path.as_posix())]
        with tarfile.open(fileobj=output, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
            for item in entries:
                if item.is_symlink():
                    raise ValueError("artifact contains a symlink and cannot be delivered")
                relative = item.relative_to(source)
                arcname = "artifact" if relative == Path(".") else f"artifact/{relative.as_posix()}"
                info = archive.gettarinfo(str(item), arcname=arcname)
                if info.isdir():
                    archive.addfile(info)
                elif info.isreg():
                    # Recheck just before opening; this closes the common
                    # local replacement race without following a link.
                    if item.is_symlink():
                        raise ValueError("artifact contains a symlink and cannot be delivered")
                    with item.open("rb") as payload:
                        archive.addfile(info, payload)
                else:
                    raise ValueError("artifact contains a non-regular file and cannot be delivered")

    def stream_grant_archive(
        self,
        grant: MaterializationGrant | str,
        *,
        lease_id: str,
        job_id: str,
        expected_grant_id: str | None = None,
        chunk_size: int = 64 * 1024,
    ) -> Iterator[bytes]:
        """Return a one-use gzip archive stream scoped to an active lease.

        The archive is built in an unnamed temporary file before HTTP headers
        are sent, so invalid/missing artifacts fail as a normal client error
        rather than a partial 200 response.  It is then yielded incrementally;
        neither a source location nor a raw bearer value enters durable state.
        """
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1024:
            raise ValueError("chunk_size must be an integer of at least 1024")
        grant_id, record = self._claim_grant(
            grant, lease_id=lease_id, job_id=job_id, expected_grant_id=expected_grant_id,
        )
        temporary = tempfile.TemporaryFile(mode="w+b")
        try:
            ref = ArtifactRef.from_dict(record["ref"])
            # Never archive a live store directory directly.  Build the tar
            # from a private snapshot that has matched the sealed inventory,
            # so the bytes delivered are exactly the bytes we verified.
            with tempfile.TemporaryDirectory(prefix="oracle-artifact-snapshot-") as snapshot_root:
                snapshot = Path(snapshot_root) / "artifact"
                try:
                    copy_verified_directory(
                        self.resolve(ref), snapshot, expected=self._published_manifest(ref),
                    )
                    self._verify_portable_file_payload(ref, snapshot)
                except ArtifactIntegrityError as error:
                    raise ValueError(f"artifact integrity verification failed for {ref.uri}: {error}") from error
                self._write_archive(snapshot, temporary)
            size = temporary.tell()
            temporary.seek(0)
        except Exception:
            temporary.close()
            self._restore_claimed_grant(grant_id)
            raise

        def stream() -> Iterator[bytes]:
            consumed = False
            try:
                if size == 0:
                    self._consume_claimed_grant(grant_id)
                    consumed = True
                    return
                sent = 0
                while chunk := temporary.read(chunk_size):
                    sent += len(chunk)
                    if sent == size:
                        self._consume_claimed_grant(grant_id)
                        consumed = True
                    yield chunk
            finally:
                temporary.close()
                if not consumed:
                    # A client disconnect does not burn its short-lived
                    # authority; it can resume by requesting the same grant.
                    self._restore_claimed_grant(grant_id)

        return stream()
