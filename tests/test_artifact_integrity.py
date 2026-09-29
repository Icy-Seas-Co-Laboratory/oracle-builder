from __future__ import annotations

from hashlib import sha256
import io
import tarfile

import pytest

from oracle_data_contracts.artifacts import (
    ArtifactIntegrityError,
    ArtifactRef,
    DirectoryContentManifest,
    copy_verified_directory,
    directory_content_digest,
)
from oracle_builder.orchestration.storage import LocalArtifactStore


def _directory(root):
    root.mkdir()
    (root / "payload.bin").write_bytes(b"first payload")
    (root / "nested").mkdir()
    (root / "nested" / "metadata.json").write_text('{"version":1}\n', encoding="utf-8")
    return root


def test_directory_manifest_is_deterministic_and_rejects_missing_extra_changed_and_links(tmp_path):
    root = _directory(tmp_path / "artifact")
    first = DirectoryContentManifest.create(root)
    assert first.digest_sha256 == DirectoryContentManifest.create(root).digest_sha256
    assert directory_content_digest(root) == first.digest_sha256
    assert DirectoryContentManifest.from_dict(first.to_dict()) == first

    (root / "payload.bin").write_bytes(b"changed")
    with pytest.raises(ArtifactIntegrityError, match="changed=payload.bin"):
        first.verify(root)
    (root / "payload.bin").write_bytes(b"first payload")
    (root / "extra").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="extra=extra"):
        first.verify(root)
    (root / "extra").unlink()
    (root / "nested" / "metadata.json").unlink()
    with pytest.raises(ArtifactIntegrityError, match="missing=nested/metadata.json"):
        first.verify(root)
    (root / "nested" / "metadata.json").write_text('{"version":1}\n', encoding="utf-8")
    (root / "link").symlink_to(tmp_path / "outside")
    with pytest.raises(ArtifactIntegrityError, match="symlink"):
        first.verify(root)


def test_verified_copy_requires_the_private_snapshot_to_match_source_identity(tmp_path):
    source = _directory(tmp_path / "source")
    expected = DirectoryContentManifest.create(source)
    (source / "payload.bin").write_bytes(b"changed after pinning")
    with pytest.raises(ArtifactIntegrityError, match="pinned directory manifest"):
        copy_verified_directory(source, tmp_path / "scratch", expected=expected)
    assert not (tmp_path / "scratch").exists()


def test_store_rejects_tampered_published_bytes_and_binds_new_grants_to_directory_identity(tmp_path):
    store = LocalArtifactStore(tmp_path / "store")
    source = tmp_path / "input.sqlite"
    source.write_bytes(b"known input")
    raw_digest = sha256(source.read_bytes()).hexdigest()
    ref = ArtifactRef("dataset", "dataset-1", "r1", raw_digest)
    published = store.ingest_file(ref, source)

    (published / "payload").write_bytes(b"tampered")
    # The grant must retain the identity sealed at publication, rather than
    # recomputing a digest from the now-mutable store directory.
    grant = store.issue_materialization_grant(ref, lease_id="lease-1", job_id="job-1")
    assert grant.content_manifest_sha256
    assert grant.delivery_dict()["content_manifest_sha256"] == grant.content_manifest_sha256
    assert grant.content_manifest_sha256 != DirectoryContentManifest.create(published).digest_sha256
    with pytest.raises(ValueError, match="integrity verification failed"):
        store.materialize_grant(grant, tmp_path / "scratch", lease_id="lease-1", job_id="job-1")
    # An integrity failure does not consume the authority, which lets a
    # control-plane repair/backfill path retry deliberately.
    assert "\"status\": \"issued\"" in (tmp_path / "store" / ".materialization-grants.json").read_text()


def test_archive_is_built_from_the_verified_snapshot_not_a_mutable_store_folder(tmp_path):
    store = LocalArtifactStore(tmp_path / "store")
    source = tmp_path / "input.sqlite"
    source.write_bytes(b"known input")
    ref = ArtifactRef("dataset", "dataset-1", fingerprint_sha256=sha256(source.read_bytes()).hexdigest())
    store.ingest_file(ref, source)
    grant = store.issue_materialization_grant(ref, lease_id="lease-1", job_id="job-1")
    archive_bytes = b"".join(store.stream_grant_archive(grant, lease_id="lease-1", job_id="job-1"))
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        assert archive.extractfile("artifact/payload").read() == b"known input"


def test_legacy_semantic_fingerprint_is_not_misread_as_a_raw_payload_hash(tmp_path):
    legacy = _directory(tmp_path / "legacy")
    # Dataset semantic fingerprints are intentionally not raw file hashes.
    ref = ArtifactRef("dataset", "legacy", fingerprint_sha256="a" * 64)
    store = LocalArtifactStore(tmp_path / "store")
    store.register_existing(ref, legacy)
    scratch = store.materialize(ref, tmp_path / "scratch")
    assert (scratch / "payload.bin").read_bytes() == b"first payload"
