from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from oracle_data_contracts.artifacts import ArtifactRef
from oracle_builder.orchestration.storage import LocalArtifactStore, MaterializationGrant, S3ReplicatedArtifactStore
from oracle_builder.orchestration.service import Orchestrator


def _valid_candidate(path):
    return {"valid": (path / "result.txt").read_text(encoding="utf-8") == "finished\n"}


def test_artifact_ref_is_portable_round_trip_and_rejects_paths():
    ref = ArtifactRef(
        "dataset", "f5b0c22a-9e14-48a1-a2f8-9c67c199a0dc", "revision-2", "a" * 64
    )
    assert ref.uri == f"dataset:f5b0c22a-9e14-48a1-a2f8-9c67c199a0dc@revision-2#{'a' * 64}"
    assert ArtifactRef.parse(ref.uri) == ref
    assert ArtifactRef.from_dict(ref.to_dict()) == ref
    with pytest.raises(ValueError):
        ArtifactRef.parse("dataset:/host-specific/path")


def test_local_store_requires_validation_and_seals_before_atomic_publication(tmp_path):
    store = LocalArtifactStore(tmp_path / "oracle-storage")
    staging = store.begin_staging("job-1", "attempt-1")
    (store.staging_path(staging) / "result.txt").write_text("finished\n", encoding="utf-8")
    ref = ArtifactRef("model_run", "run-1", "1")

    with pytest.raises(ValueError, match="sealed"):
        store.publish(staging, ref)
    validation = store.validate(staging, _valid_candidate)
    assert validation.valid
    store.seal(staging)
    published = store.publish(staging, ref)

    assert published == tmp_path / "oracle-storage" / "model_run" / "run-1" / "1"
    assert (published / "result.txt").read_text(encoding="utf-8") == "finished\n"
    assert not staging.path.exists()
    assert store.resolve(ref) == published


def test_local_store_does_not_publish_invalid_candidate_or_overwrite(tmp_path):
    store = LocalArtifactStore(tmp_path / "oracle-storage")
    staging = store.begin_staging("job-1", "attempt-1")
    (store.staging_path(staging) / "result.txt").write_text("wrong\n", encoding="utf-8")
    assert not store.validate(staging, _valid_candidate).valid
    with pytest.raises(ValueError, match="validated"):
        store.seal(staging)

    ref = ArtifactRef("model_run", "run-1")
    second = store.begin_staging("job-1", "attempt-2")
    (store.staging_path(second) / "result.txt").write_text("finished\n", encoding="utf-8")
    store.validate(second, _valid_candidate)
    store.seal(second)
    store.publish(second, ref)
    third = store.begin_staging("job-1", "attempt-3")
    (store.staging_path(third) / "result.txt").write_text("finished\n", encoding="utf-8")
    store.validate(third, _valid_candidate)
    store.seal(third)
    with pytest.raises(FileExistsError):
        store.publish(third, ref)
    assert third.path.is_dir(), "failed publication must leave the sealed candidate recoverable"


def test_local_store_registers_existing_folder_without_moving_and_materializes(tmp_path):
    legacy = tmp_path / "project-runs" / "old-model"
    legacy.mkdir(parents=True)
    (legacy / "artifact.json").write_text("{}\n", encoding="utf-8")
    store = LocalArtifactStore(tmp_path / "oracle-storage")
    ref = ArtifactRef("model_run", "legacy-1")

    assert store.register_existing(ref, legacy) == legacy
    assert store.resolve(ref) == legacy
    scratch = store.materialize(ref, tmp_path / "worker-scratch" / "input")
    assert (scratch / "artifact.json").read_text(encoding="utf-8") == "{}\n"
    assert legacy.exists(), "registering must not relocate compatibility artifacts"


def test_materialization_grant_is_scoped_single_use_and_never_persists_bearer_token(tmp_path):
    source = tmp_path / "legacy" / "dataset"
    source.mkdir(parents=True)
    (source / "payload.txt").write_text("portable\n", encoding="utf-8")
    store = LocalArtifactStore(tmp_path / "oracle-storage")
    ref = ArtifactRef("dataset", "dataset-1", "revision-1")
    store.register_existing(ref, source)

    grant = store.issue_materialization_grant(ref, lease_id="lease-1", job_id="job-1")
    assert isinstance(grant, MaterializationGrant)
    assert "token" not in grant.to_dict()
    assert grant.delivery_dict()["token"] == grant.token
    persisted = (tmp_path / "oracle-storage" / ".materialization-grants.json").read_text(encoding="utf-8")
    assert grant.token not in persisted
    assert "token_sha256" in persisted

    with pytest.raises(PermissionError, match="scope"):
        store.materialize_grant(grant, tmp_path / "wrong-scratch", lease_id="other-lease", job_id="job-1")
    scratch = store.materialize_grant(grant.token, tmp_path / "worker-scratch" / "input", lease_id="lease-1", job_id="job-1")
    assert (scratch / "payload.txt").read_text(encoding="utf-8") == "portable\n"
    with pytest.raises(PermissionError, match="already"):
        store.materialize_grant(grant, tmp_path / "retry-scratch", lease_id="lease-1", job_id="job-1")


def test_materialization_grant_expiry_and_symlink_protection(tmp_path):
    source = tmp_path / "legacy" / "dataset"
    source.mkdir(parents=True)
    (source / "payload.txt").write_text("portable\n", encoding="utf-8")
    store = LocalArtifactStore(tmp_path / "oracle-storage")
    ref = ArtifactRef("dataset", "dataset-1")
    store.register_existing(ref, source)
    grant = store.issue_materialization_grant(ref, lease_id="lease-1", job_id="job-1")

    grants_path = tmp_path / "oracle-storage" / ".materialization-grants.json"
    state = json.loads(grants_path.read_text(encoding="utf-8"))
    state["grants"][grant.grant_id]["expires_at"] = "2000-01-01T00:00:00+00:00"
    grants_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(PermissionError, match="expired"):
        store.materialize_grant(grant, tmp_path / "scratch", lease_id="lease-1", job_id="job-1")

    fresh = store.issue_materialization_grant(ref, lease_id="lease-2", job_id="job-2")
    (source / "outside").symlink_to(tmp_path / "not-an-artifact")
    with pytest.raises(ValueError, match="symlink"):
        store.materialize_grant(fresh, tmp_path / "scratch-2", lease_id="lease-2", job_id="job-2")
    # A failed copy must not burn a grant; removing the unsafe input permits a
    # retry with the same lease-scoped bearer token.
    (source / "outside").unlink()
    assert store.materialize_grant(fresh, tmp_path / "scratch-3", lease_id="lease-2", job_id="job-2").is_dir()


def test_orchestrator_owns_an_injectable_artifact_store(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = LocalArtifactStore(tmp_path / "storage")
    orchestrator = Orchestrator(
        tmp_path / "orchestrator.sqlite", workspace_root=workspace, artifact_store=store,
    )
    assert orchestrator.artifact_store is store


def test_s3_replica_keeps_folder_contract_and_writes_completion_marker_last(tmp_path):
    class FakeS3:
        def __init__(self):
            self.uploads: list[tuple[str, str, str]] = []
            self.markers: list[tuple[str, str, bytes]] = []

        def upload_file(self, filename, bucket, key):
            self.uploads.append((filename, bucket, key))

        def put_object(self, *, Bucket, Key, Body):
            self.markers.append((Bucket, Key, Body))

    local = LocalArtifactStore(tmp_path / "oracle-storage")
    remote = FakeS3()
    store = S3ReplicatedArtifactStore(local, bucket="oracle-artifacts", prefix="dev", client=remote)
    staging = store.begin_staging("job-1", "attempt-1")
    (store.staging_path(staging) / "result.txt").write_text("finished\n", encoding="utf-8")
    ref = ArtifactRef("model_run", "run-1", "1")
    assert store.validate(staging, _valid_candidate).valid
    store.seal(staging)
    published = store.publish(staging, ref)

    assert published == local.resolve(ref)
    assert [key for _source, _bucket, key in remote.uploads] == ["dev/model_run/run-1/1/result.txt"]
    assert remote.markers[0][1] == "dev/model_run/run-1/1/.oracle-artifact.json"
    marker = json.loads(remote.markers[0][2])
    assert marker["artifact_ref"] == ref.to_dict()
    assert marker["files"][0]["path"] == "result.txt"


def test_s3_replica_verifies_and_restores_a_missing_local_artifact(tmp_path):
    class Body:
        def __init__(self, value): self.value = value
        def read(self): return self.value

    class FakeS3:
        def __init__(self): self.objects = {}
        def upload_file(self, filename, bucket, key): self.objects[(bucket, key)] = Path(filename).read_bytes()
        def put_object(self, *, Bucket, Key, Body): self.objects[(Bucket, Key)] = Body
        def get_object(self, *, Bucket, Key): return {"Body": Body(self.objects[(Bucket, Key)])}
        def download_file(self, bucket, key, filename): Path(filename).write_bytes(self.objects[(bucket, key)])

    local = LocalArtifactStore(tmp_path / "oracle-storage")
    remote = FakeS3()
    store = S3ReplicatedArtifactStore(local, bucket="oracle-artifacts", prefix="dev", client=remote)
    staging = store.begin_staging("job-1", "attempt-1")
    (store.staging_path(staging) / "result.txt").write_text("finished\n", encoding="utf-8")
    ref = ArtifactRef("model_run", "run-1", "1")
    assert store.validate(staging, _valid_candidate).valid
    store.seal(staging)
    store.publish(staging, ref)
    assert store.verify_replica(ref)["files"] == 1

    shutil.rmtree(local.resolve(ref))
    restored = store.restore(ref)
    assert restored == local.resolve(ref)
    assert (restored / "result.txt").read_text(encoding="utf-8") == "finished\n"
    with pytest.raises(FileExistsError): store.restore(ref)


def test_orchestrator_persists_replica_failure_and_retry_state(tmp_path):
    class FailingS3:
        def upload_file(self, *_args): raise OSError("object store unavailable")
        def put_object(self, **_kwargs): raise AssertionError("marker must not be written after a failed upload")

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    local = LocalArtifactStore(tmp_path / "storage")
    store = S3ReplicatedArtifactStore(local, bucket="oracle-artifacts", client=FailingS3())
    orchestrator = Orchestrator(tmp_path / "orchestrator.sqlite", workspace_root=workspace, artifact_store=store)
    source = tmp_path / "source.bin"
    source.write_bytes(b"input")
    ref = ArtifactRef("dataset", "input-1", fingerprint_sha256=__import__("hashlib").sha256(b"input").hexdigest())
    store.ingest_file(ref, source)
    recorded = orchestrator.record_artifact_replica(ref)
    assert recorded["status"] == "failed"
    assert recorded["attempts"] == 0
    retried = orchestrator.replicate_artifact(ref)
    assert retried["status"] == "failed"
    assert retried["attempts"] == 1
