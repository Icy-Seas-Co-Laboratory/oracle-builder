"""CPU-free integration regressions for immutable sequential training units."""
from __future__ import annotations

import io
import hashlib
import json
from pathlib import Path
import uuid
import zipfile

import pytest

from oracle_builder.artifacts import (
    create_run_artifact,
    read_run_config,
    read_run_manifest,
    seal_run_artifact,
    update_run_artifact,
    validate_run_artifact,
)
from oracle_builder.orchestration.service import Orchestrator
from oracle_builder.training.control import FileSegmentControl
from oracle_builder.training.recovery import RECOVERY_SCHEMA, recovery_config_hash
from oracle_builder.worker.control import WorkerControl
from oracle_data_contracts.artifacts import ArtifactRef, directory_content_digest
from oracle_data_contracts.work_units import WorkUnit, WorkUnitV2


def _segment_artifact(
    root: Path, *, run_id: str, completed_epoch: int, stop_epoch: int,
    interrupted: bool = False, next_phase: str | None = None,
) -> Path:
    root.mkdir(parents=True)
    source = root / "source.toml"
    source.write_text('[run]\ntask = "classification"\nmodel = "simple_cnn"\n', encoding="utf-8")
    config = {
        "run": {"run_id": run_id, "task": "classification", "model": "simple_cnn"},
        "data": {"input_shape": [8, 8, 1], "num_classes": 2},
        "training": {"loss": "sparse_categorical_crossentropy"},
        "dataset": {"dataset_id": str(uuid.uuid4()), "fingerprint_sha256": "a" * 64},
    }
    run = root / "run"
    create_run_artifact(run, run_id=run_id, name="segment", config=config, source_config=source)
    # Construct the smallest authentic recovery generation: resume validation
    # checks its identity, resolved-contract hash, and exact checkpoint bytes.
    # The model payload is deliberately opaque here; these scheduler tests do
    # not execute TensorFlow.
    generation_id = "scheduler-fixture"
    model = run / "model" / "recovery" / "generations" / generation_id / "model.keras"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"scheduler recovery checkpoint")
    checkpoint_digest = hashlib.sha256(model.read_bytes()).hexdigest()
    latest = run / "model" / "recovery" / "latest.keras"
    latest.write_bytes(model.read_bytes())
    manifest = read_run_manifest(run)
    resolved = read_run_config(run)
    recovery = {
        "schema": RECOVERY_SCHEMA,
        "artifact_id": manifest["artifact_id"],
        "run_id": run_id,
        "phase": "supervised",
        "completed_epoch": completed_epoch,
        "model_path": f"model/recovery/generations/{generation_id}/model.keras",
        "model_sha256": checkpoint_digest,
        "config_sha256": recovery_config_hash(resolved),
        "continuation": {"boundary": "epoch", "model_optimizer_state": "preserved"},
        "generation": {
            "id": generation_id,
            "path": f"model/recovery/generations/{generation_id}",
            "schema": "oracle_builder_training_checkpoint_generation/v1",
        },
    }
    (model.parent / "state.json").write_text(json.dumps(recovery), encoding="utf-8")
    (run / "model" / "recovery" / "state.json").write_text(json.dumps(recovery), encoding="utf-8")
    (root / "run" / "segment-result.json").write_text(json.dumps({
        "schema": "oracle_builder_training_segment_result/v1",
        "phase": "train",
        "cursor": {"completed_epoch": completed_epoch},
        "checkpoint": "model/recovery/state.json",
        # The workflow advances to finalization only after the full run
        # budget, not merely after this unit's boundary.
        "next_phase": next_phase or ("finalize" if completed_epoch >= 2 else "train"),
        "interrupted": interrupted,
    }), encoding="utf-8")
    update_run_artifact(root / "run", status="interrupted")
    seal_run_artifact(root / "run")
    assert validate_run_artifact(root / "run")["valid"]
    return root / "run"


def _zip_directory(root: Path) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for item in sorted(root.rglob("*")):
            if item.is_file():
                archive.writestr(item.relative_to(root).as_posix(), item.read_bytes())
    return output.getvalue()


def _sequential_lease(tmp_path, *, unit_epochs: int = 1):
    orch = Orchestrator(tmp_path / "control.sqlite", workspace_root=tmp_path)
    pool = orch.create_worker_pool(name="sequential", allowed_actions=["train"])
    worker = orch.register_worker(
        pool_id=pool["pool_id"], name="worker", registration_token=pool["registration_token"],
        capabilities={"actions": ["train"], "worker_control_v1": True, "execution_instance_id": "boot"},
    )
    run_id, job_id = str(uuid.uuid4()), str(uuid.uuid4())
    unit = WorkUnitV2(
        run_id=run_id, work_unit_id=job_id, sequence=0, phase="train", predecessor_work_unit_id=None,
        inputs={"dataset": ArtifactRef("dataset", "dataset", "r1", "a" * 64)},
        configuration=ArtifactRef("configuration", "config", "r1", "b" * 64), resources={},
        parameters={"execution_segment": {"phase": "train", "start_epoch": 0, "stop_epoch": unit_epochs, "total_epochs": 2}},
        start_cursor={"completed_epoch": 0}, stop_boundary={"completed_epoch": unit_epochs},
        output_contract={"kind": "training_checkpoint", "schema_version": 1},
        compatibility={"worker_control_v1": True},
    )
    now = "2026-09-29T00:00:00+00:00"
    with orch._connection() as db:
        db.execute(
            """INSERT INTO jobs(job_id,specification_id,oracle_serve_url,action,parameters_json,resources_json,
               work_unit_json,work_unit_sha256,worker_pool_id,status,remote_status,submitted_at,updated_at)
               VALUES(?,NULL,?,'train','{}','{}',?,?,?,'queued','queued',?,?)""",
            (job_id, f"pull://{pool['pool_id']}", unit.canonical_json(), unit.sha256, pool["pool_id"], now, now),
        )
        db.execute(
            """INSERT INTO execution_runs(run_id,specification_id,queued_run_id,worker_pool_id,status,current_job_id,
               completed_epoch,total_epochs,unit_epochs,checkpoint_ref_json,created_at,updated_at)
               VALUES(?,?,NULL,?,'queued',?,0,2,?,NULL,?,?)""",
            (run_id, run_id, pool["pool_id"], job_id, unit_epochs, now, now),
        )
    lease = orch.acquire_next_worker_lease(worker_id=worker["worker_id"], worker_token=worker["worker_token"], ttl_seconds=60)
    assert lease is not None
    credentials = {"lease_id": lease["lease_id"], "worker_id": worker["worker_id"], "worker_token": worker["worker_token"], "lease_token": lease["lease_token"]}
    return orch, worker, unit, lease, credentials


def _complete_segment(orch, credentials, artifact: Path):
    acknowledged = orch.acknowledge_worker_lease(**credentials)
    orch.upload_worker_lease_output_archive(**credentials, archive=_zip_directory(artifact))
    return orch.complete_worker_lease(**credentials, success=True)


def _units_for_run(orch, run_id: str):
    with orch._connection() as db:
        rows = db.execute("SELECT job_id,status,work_unit_json FROM jobs ORDER BY submitted_at, job_id").fetchall()
    return [
        {"job_id": row["job_id"], "status": row["status"], "work_unit": json.loads(row["work_unit_json"])}
        for row in rows
        if json.loads(row["work_unit_json"] or "{}").get("run_id") == run_id
    ]


def test_worker_directive_file_is_consumed_at_the_training_safe_boundary(tmp_path):
    """The worker's durable directive wire must match the Keras callback wire."""
    control_path = tmp_path / "execution-control.json"
    worker_control = WorkerControl(
        client=object(), credentials={}, control_file=control_path, ttl_seconds=30,
    )
    training_control = FileSegmentControl(control_path)

    worker_control._interrupt("pause", {"command_id": "pause-1", "sequence": 7, "reason": "operator"})
    assert training_control.poll().command == "pause"
    assert training_control.poll().sequence == 7

    worker_control._interrupt("stop_now", {"command_id": "stop-2", "sequence": 8})
    assert training_control.poll().command == "stop"
    assert training_control.poll().sequence == 8


def test_only_explicit_v2_portability_allows_a_reverified_batch_on_another_worker(tmp_path):
    orch = Orchestrator(tmp_path / "control.sqlite", workspace_root=tmp_path)
    pool = orch.create_worker_pool(name="portable", allowed_actions=["train"])
    origin = orch.register_worker(
        pool_id=pool["pool_id"], name="origin", registration_token=pool["registration_token"],
        capabilities={"actions": ["train"], "worker_control_v1": True, "training_verification_v1": True, "execution_instance_id": "origin"},
    )
    candidate = orch.register_worker(
        pool_id=pool["pool_id"], name="candidate", registration_token=pool["registration_token"],
        capabilities={"actions": ["train"], "worker_control_v1": True, "training_verification_v1": True, "execution_instance_id": "candidate"},
    )
    queue_execution = {
        "mode": "verified", "batch_size": 4,
        "verified_worker_id": origin["worker_id"], "verified_environment_sha256": "f" * 64,
    }
    portable = WorkUnitV2(
        run_id=str(uuid.uuid4()), work_unit_id=str(uuid.uuid4()), sequence=0, phase="train",
        predecessor_work_unit_id=None,
        inputs={"dataset": ArtifactRef("dataset", "data", "r1", "a" * 64)},
        configuration=ArtifactRef("configuration", "config", "r1", "b" * 64), resources={},
        parameters={"queue_execution": {**queue_execution, "revalidate_fixed_batch": True}},
        start_cursor={"completed_epoch": 0}, stop_boundary={"completed_epoch": 1},
        output_contract={"kind": "training_checkpoint", "schema_version": 1},
        compatibility={"worker_control_v1": True},
    ).to_dict()
    legacy = WorkUnit(
        work_unit_id=str(uuid.uuid4()), attempt_id=str(uuid.uuid4()), specification_id=str(uuid.uuid4()),
        action="train", inputs={"dataset": ArtifactRef("dataset", "data")}, configuration=None,
        staging=ArtifactRef("staging", "stage"), resources={}, parameters={"queue_execution": queue_execution},
    ).to_dict()
    with orch._connection() as db:
        candidate_row = db.execute("SELECT * FROM registered_workers WHERE worker_id=?", (candidate["worker_id"],)).fetchone()
        assert orch._worker_can_execute_unit(candidate_row, portable)
        assert not orch._worker_can_execute_unit(candidate_row, legacy)

    portable["compatibility"] = {"worker_control_v1": True, "missing_capability": True}
    with orch._connection() as db:
        candidate_row = db.execute("SELECT * FROM registered_workers WHERE worker_id=?", (candidate["worker_id"],)).fetchone()
        assert not orch._worker_can_execute_unit(candidate_row, portable)


def test_completed_segment_commits_exactly_one_immutable_successor_and_duplicate_completion_is_safe(tmp_path):
    orch, _worker, unit, lease, credentials = _sequential_lease(tmp_path)
    artifact = _segment_artifact(tmp_path / "artifact", run_id=unit.run_id, completed_epoch=1, stop_epoch=1)
    first = _complete_segment(orch, credentials, artifact)
    again = orch.complete_worker_lease(**credentials, success=True)
    assert again == first

    run = orch.execution_run(unit.run_id)
    assert run["completed_epoch"] == 1
    assert run["status"] == "queued"
    assert run["checkpoint_ref"]["kind"] == "job_output"
    units = _units_for_run(orch, unit.run_id)
    assert len(units) == 2
    successor = units[-1]
    assert successor["status"] == "queued"
    assert successor["work_unit"]["sequence"] == 1
    assert successor["work_unit"]["predecessor_work_unit_id"] == lease["job_id"]
    assert successor["work_unit"]["parameters"]["execution_segment"] == {
        "phase": "train", "start_epoch": 1, "stop_epoch": 2, "total_epochs": 2,
    }


def test_pause_commits_checkpoint_then_requires_explicit_resume(tmp_path):
    orch, _worker, unit, lease, credentials = _sequential_lease(tmp_path)
    command = orch.request_job_command(lease["job_id"], action="pause", reason="operator pause")
    artifact = _segment_artifact(tmp_path / "artifact", run_id=unit.run_id, completed_epoch=1, stop_epoch=1, interrupted=True)
    _complete_segment(orch, credentials, artifact)

    run = orch.execution_run(unit.run_id)
    successor = _units_for_run(orch, unit.run_id)[-1]
    assert run["status"] == successor["status"] == "paused"
    assert orch.job_commands(lease["job_id"])[0]["command_id"] == command["command_id"]
    assert orch.job_commands(lease["job_id"])[0]["status"] == "applied"
    resumed = orch.request_job_command(successor["job_id"], action="resume", reason="continue")
    assert resumed["status"] == "applied"
    assert orch.execution_run(unit.run_id)["status"] == "queued"
    assert orch.job(successor["job_id"])["status"] == "queued"


def test_pending_restart_fences_segment_commit_before_any_successor_is_created(tmp_path):
    orch, _worker, unit, lease, credentials = _sequential_lease(tmp_path)
    orch.acknowledge_worker_lease(**credentials)
    orch.request_job_command(lease["job_id"], action="restart", reason="discard unit")
    artifact = _segment_artifact(tmp_path / "artifact", run_id=unit.run_id, completed_epoch=1, stop_epoch=1)
    orch.upload_worker_lease_output_archive(**credentials, archive=_zip_directory(artifact))
    with pytest.raises(ValueError, match="stopped or restarted"):
        orch.complete_worker_lease(**credentials, success=True)
    assert len(_units_for_run(orch, unit.run_id)) == 1


def test_forged_early_finalize_result_cannot_skip_remaining_training(tmp_path):
    orch, _worker, unit, _lease, credentials = _sequential_lease(tmp_path)
    artifact = _segment_artifact(
        tmp_path / "artifact", run_id=unit.run_id, completed_epoch=1, stop_epoch=1,
        next_phase="finalize",
    )
    orch.acknowledge_worker_lease(**credentials)
    orch.upload_worker_lease_output_archive(**credentials, archive=_zip_directory(artifact))
    with pytest.raises(ValueError, match="authoritative training progress"):
        orch.complete_worker_lease(**credentials, success=True)
    assert len(_units_for_run(orch, unit.run_id)) == 1


def test_reconciliation_commits_published_segment_without_retraining(tmp_path):
    orch, _worker, unit, lease, credentials = _sequential_lease(tmp_path)
    acknowledged = orch.acknowledge_worker_lease(**credentials)
    artifact = _segment_artifact(tmp_path / "artifact", run_id=unit.run_id, completed_epoch=1, stop_epoch=1)
    orch.upload_worker_lease_output_archive(**credentials, archive=_zip_directory(artifact))
    staging = orch.artifact_store.staging(lease["job_id"], acknowledged["output_attempt_id"])
    staged_path = orch.artifact_store.staging_path(staging)
    digest = directory_content_digest(staged_path)
    assert orch.artifact_store.validate(staging, validate_run_artifact).valid
    orch.artifact_store.seal(staging)
    ref = ArtifactRef(
        "job_output", lease["job_id"], revision=acknowledged["output_attempt_id"],
        fingerprint_sha256=digest,
    )
    orch.artifact_store.publish(staging, ref)

    assert orch.reconcile_worker_publications() == 1
    run = orch.execution_run(unit.run_id)
    assert run["completed_epoch"] == 1 and len(_units_for_run(orch, unit.run_id)) == 2
    assert orch.reconcile_worker_publications() == 0
