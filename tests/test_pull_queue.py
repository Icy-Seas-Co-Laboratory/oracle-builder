from __future__ import annotations

import io
import hashlib
import json
import sqlite3
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from oracle_builder.data.sqlite_dataset import create_synthetic_classification
from oracle_builder.orchestration.api import create_app
from oracle_builder.orchestration.service import Orchestrator
from oracle_data_contracts.artifacts import ArtifactRef


@pytest.fixture
def queue_setup(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    dataset_path = workspace / "frozen.sqlite"
    create_synthetic_classification(dataset_path, n=9, shape=(16, 16, 1), classes=3)
    with sqlite3.connect(dataset_path) as db:
        db.execute("UPDATE dataset SET lifecycle='frozen'")
    orch = Orchestrator(tmp_path / "control.sqlite", workspace_root=workspace, artifact_root=tmp_path / "artifacts")
    dataset = orch.ingest_dataset(dataset_path)
    definition = orch.create_model_definition(name="portable-resnet", template_id="resnet18")
    pool = orch.create_worker_pool(name="train-pool", allowed_actions=["train"])
    worker = orch.register_worker(pool_id=pool["pool_id"], name="Verifier", registration_token=pool["registration_token"],
                                  capabilities={"actions": ["train"], "training_verification_v1": True, "work_unit_v2": True, "training_segments_v1": True, "worker_control_v1": True})

    def queue(**kwargs):
        return orch.queue_model_definition_for_pool(
            definition["definition_id"], name="portable run", dataset_id=dataset["dataset_id"],
            worker_pool_id=pool["pool_id"], resources={"gpu_count": 0}, epochs=1, **kwargs)
    return orch, pool, worker, queue


def finish_verification(orch, worker, *, batch_size=4, success=True, report_overrides=None):
    lease = orch.acquire_next_worker_lease(worker_id=worker["worker_id"], worker_token=worker["worker_token"])
    assert lease and lease["work_unit"]["parameters"]["queue_execution"]["phase"] == "verify"
    credentials = {key: lease[key] for key in ("lease_id", "lease_token")}
    credentials.update(worker_id=worker["worker_id"], worker_token=worker["worker_token"])
    orch.acknowledge_worker_lease(**credentials)
    if success:
        execution = lease["work_unit"]["parameters"]["queue_execution"]
        report = {"ready": True, "batch_size": batch_size, "work_unit_id": lease["job_id"],
                  "checks": ["configuration_resolution", "dataset_loading", "model_forward_backward"]}
        if execution.get("required_execution_contract_version") == 2:
            report.update({"execution_contract_version": 2, "unit_epochs": 1, "unit_steps": 0})
        if report_overrides:
            report.update(report_overrides)
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as bundle:
            if execution.get("required_execution_contract_version") == 2:
                import tempfile
                from pathlib import Path
                from oracle_builder.config import resolve_config
                from oracle_data_contracts.artifacts.splits import create_split_manifest
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    unit = lease['work_unit']
                    config_path = orch.artifact_store.resolve(ArtifactRef.from_dict(unit['configuration'])) / 'payload'
                    dataset_path = orch.artifact_store.resolve(ArtifactRef.from_dict(unit['inputs']['input'])) / 'payload'
                    config = resolve_config(config_path, dataset_path, root)
                    manifest = create_split_manifest(root, dataset_path, config)
                    report['split_manifest'] = {key:value for key,value in manifest.items() if key != 'assignments'}
                    bundle.writestr('protocol/splits.json', (root/'protocol/splits.json').read_bytes())
            bundle.writestr("verification.json", json.dumps(report))
        orch.upload_worker_lease_output_archive(**credentials, archive=output.getvalue())
    orch.complete_worker_lease(**credentials, success=success, message="Passed" if success else "Model preflight failed")
    return lease


def test_v2_queue_rejects_legacy_verification_report(queue_setup):
    orch, _pool, worker, queue = queue_setup
    queued = queue(batch_size=4)
    orch.verify_queued_runs_for_pool(queued_run_ids=[queued["queued_run_id"]])
    with pytest.raises(ValueError, match="required execution contract V2"):
        finish_verification(orch, worker, report_overrides={"execution_contract_version": 1})
    assert orch.queued_run(queued["queued_run_id"])["status"] == "verifying"


def test_queue_verify_then_explicit_start(queue_setup):
    orch, pool, worker, queue = queue_setup
    queued = queue(batch_size=4)
    assert queued["status"] == "pending_verification"
    assert not queued["start_authorized"]
    assert orch.jobs() == []
    spec = orch.specification(queued["specification_id"])
    assert str(orch.workspace_root) not in str(spec["parameters"])
    assert orch.artifact_store.resolve(ArtifactRef.from_dict(spec["parameters"]["artifact_inputs"]["input"])).joinpath("payload").is_file()
    with pytest.raises(ValueError, match="verified"):
        orch.authorize_queued_runs_for_pool(queued_run_ids=[queued["queued_run_id"]], worker_pool_id=pool["pool_id"])
    with pytest.raises(ValueError, match="Verify"):
        orch.enqueue_specification_for_pool(queued["specification_id"], pool["pool_id"])
    orch.verify_queued_runs_for_pool(queued_run_ids=[queued["queued_run_id"]])
    assert orch.queued_run(queued["queued_run_id"])["status"] == "verifying"
    finish_verification(orch, worker)
    verified = orch.queued_run(queued["queued_run_id"])
    assert verified["status"] == "ready" and verified["preflight_status"] == "valid"
    assert verified["batch_size"] == 4
    assert not verified["start_authorized"]
    assert len(orch.jobs()) == 1
    assert orch.acquire_next_worker_lease(worker_id=worker["worker_id"], worker_token=worker["worker_token"]) is None
    started = orch.authorize_queued_runs_for_pool(queued_run_ids=[queued["queued_run_id"]], worker_pool_id=pool["pool_id"])
    assert len(started["dispatched"]) == 1
    unit = started["dispatched"][0]["work_unit"]
    assert unit["parameters"]["queue_execution"]["mode"] == "verified"
    assert unit["parameters"]["queue_execution"]["batch_size"] == 4
    assert str(orch.workspace_root) not in str(unit)


def test_legacy_verified_training_publishes_and_indexes_real_manifest(queue_setup, tmp_path):
    import uuid
    from oracle_builder.artifacts import create_run_artifact, update_run_artifact, seal_run_artifact

    orch, pool, worker, queue = queue_setup
    queued = queue(batch_size=4)
    orch.verify_queued_runs_for_pool(queued_run_ids=[queued["queued_run_id"]])
    # Retained legacy work units do not require the V2 segment contract.  A
    # migration fixture is used here because new intake always emits V2.
    with orch._connection() as db:
        row = db.execute("SELECT job_id,work_unit_json FROM jobs WHERE queued_run_id=?", (queued["queued_run_id"],)).fetchone()
        unit = json.loads(row["work_unit_json"])
        unit["parameters"]["queue_execution"].pop("required_execution_contract_version", None)
        payload = json.dumps(unit, sort_keys=True, separators=(",", ":"))
        db.execute("UPDATE jobs SET work_unit_json=?,work_unit_sha256=? WHERE job_id=?",
                   (payload, hashlib.sha256(payload.encode()).hexdigest(), row["job_id"]))
    finish_verification(orch, worker)
    orch.authorize_queued_runs_for_pool(queued_run_ids=[queued["queued_run_id"]], worker_pool_id=pool["pool_id"])
    lease = orch.acquire_next_worker_lease(worker_id=worker["worker_id"], worker_token=worker["worker_token"])
    credentials = {key: lease[key] for key in ("lease_id", "lease_token")}
    credentials.update(worker_id=worker["worker_id"], worker_token=worker["worker_token"])
    orch.acknowledge_worker_lease(**credentials)
    run = tmp_path / "real-manifest"
    source = tmp_path / "source.toml"
    source.write_text('[run]\ntask = "classification"\nmodel = "simple_cnn"\n')
    create_run_artifact(run, run_id=str(uuid.uuid4()), name="publication fixture", source_config=source, config={
        "run": {"task": "classification", "model": "simple_cnn"},
        "data": {"input_shape": [16, 16, 1], "num_classes": 3},
        "dataset": {"dataset_id": str(uuid.uuid4()), "lifecycle": "frozen"},
    })
    # A real interrupted artifact exercises the publication contract without
    # requiring TensorFlow or fabricated model bytes in this regression.
    update_run_artifact(run, status="interrupted")
    seal_run_artifact(run)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as bundle:
        for path in run.rglob("*"):
            if path.is_file():
                bundle.write(path, path.relative_to(run))
    orch.upload_worker_lease_output_archive(**credentials, archive=output.getvalue())
    completion = orch.complete_worker_lease(**credentials, success=True)
    assert completion["output_ref"]["kind"] == "model_run"
    assert orch.queued_run(queued["queued_run_id"])["status"] == "indexed"


def test_auto_batch_is_resolved_before_start_and_pinned_to_verifying_worker(queue_setup):
    orch, pool, worker, queue = queue_setup
    queued = queue(batch_size_mode="auto", maximum_batch_size=64)
    orch.verify_queued_runs_for_pool(queued_run_ids=[queued["queued_run_id"]], start_after_verification=True)
    finish_verification(orch, worker, batch_size=24)
    assert orch.queued_run(queued["queued_run_id"])["batch_size"] == 24
    assert orch.queued_run(queued["queued_run_id"])["status"] == "queued"
    assert len(orch.jobs()) == 2
    other = orch.register_worker(pool_id=pool["pool_id"], name="other", registration_token=pool["registration_token"], capabilities={"actions": ["train"], "training_verification_v1": True})
    assert orch.acquire_next_worker_lease(worker_id=other["worker_id"], worker_token=other["worker_token"]) is None
    lease = orch.acquire_next_worker_lease(worker_id=worker["worker_id"], worker_token=worker["worker_token"])
    assert lease["work_unit"]["parameters"]["queue_execution"]["batch_size"] == 24
    assert lease["work_unit"]["parameters"]["queue_execution"].get("phase") != "verify"


def test_failed_verification_never_autostarts_and_can_be_retried(queue_setup):
    orch, pool, worker, queue = queue_setup
    queued = queue(batch_size=4)
    ids = [queued["queued_run_id"]]
    orch.verify_queued_runs_for_pool(queued_run_ids=ids, start_after_verification=True)
    finish_verification(orch, worker, success=False)
    failed = orch.queued_run(ids[0])
    assert failed["status"] == "needs_attention" and not failed["start_authorized"]
    assert len(orch.jobs()) == 1
    orch.verify_queued_runs_for_pool(queued_run_ids=ids)
    finish_verification(orch, worker)
    assert orch.queued_run(ids[0])["status"] == "ready"
    assert len(orch.jobs()) == 2


def test_verification_batch_is_atomic_and_old_workers_cannot_train_it(queue_setup):
    orch, pool, worker, queue = queue_setup
    queued = queue(batch_size=4)
    with pytest.raises(KeyError):
        orch.verify_queued_runs_for_pool(queued_run_ids=[queued["queued_run_id"], "missing"])
    assert not orch.jobs()
    assert orch.queued_run(queued["queued_run_id"])["status"] == "pending_verification"
    orch.verify_queued_runs_for_pool(queued_run_ids=[queued["queued_run_id"]])
    old = orch.register_worker(pool_id=pool["pool_id"], name="old", registration_token=pool["registration_token"], capabilities={"actions": ["train"]})
    assert orch.acquire_next_worker_lease(worker_id=old["worker_id"], worker_token=old["worker_token"]) is None
    with pytest.raises(ValueError, match="unstarted"):
        orch.verify_queued_runs_for_pool(queued_run_ids=[queued["queued_run_id"]])


def test_start_selection_is_atomic_and_concurrent_start_is_single_job(queue_setup):
    orch, pool, worker, queue = queue_setup
    queued, unverified = queue(batch_size=4), queue(batch_size=4)
    orch.verify_queued_runs_for_pool(queued_run_ids=[queued["queued_run_id"]])
    finish_verification(orch, worker)
    with pytest.raises(ValueError, match="verified"):
        orch.authorize_queued_runs_for_pool(queued_run_ids=[queued["queued_run_id"], unverified["queued_run_id"]], worker_pool_id=pool["pool_id"])
    assert len(orch.jobs()) == 1
    assert not orch.queued_run(queued["queued_run_id"])["start_authorized"]
    def start():
        try:
            orch.authorize_queued_runs_for_pool(queued_run_ids=[queued["queued_run_id"]], worker_pool_id=pool["pool_id"])
            return True
        except ValueError:
            return False
    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(lambda _: start(), range(2))) == [False, True]
    assert len(orch.jobs()) == 2


def test_verification_consent_survives_restart(queue_setup):
    orch, pool, worker, queue = queue_setup
    queued = queue(batch_size=4)
    orch.verify_queued_runs_for_pool(queued_run_ids=[queued["queued_run_id"]], start_after_verification=True)
    restarted = Orchestrator(orch.database, workspace_root=orch.workspace_root, artifact_root=orch.artifact_root)
    finish_verification(restarted, worker)
    assert restarted.queued_run(queued["queued_run_id"])["status"] == "queued"
    assert len(restarted.jobs()) == 2


def test_queue_verify_api_defaults_to_no_training(queue_setup):
    orch, pool, worker, queue = queue_setup
    queued = queue(batch_size=4)
    with TestClient(create_app(orch)) as client:
        result = client.post('/v1/queued-runs:verify', json={"queued_run_ids": [queued["queued_run_id"]]})
        assert result.status_code == 202, result.text
        assert not orch.queued_run(queued["queued_run_id"])["start_authorized"]
        result = client.post('/v1/queued-runs:start', json={"queued_run_ids": [queued["queued_run_id"]], "worker_pool_id": pool["pool_id"]})
        assert result.status_code == 422


def test_restarted_worker_updates_verification_capability_on_poll(queue_setup):
    orch, pool, worker, queue = queue_setup
    queued = queue(batch_size=4)
    orch.verify_queued_runs_for_pool(queued_run_ids=[queued['queued_run_id']])
    old = orch.register_worker(pool_id=pool['pool_id'], name='upgraded', registration_token=pool['registration_token'], capabilities={'actions': ['train']})
    with TestClient(create_app(orch)) as client:
        response = client.post(f"/v1/workers/{old['worker_id']}:lease", headers={'Authorization': f"Bearer {old['worker_token']}"}, json={'capabilities': {'actions': ['train'], 'training_verification_v1': True, 'work_unit_v2': True, 'training_segments_v1': True, 'worker_control_v1': True}})
        assert response.status_code == 200, response.text
        assert response.json()['work_unit']['parameters']['queue_execution']['phase'] == 'verify'
    assert orch.registered_worker(old['worker_id'])['capabilities']['training_verification_v1'] is True


def test_published_verification_recovers_after_completion_transaction_crash(queue_setup, monkeypatch):
    orch, pool, worker, queue = queue_setup
    queued = queue(batch_size=4)
    orch.verify_queued_runs_for_pool(queued_run_ids=[queued['queued_run_id']], start_after_verification=True)
    original = orch._record_verification_in_connection
    def crash(*args, **kwargs):
        raise RuntimeError('simulated database transaction crash')
    monkeypatch.setattr(orch, '_record_verification_in_connection', crash)
    with pytest.raises(RuntimeError, match='simulated'):
        finish_verification(orch, worker)
    assert orch.queued_run(queued['queued_run_id'])['status'] == 'verifying'
    monkeypatch.setattr(orch, '_record_verification_in_connection', original)
    assert orch.reconcile_worker_publications() == 1
    assert orch.queued_run(queued['queued_run_id'])['status'] == 'queued'
    assert len(orch.jobs()) == 2
    assert orch.reconcile_worker_publications() == 0
    assert len(orch.jobs()) == 2


def test_cancelled_verification_cannot_autostart(queue_setup):
    orch, pool, worker, queue = queue_setup
    queued = queue(batch_size=4)
    result = orch.verify_queued_runs_for_pool(queued_run_ids=[queued['queued_run_id']], start_after_verification=True)
    job = result['dispatched'][0]
    orch.request_pull_job_cancellation(job['job_id'])
    assert orch.queued_run(queued['queued_run_id'])['status'] == 'cancelled'
    assert not orch.queued_run(queued['queued_run_id'])['start_authorized']
    assert orch.acquire_next_worker_lease(worker_id=worker['worker_id'], worker_token=worker['worker_token']) is None


def test_verification_success_is_retained_if_pool_disabled_before_autostart(queue_setup):
    orch, pool, worker, queue = queue_setup
    queued = queue(batch_size=4)
    orch.verify_queued_runs_for_pool(queued_run_ids=[queued['queued_run_id']], start_after_verification=True)
    original = orch._record_verification_in_connection
    def disable_then_record(db, *args):
        db.execute('UPDATE worker_pools SET enabled=0 WHERE pool_id=?', (pool['pool_id'],))
        return original(db, *args)
    orch._record_verification_in_connection = disable_then_record
    finish_verification(orch, worker)
    verified = orch.queued_run(queued['queued_run_id'])
    assert verified['status'] == 'ready'
    assert verified['preflight_status'] == 'valid'
    assert not verified['start_authorized']
    assert len(orch.jobs()) == 1


@pytest.mark.parametrize('start', [False, True])
def test_worker_environment_change_invalidates_verified_batch(queue_setup, start):
    orch, pool, worker, queue = queue_setup
    queued = queue(batch_size=4)
    orch.verify_queued_runs_for_pool(queued_run_ids=[queued['queued_run_id']], start_after_verification=start)
    finish_verification(orch, worker)
    changed = {'actions': ['train'], 'training_verification_v1': True, 'work_unit_v2': True, 'training_segments_v1': True, 'worker_control_v1': True, 'execution_instance_id': 'new-worker-boot'}
    leased = orch.acquire_next_worker_lease(worker_id=worker['worker_id'], worker_token=worker['worker_token'], capabilities=changed)
    if start:
        # A V2 run verifies the pinned batch at the segment boundary, so an
        # authorized continuation remains schedulable after a worker restart.
        assert leased is not None
        assert orch.queued_run(queued['queued_run_id'])['status'] == 'leased'
        return
    assert leased is None
    held = orch.queued_run(queued['queued_run_id'])
    assert held['status'] == 'needs_attention'
    assert not held['start_authorized']
    assert 'restarted' in held['failure_reason']
    # An explicit new verification restores the ability to start on this boot.
    orch.verify_queued_runs_for_pool(queued_run_ids=[queued['queued_run_id']])
    finish_verification(orch, worker)
    assert orch.queued_run(queued['queued_run_id'])['status'] == 'ready'
