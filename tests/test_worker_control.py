"""Focused contract checks for the worker-side control parent.

These tests deliberately use the dependency-light control/supervisor modules;
they do not import TensorFlow or start an Oracle API server.
"""
from __future__ import annotations

import subprocess
import sys
import time

import pytest

from oracle_data_contracts.artifacts import ArtifactRef
from oracle_data_contracts.work_units import WorkUnit
from oracle_builder.orchestration.service import Orchestrator
from oracle_builder.worker.control import ExecutionInterrupted, WorkerControl
from oracle_builder.worker.supervisor import terminate_process_group


class _TransientlyOfflineClient:
    def heartbeat(self, **_kwargs):
        raise TimeoutError("network temporarily unavailable")

    def commands(self, **_kwargs):
        raise TimeoutError("network temporarily unavailable")


def _control(tmp_path, client, *, ttl_seconds=1):
    return WorkerControl(
        client,
        credentials={"lease_id": "lease", "worker_token": "worker", "lease_token": "lease-token"},
        control_file=tmp_path / "control.json",
        ttl_seconds=ttl_seconds,
        interval=0.02,
    )


def test_watchdog_revokes_execution_after_renewal_deadline_when_transport_is_offline(tmp_path):
    control = _control(tmp_path, _TransientlyOfflineClient(), ttl_seconds=1)
    control.start()
    try:
        deadline = time.monotonic() + 1.5
        while not control.halt.is_set() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert control.halt.is_set()
        with pytest.raises(ExecutionInterrupted, match="lease_lost"):
            control.check()
    finally:
        control.stop()


def test_control_file_is_atomic_and_urgent_commands_supersede_safe_boundary_commands(tmp_path):
    control = _control(tmp_path, _TransientlyOfflineClient())
    pause = {"command_id": "pause-1", "action": "pause"}
    control._interrupt("pause", pause)
    assert not control.halt.is_set()
    assert '"command": "pause"' in (tmp_path / "control.json").read_text(encoding="utf-8")

    stop = {"command_id": "stop-1", "action": "stop_now"}
    control._interrupt("stop_now", stop)
    control._interrupt("pause", pause)  # delayed duplicate cannot undo stop
    assert control.halt.is_set()
    with pytest.raises(ExecutionInterrupted) as interrupted:
        control.check()
    assert interrupted.value.action == "stop_now"
    assert interrupted.value.command == stop


def test_heartbeat_cancellation_does_not_reuse_an_older_pause_receipt(tmp_path):
    """A stop request may race a previously accepted boundary directive."""
    class CancellingClient:
        calls = 0

        def heartbeat(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return {"cancel_requested": True}
            control.done.set()
            return {"cancel_requested": False}

    control = _control(tmp_path, CancellingClient())
    control.command = {"command_id": "old-pause", "action": "pause", "sequence": 1}
    control._heartbeat()

    assert control.action == "stop_now"
    assert control.command is None


def test_process_group_termination_stops_a_supervised_child_promptly():
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    started = time.monotonic()
    try:
        terminate_process_group(process, grace_seconds=0.2)
        assert process.poll() is not None
        assert time.monotonic() - started < 2
    finally:
        terminate_process_group(process, grace_seconds=0.1)


def _leased_control_job(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    orchestrator = Orchestrator(tmp_path / "control.sqlite", workspace_root=workspace)
    pool = orchestrator.create_worker_pool(name="control", allowed_actions=["train"])
    worker = orchestrator.register_worker(
        pool_id=pool["pool_id"], name="control-worker", registration_token=pool["registration_token"],
        capabilities={"actions": ["train"], "worker_control_v1": True, "execution_instance_id": "boot-1"},
    )
    unit = WorkUnit(
        work_unit_id="00000000-0000-4000-8000-000000000001",
        attempt_id="00000000-0000-4000-8000-000000000002",
        specification_id="00000000-0000-4000-8000-000000000003",
        action="train", inputs={"dataset": ArtifactRef("dataset", "control-dataset")},
        configuration=None, staging=ArtifactRef("staging", "control-stage"), resources={},
    )
    orchestrator.enqueue_work_unit_for_pool(pool_id=pool["pool_id"], work_unit=unit.to_dict())
    lease = orchestrator.acquire_next_worker_lease(
        worker_id=worker["worker_id"], worker_token=worker["worker_token"], ttl_seconds=30,
    )
    assert lease is not None
    return orchestrator, worker, unit, lease


def test_durable_command_is_idempotent_and_generation_scoped(tmp_path):
    orchestrator, worker, unit, lease = _leased_control_job(tmp_path)
    command_id = "00000000-0000-4000-8000-000000000004"
    first = orchestrator.request_job_command(unit.work_unit_id, action="restart", reason="retry unit", command_id=command_id)
    duplicate = orchestrator.request_job_command(unit.work_unit_id, action="restart", reason="retry unit", command_id=command_id)
    assert duplicate == first
    assert first["generation"] == 1

    polled = orchestrator.poll_worker_commands(
        lease_id=lease["lease_id"], worker_id=worker["worker_id"], worker_token=worker["worker_token"],
        lease_token=lease["lease_token"], wait_seconds=0,
    )
    assert [command["command_id"] for command in polled["commands"]] == [command_id]
    accepted = orchestrator.acknowledge_worker_command(
        lease_id=lease["lease_id"], worker_id=worker["worker_id"], worker_token=worker["worker_token"],
        lease_token=lease["lease_token"], command_id=command_id, status="accepted",
    )
    # At-least-once command delivery must return the original durable receipt.
    assert orchestrator.acknowledge_worker_command(
        lease_id=lease["lease_id"], worker_id=worker["worker_id"], worker_token=worker["worker_token"],
        lease_token=lease["lease_token"], command_id=command_id, status="accepted",
    ) == accepted
    applied = orchestrator.acknowledge_worker_command(
        lease_id=lease["lease_id"], worker_id=worker["worker_id"], worker_token=worker["worker_token"],
        lease_token=lease["lease_token"], command_id=command_id, status="applied", result={"executor_stopped": True},
    )
    assert applied["status"] == "applied"
    assert orchestrator.job(unit.work_unit_id)["status"] == "queued"
    # Applying restart releases the lease. A worker that loses its response
    # must still be able to replay the terminal acknowledgement and obtain
    # the original receipt without reviving the stale lease.
    assert orchestrator.acknowledge_worker_command(
        lease_id=lease["lease_id"], worker_id=worker["worker_id"], worker_token=worker["worker_token"],
        lease_token=lease["lease_token"], command_id=command_id, status="applied", result={"executor_stopped": True},
    ) == applied


def test_heartbeat_rejects_a_stale_attempt_generation(tmp_path):
    orchestrator, worker, _unit, lease = _leased_control_job(tmp_path)
    with pytest.raises(ValueError, match="stale attempt generation"):
        orchestrator.heartbeat_worker_lease(
            lease_id=lease["lease_id"], worker_id=worker["worker_id"], worker_token=worker["worker_token"],
            lease_token=lease["lease_token"], ttl_seconds=30,
            telemetry={"generation": 99, "worker_boot_id": "boot-1"},
        )
    with pytest.raises(ValueError, match="different worker boot"):
        orchestrator.heartbeat_worker_lease(
            lease_id=lease["lease_id"], worker_id=worker["worker_id"], worker_token=worker["worker_token"],
            lease_token=lease["lease_token"], ttl_seconds=30,
            telemetry={"generation": 1, "worker_boot_id": "other-boot"},
        )
