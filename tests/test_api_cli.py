from __future__ import annotations

import sys

import pytest

from oracle_builder.api import cli
from oracle_builder.api.app import app_from_environment


def _capture_serve_start(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    started: list[object] = []

    def run(app, **_kwargs):
        started.append(app)

    monkeypatch.setattr(cli.uvicorn, "run", run)
    return started


def test_compute_capacity_cli_arguments_are_passed_to_compute_service(monkeypatch):
    started = _capture_serve_start(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        ["oracle-serve", "--compute-worker-slots", "3", "--compute-cpu-capacity", "7"],
    )

    cli.main()

    compute = started[0].state.compute
    try:
        assert compute.status()["resources"] == {
            "cpu_capacity": 7,
            "cpu_in_use": 0,
            "gpu_leases": [],
            "worker_slots": 3,
        }
    finally:
        compute.close()


def test_compute_capacity_environment_prefers_launcher_names_and_keeps_legacy(monkeypatch):
    started = _capture_serve_start(monkeypatch)
    monkeypatch.setenv("ORACLE_COMPUTE_WORKER_SLOTS", "2")
    monkeypatch.setenv("ORACLE_COMPUTE_CPU_CAPACITY", "6")
    monkeypatch.setenv("ORACLE_BUILDER_COMPUTE_WORKER_SLOTS", "4")
    monkeypatch.setenv("ORACLE_BUILDER_COMPUTE_CPU_CAPACITY", "8")
    monkeypatch.setattr(sys, "argv", ["oracle-serve"])

    cli.main()

    compute = started[0].state.compute
    try:
        resources = compute.status()["resources"]
        assert resources["worker_slots"] == 2
        assert resources["cpu_capacity"] == 6
    finally:
        compute.close()


def test_asgi_environment_uses_the_same_preferred_capacity_names(monkeypatch):
    monkeypatch.setenv("ORACLE_COMPUTE_WORKER_SLOTS", "2")
    monkeypatch.setenv("ORACLE_COMPUTE_CPU_CAPACITY", "6")
    monkeypatch.setenv("ORACLE_BUILDER_COMPUTE_WORKER_SLOTS", "4")
    monkeypatch.setenv("ORACLE_BUILDER_COMPUTE_CPU_CAPACITY", "8")
    app = app_from_environment()
    compute = app.state.compute
    try:
        resources = compute.status()["resources"]
        assert resources["worker_slots"] == 2
        assert resources["cpu_capacity"] == 6
    finally:
        compute.close()


@pytest.mark.parametrize("arguments", [
    ["--compute-worker-slots", "0"],
    ["--compute-cpu-capacity", "-1"],
])
def test_compute_capacity_cli_rejects_non_positive_values(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["oracle-serve", *arguments])
    with pytest.raises(SystemExit) as exc_info:
        cli.main()
    assert exc_info.value.code == 2
