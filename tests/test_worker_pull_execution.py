from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile
import time
import uuid

import pytest

from oracle_builder.worker.pull import (
    DeterministicPackagingExecutor, ExecutorRegistry, OrchestratorPullClient,
    PullProtocolError, PullWorkerLoop, PullWorkerRuntime, WorkerCredentials,
    extract_tar_safely, load_worker_credentials, save_worker_credentials,
    training_progress_payload, archive_directory_to_file, upload_output_archive,
    recover_output_upload, OutputPublicationError,
)
from oracle_data_contracts.artifacts import ArtifactRef
from oracle_data_contracts.work_units import WorkUnit


def _tar(contents: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, value in contents.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(value)
            archive.addfile(entry, io.BytesIO(value))
    return buffer.getvalue()


class _Client:
    def __init__(self):
        self.calls: list[tuple[str, object]] = []
        self.output_parts: list[bytes] = []

    def acknowledge(self, **kwargs):
        self.calls.append(("acknowledge", kwargs))
        return {}

    def events(self, **kwargs):
        self.calls.append(("events", kwargs["events"]))
        return {}

    def artifact_grant(self, **kwargs):
        ref = kwargs["ref"]
        self.calls.append(("grant", ref))
        return {"grant": {"token": "artifact-token"}, "download_path": "/v1/delivery/grant-1"}

    def download_artifact(self, **kwargs):
        self.calls.append(("download", kwargs["download_path"]))
        return _tar({"artifact/artifact.txt": b"input"})

    def start_output_upload(self, **kwargs):
        self.calls.append(("upload-start", kwargs))
        return {"upload_id": "upload", "part_size": 64 * 1024, "uploaded_parts": []}

    def upload_output_part(self, **kwargs):
        self.calls.append(("upload-part", {key: value for key, value in kwargs.items() if key != "data"}))
        self.output_parts.append(kwargs["data"])
        return {}

    def finalize_output_upload(self, **kwargs):
        self.calls.append(("upload-finalize", kwargs))
        return {}

    def complete(self, **kwargs):
        self.calls.append(("complete", kwargs))
        return {}


class _Executor:
    def execute(self, *, work_unit, inputs, configuration, staging, emit_event):
        assert (inputs["dataset"] / "artifact.txt").read_bytes() == b"input"
        assert configuration is None
        assert not any(str(path).startswith("/control-plane") for path in inputs.values())
        (staging / "result.txt").write_text("done", encoding="utf-8")
        emit_event("progress", "halfway", {"fraction": 0.5})
        return {"records": 1}


def _lease() -> dict:
    unit = WorkUnit(
        work_unit_id="00000000-0000-4000-8000-000000000001",
        attempt_id="00000000-0000-4000-8000-000000000002",
        specification_id="00000000-0000-4000-8000-000000000003",
        action="train",
        inputs={"dataset": ArtifactRef("dataset", "source-1")},
        configuration=None,
        staging=ArtifactRef("staging", "out-1"),
        resources={},
    )
    return {
        "lease": {"lease_id": "lease-1", "lease_token": "lease-token"},
        "work_unit": unit.to_dict(),
    }


def test_runtime_acknowledges_materializes_executes_publishes_and_completes(tmp_path):
    client = _Client()
    runtime = PullWorkerRuntime(client, scratch_root=tmp_path / "scratch", executor=_Executor())

    result = runtime.run_lease(worker_id="worker-1", worker_token="worker-token", lease_response=_lease())

    assert result["status"] == "succeeded"
    assert result["result"] == {"records": 1}
    names = [name for name, _ in client.calls]
    assert names[0] == "acknowledge"
    assert names.index("grant") < names.index("download") < names.index("upload-start") < names.index("complete")
    timing_events = [event for name, value in client.calls if name == "events" for event in value if event["type"] == "stage_timing"]
    assert {event["data"]["stage"] for event in timing_events} == {
        "lease_acknowledgement", "artifact_materialization", "execution",
        "output_archiving", "output_publication", "worker_total",
    }
    assert all(event["data"]["schema"] == "oracle_timing_v1" and event["data"]["duration_seconds"] >= 0 for event in timing_events)
    uploaded = b"".join(client.output_parts)
    with tarfile.open(fileobj=io.BytesIO(uploaded), mode="r:") as archive:
        assert archive.extractfile("result.txt").read() == b"done"
    assert next(value for name, value in client.calls if name == "complete")["outcome"] == "succeeded"


def test_disk_archive_and_resumable_output_upload_skip_accepted_parts_and_retry(tmp_path):
    staging = tmp_path / "staging"; staging.mkdir()
    (staging / "result.bin").write_bytes(b"x" * 150_000)
    archive = tmp_path / "output.tar"
    size, digest = archive_directory_to_file(staging, archive)
    assert size == archive.stat().st_size and len(digest) == 64

    class Multipart:
        def __init__(self): self.parts = []; self.failed = False
        def start_output_upload(self, **kwargs):
            assert kwargs["archive_size"] == size
            return {"upload_id": "upload-1", "part_size": 64 * 1024, "uploaded_parts": [0]}
        def upload_output_part(self, **kwargs):
            if kwargs["part_number"] == 1 and not self.failed:
                self.failed = True
                raise RuntimeError("temporary")
            self.parts.append((kwargs["part_number"], len(kwargs["data"])))
            return {}
        def finalize_output_upload(self, **kwargs): return {"finalized": True}
    client = Multipart()
    assert upload_output_archive(client, archive_path=archive, lease_id="lease", worker_token="worker", lease_token="token", part_size=64 * 1024, sleep=lambda _: None) == {"finalized": True}
    assert [part for part, _ in client.parts] == [1, 2]
    assert all(length <= 64 * 1024 for _, length in client.parts)


def test_recovery_metadata_contains_no_tokens_and_resumes(tmp_path):
    root = tmp_path / "recovery"; root.mkdir()
    archive = root / "output.tar"; archive.write_bytes(b"x" * 70_000)
    metadata = {"version": 1, "lease_id": "lease", "archive": "output.tar", "archive_size": 70_000,
                "archive_sha256": "unused", "part_size": 64 * 1024}
    path = root / "recovery.json"; path.write_text(json.dumps(metadata)); path.chmod(0o600)
    class Client:
        def start_output_upload(self, **kwargs): return {"upload_id": "u", "part_size": 64 * 1024, "uploaded_parts": []}
        def upload_output_part(self, **kwargs): return {}
        def finalize_output_upload(self, **kwargs): return {"ok": True}
    assert recover_output_upload(Client(), recovery_dir=root, worker_token="fresh-worker-token", lease_token="fresh-lease-token", sleep=lambda _: None) == {"ok": True}
    assert not root.exists()


def test_runtime_retains_output_for_recovery_when_multipart_publication_fails(tmp_path):
    class FailingMultipart(_Client):
        def start_output_upload(self, **kwargs):
            raise RuntimeError("network unavailable")
    client = FailingMultipart()
    runtime = PullWorkerRuntime(client, scratch_root=tmp_path / "scratch", executor=_Executor())
    with pytest.raises(OutputPublicationError, match="could not be created"):
        runtime.run_lease(worker_id="worker-1", worker_token="worker-token", lease_response=_lease())
    retained = list((tmp_path / "scratch" / "recovery").iterdir())
    assert len(retained) == 1
    metadata = json.loads((retained[0] / "recovery.json").read_text())
    assert metadata["archive"] == "output.tar"
    assert "token" not in json.dumps(metadata).lower()
    assert (retained[0] / "output.tar").is_file()
    assert not any(name == "complete" for name, _ in client.calls)


def test_client_assigns_event_ids_for_idempotent_receipts(monkeypatch):
    captured = {}
    client = OrchestratorPullClient("https://orchestrator.example")
    monkeypatch.setattr(client, "_request", lambda method, path, body, **kwargs: captured.update(body) or {})
    client.events(lease_id="lease-1", worker_token="worker", lease_token="lease", events=[{
        "type": "progress", "message": "working", "data": {},
    }])
    uuid.UUID(captured["events"][0]["event_id"])


def test_client_uses_the_worker_cancellation_route(monkeypatch):
    captured = {}
    client = OrchestratorPullClient("https://orchestrator.example")

    def request(method, path, body, **kwargs):
        captured.update({"method": method, "path": path, "body": body, "worker_token": kwargs["worker_token"]})
        return {"cancel_requested": False}

    monkeypatch.setattr(client, "_request", request)
    assert client.cancellation(lease_id="lease-1", worker_token="worker", lease_token="lease") is False
    assert captured == {
        "method": "POST", "path": "/v1/worker-leases/lease-1/cancellation",
        "body": {"lease_token": "lease"}, "worker_token": "worker",
    }


def test_client_uses_a_separate_long_timeout_for_artifact_transfers(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"artifact"

    def urlopen(request, timeout):
        captured.update({"url": request.full_url, "timeout": timeout})
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    client = OrchestratorPullClient(
        "https://orchestrator.example", timeout_seconds=3, artifact_timeout_seconds=123,
    )
    assert client.download_artifact(
        worker_token="worker", lease_token="lease", artifact_token="grant",
        download_path="/v1/worker-leases/lease/artifact-grants/grant:download",
    ) == b"artifact"
    assert captured == {
        "url": "https://orchestrator.example/v1/worker-leases/lease/artifact-grants/grant:download",
        "timeout": 123,
    }


def test_training_progress_payload_is_path_free_and_bounded(tmp_path):
    status = tmp_path / "training-status.json"
    status.write_text(json.dumps({
        "updated_at": "2026-01-01T00:00:00Z", "phase": "Supervised training",
        "message": "Completed batch 200", "progress": {"epoch": 1, "total_epochs": 3, "batch": 200},
        "timing": {"elapsed_seconds": 20.0},
        "metrics": {
            "current_batch": {"loss": 0.4, "accuracy": 0.8, "not_a_number": float("nan")},
            "last_completed_epoch": {"loss": 0.5},
            "current_epoch_history": [{"batch": index, "loss": index / 100} for index in range(160)],
        },
        "history": [{"epoch": index, "loss": index / 100} for index in range(70)],
        "worker_path": "/must-not-leave-worker",
    }), encoding="utf-8")
    payload = training_progress_payload(status)
    assert payload is not None
    assert payload["progress"] == {"epoch": 1, "total_epochs": 3, "batch": 200}
    assert payload["metrics"] == {"loss": 0.4, "accuracy": 0.8}
    assert len(payload["trace"]) == 60
    assert len(payload["history"]) == 50
    assert "worker_path" not in payload


def test_runtime_records_failed_completion_when_executor_fails(tmp_path):
    class Failing:
        def execute(self, **kwargs):
            raise RuntimeError("not enough memory")

    client = _Client()
    runtime = PullWorkerRuntime(client, scratch_root=tmp_path / "scratch", executor=Failing())
    with pytest.raises(RuntimeError, match="not enough memory"):
        runtime.run_lease(worker_id="worker-1", worker_token="worker-token", lease_response=_lease())
    complete = [value for name, value in client.calls if name == "complete"]
    assert complete[-1]["outcome"] == "failed"
    assert "not enough memory" in complete[-1]["error"]


def test_runtime_renews_lease_while_executor_runs(tmp_path):
    class RenewingClient(_Client):
        def renew(self, **kwargs):
            self.calls.append(("renew", kwargs))
            return {}
    class Slow:
        def execute(self, *, staging, **kwargs):
            time.sleep(.06)
            (staging / "done").write_text("done")
            return {}
    client = RenewingClient()
    runtime = PullWorkerRuntime(client, scratch_root=tmp_path / "scratch", executor=Slow())
    runtime.run_lease(worker_id="worker", worker_token="token", lease_response=_lease(),
                      lease_ttl_seconds=2, renew_interval_seconds=.01)
    assert any(name == "renew" for name, _ in client.calls)


def test_archive_extraction_rejects_parent_paths_and_links(tmp_path):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        entry = tarfile.TarInfo("../escape")
        entry.size = 1
        archive.addfile(entry, io.BytesIO(b"x"))
    with pytest.raises(PullProtocolError, match="outside"):
        extract_tar_safely(buffer.getvalue(), tmp_path / "out")

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        entry = tarfile.TarInfo("link")
        entry.type = tarfile.SYMTYPE
        entry.linkname = "elsewhere"
        archive.addfile(entry)
    with pytest.raises(PullProtocolError, match="link or special"):
        extract_tar_safely(buffer.getvalue(), tmp_path / "out2")


def test_fixed_registry_and_packaging_executor_publish_contract_output(tmp_path):
    registry = ExecutorRegistry({"package": DeterministicPackagingExecutor()})
    staging = tmp_path / "staging"
    staging.mkdir()
    result = registry.execute(work_unit={"work_unit_id": "u", "attempt_id": "a", "action": "package"},
                              inputs={}, configuration=None, staging=staging,
                              emit_event=lambda *_: None)
    assert result == {"output": "work-unit-manifest.json"}
    assert '"action":"package"' in (staging / "work-unit-manifest.json").read_text()
    with pytest.raises(PullProtocolError, match="no registered executor"):
        registry.execute(work_unit={"action": "train"}, inputs={}, configuration=None,
                         staging=staging, emit_event=lambda *_: None)


def test_private_worker_credentials_round_trip_and_reject_public_file(tmp_path):
    path = tmp_path / "credentials.json"
    credentials = WorkerCredentials("worker", "secret", "pool", "https://orch")
    save_worker_credentials(path, credentials)
    assert path.stat().st_mode & 0o777 == 0o600
    assert load_worker_credentials(path) == credentials
    path.chmod(0o644)
    with pytest.raises(PullProtocolError, match="must not be group/world"):
        load_worker_credentials(path)


def test_operational_loop_registers_once_reuses_credentials_and_stops_at_max_jobs(tmp_path):
    class LoopClient:
        base_url = "https://orch"
        def __init__(self): self.registered = 0; self.polled = 0
        def register(self, **kwargs):
            self.registered += 1
            return {"worker": {"worker_id": "worker"}, "worker_token": "secret"}
        def poll(self, **kwargs):
            self.polled += 1
            return {"lease": {"lease_id": "lease", "lease_token": "token"}, "work_unit": {}}
    class Runtime:
        def __init__(self): self.calls = 0
        def run_lease(self, **kwargs): self.calls += 1; return {"status": "succeeded"}
    client, runtime = LoopClient(), Runtime()
    loop = PullWorkerLoop(client, runtime=runtime, pool_id="pool", name="name",
                          credentials_path=tmp_path / "cred", registration_token="join",
                          capabilities={}, lease_ttl_seconds=2, poll_interval_seconds=.001)
    assert loop.run(max_jobs=2) == 2
    assert (client.registered, runtime.calls) == (1, 2)
    # A restart uses its private server-issued token rather than attempting to
    # consume the one-time pool join token again.
    second = PullWorkerLoop(client, runtime=runtime, pool_id="pool", name="name",
                            credentials_path=tmp_path / "cred", registration_token=None,
                            capabilities={}, lease_ttl_seconds=2, poll_interval_seconds=.001)
    assert second.run(once=True) == 1
    assert client.registered == 1
