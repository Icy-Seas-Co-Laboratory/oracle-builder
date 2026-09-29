from __future__ import annotations

import pytest

from oracle_builder.worker import cli
from oracle_builder.worker.pull import WorkerCredentials, save_worker_credentials


def test_pull_worker_requires_control_plane_identity_and_fixed_executor():
    args = cli.build_parser().parse_args([
        "--orchestrator-url", "http://control", "--pool-id", "pool",
        "--credentials-file", "identity.json", "--scratch-root", "scratch", "--executor", "train",
    ])
    assert args.pool_id == "pool"
    assert args.executor == ["train"]


@pytest.mark.parametrize("arguments", [["--role", "compute"], ["run-pull"], ["--host", "127.0.0.1"], ["--model", "x=y"]])
def test_legacy_http_worker_options_are_not_a_public_surface(arguments):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(arguments)


def test_recover_output_mode_uses_existing_worker_credentials_and_completes(tmp_path, monkeypatch, capsys):
    credentials = tmp_path / "credentials.json"
    save_worker_credentials(credentials, WorkerCredentials("worker", "worker-secret", "pool", "https://orch"))
    recovery = tmp_path / "recovery"; recovery.mkdir()
    (recovery / "recovery.json").write_text('{"lease_id":"lease-1","archive":"output.tar","part_size":65536}')
    calls = []
    class Client:
        def __init__(self, url): calls.append(("client", url))
        def complete(self, **kwargs): calls.append(("complete", kwargs)); return {}
    def recover(client, **kwargs):
        calls.append(("recover", kwargs)); return {}
    monkeypatch.setattr(cli, "OrchestratorPullClient", Client)
    monkeypatch.setattr(cli, "recover_output_upload", recover)
    cli.main(["--orchestrator-url", "https://orch", "--credentials-file", str(credentials),
              "--recover-output", str(recovery), "--recover-lease-token", "lease-secret"])
    assert calls[1][0] == "recover"
    assert calls[1][1]["worker_token"] == "worker-secret"
    assert calls[2] == ("complete", {"lease_id": "lease-1", "worker_token": "worker-secret", "lease_token": "lease-secret", "outcome": "succeeded"})
    assert not recovery.exists()
    assert "recovered_output" in capsys.readouterr().out
