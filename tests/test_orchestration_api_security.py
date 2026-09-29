from __future__ import annotations

from hashlib import sha256

from fastapi.testclient import TestClient

from oracle_builder.orchestration.api import create_app, role_token_digests
from oracle_builder.orchestration.service import Orchestrator


def _orchestrator(tmp_path):
    return Orchestrator(tmp_path / "orchestrator.sqlite", workspace_root=tmp_path)


def _app(tmp_path, **kwargs):
    return create_app(_orchestrator(tmp_path), **kwargs)


def test_hashed_role_tokens_gate_mutations_but_not_anonymous_reads(tmp_path):
    token = "operator-secret"
    digest = sha256(token.encode()).hexdigest()
    with TestClient(_app(tmp_path, role_tokens={"operator": f"sha256:{digest}"})) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/v1/worker-pools").status_code == 200
        assert client.post("/v1/worker-pools", json={"name": "pool", "allowed_actions": ["train"]}).status_code == 401
        assert client.post("/v1/worker-pools", headers={"Authorization": "Bearer wrong"}, json={"name": "pool", "allowed_actions": ["train"]}).status_code == 401
        created = client.post("/v1/worker-pools", headers={"Authorization": f"Bearer {token}"}, json={"name": "pool", "allowed_actions": ["train"]})
    assert created.status_code == 201


def test_analyst_is_allowed_only_for_analytical_post_requests(tmp_path):
    analyst = "analyst-secret"
    digest = sha256(analyst.encode()).hexdigest()
    with TestClient(_app(tmp_path, role_tokens={"analyst": digest})) as client:
        # Query is authorized before its empty-catalog validation path.
        response = client.post("/v1/artifacts/catalog/query", headers={"Authorization": f"Bearer {analyst}"}, json={})
        assert response.status_code != 401
        assert client.post("/v1/worker-pools", headers={"Authorization": f"Bearer {analyst}"}, json={"name": "pool", "allowed_actions": ["train"]}).status_code == 401


def test_anonymous_reads_are_rate_limited_per_source(tmp_path):
    token = "operator-secret"
    with TestClient(_app(
        tmp_path, role_tokens={"operator": sha256(token.encode()).hexdigest()},
        anonymous_get_limit=2, anonymous_get_window_seconds=60,
    )) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/live").status_code == 200
        blocked = client.get("/health/live")
    assert blocked.status_code == 429
    assert blocked.headers["retry-after"]


def test_explicitly_unauthenticated_local_app_does_not_rate_limit_its_gui_reads(tmp_path):
    with TestClient(_app(tmp_path, anonymous_get_limit=2, anonymous_get_window_seconds=60)) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/live").status_code == 200


def test_role_token_configuration_rejects_plaintext_and_unknown_roles():
    try:
        role_token_digests({"operator": "plaintext"})
    except ValueError as exc:
        assert "SHA-256" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("plaintext token configuration was accepted")
    try:
        role_token_digests({"worker": "0" * 64})
    except ValueError as exc:
        assert "unsupported" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unknown role configuration was accepted")


def test_successful_mutation_is_audited_without_request_body(tmp_path):
    token = "operator-secret"
    orchestrator = _orchestrator(tmp_path)
    with TestClient(create_app(orchestrator, role_tokens={"operator": sha256(token.encode()).hexdigest()})) as client:
        response = client.post("/v1/worker-pools", headers={"Authorization": f"Bearer {token}", "X-Request-Id": "request-1"}, json={"name": "pool", "allowed_actions": ["train"]})
    assert response.status_code == 201
    with orchestrator._connection() as db:
        event = db.execute("SELECT actor_role,method,path,outcome,request_id FROM audit_events").fetchone()
    assert tuple(event) == ("operator", "POST", "/v1/worker-pools", "succeeded", "request-1")
