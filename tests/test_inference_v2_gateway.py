"""Public V2 gateway keeps artifact selection and authorization at the control plane."""

from hashlib import sha256

import httpx
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from oracle_builder.orchestration.api import create_app


FINGERPRINT = "a" * 64
ARTIFACT = {
    "artifact_id": "model-1", "artifact_type": "model_run", "lifecycle": "sealed",
    "status": "complete", "fingerprint_sha256": FINGERPRINT,
    "name": "Ready model", "task": "classification", "path": "/private/never-expose",
}


class _Catalog:
    def artifact(self, artifact_id):
        return ARTIFACT if artifact_id == "model-1" else None

    def artifacts(self):
        return [ARTIFACT, {**ARTIFACT, "artifact_id": "draft", "lifecycle": "draft"}]

    def reconcile_startup(self):
        return {}

    def reconcile_worker_deployments(self):
        return {}

    def start_operation_runner(self):
        pass

    def stop_operation_runner(self):
        pass


def _app(calls):
    token = "analyst-token"
    app = create_app(
        _Catalog(), role_tokens={"analyst": sha256(token.encode()).hexdigest()},
        inference_runtime_url="http://127.0.0.1:8111", inference_runtime_token="internal-secret",
    )

    def reply(request):
        calls.append(request)
        assert request.headers["authorization"] == "Bearer internal-secret"
        assert request.headers["x-artifact-fingerprint"] == FINGERPRINT
        if request.url.path.endswith(":warm"):
            return httpx.Response(200, json={"state": "ready"})
        if request.url.path.endswith(":describe"):
            assert request.method == "GET"
            return httpx.Response(200, json={"capabilities": {
                "model_task": "classification", "request_task": "classification",
                "required_inputs": ["image"], "input_shape": [4, 4, 1],
                "default_outputs": ["class_probabilities", "primary_decision"],
                "possible_outputs": ["class_probabilities", "primary_decision"],
            }})
        return httpx.Response(200, content=b"result-frame", headers={"content-type": "application/vnd.oracle-builder.inference.v2+npz"})

    app.state.inference_http_client = httpx.AsyncClient(transport=httpx.MockTransport(reply))
    return app, token


def test_gateway_catalog_fingerprint_and_analyst_proxy():
    calls = []
    app, token = _app(calls)
    with TestClient(app) as client:
        models = client.get("/v2/inference/models").json()["models"]
        assert [row["artifact_id"] for row in models] == ["model-1"]
        assert "path" not in models[0]
        assert client.get("/v2/inference/models/model-1").status_code == 401
        detail = client.get("/v2/inference/models/model-1", headers={"Authorization": f"Bearer {token}"}).json()
        assert detail["capabilities"]["required_inputs"] == ["image"]
        assert "path" not in detail
        assert client.post("/v2/inference/models/model-1:warm").status_code == 401
        headers = {"Authorization": f"Bearer {token}", "X-Artifact-Fingerprint": "b" * 64}
        assert client.post("/v2/inference/models/model-1:warm", headers=headers).status_code == 409
        assert len(calls) == 1
        headers["X-Artifact-Fingerprint"] = FINGERPRINT
        assert client.post("/v2/inference/models/model-1:warm", headers=headers).status_code == 200
        headers["Content-Type"] = "application/vnd.oracle-builder.inference.v2+npz"
        result = client.post("/v2/inference/models/model-1:predict", headers=headers, content=b"request-frame")
        assert result.status_code == 200
        assert result.content == b"result-frame"
        assert len(calls) == 3


def test_gateway_websocket_auth_and_multiple_binary_frames():
    calls = []
    app, token = _app(calls)
    with TestClient(app) as client:
        try:
            with client.websocket_connect("/v2/inference/models/model-1/stream"):
                raise AssertionError("unauthorized websocket accepted")
        except WebSocketDisconnect as exc:
            assert exc.code == 1008
        with client.websocket_connect(
            "/v2/inference/models/model-1/stream",
            headers={"Authorization": f"Bearer {token}", "X-Artifact-Fingerprint": FINGERPRINT},
        ) as websocket:
            websocket.send_bytes(b"first")
            assert websocket.receive_bytes() == b"result-frame"
            websocket.send_bytes(b"second")
            assert websocket.receive_bytes() == b"result-frame"
    assert len(calls) == 2


def test_gateway_lists_embedding_and_clustering_artifacts():
    calls = []
    app, _ = _app(calls)
    app.state.orchestrator.artifacts = lambda: [
        {**ARTIFACT, "artifact_id": "classifier", "task": "classification"},
        {**ARTIFACT, "artifact_id": "encoder", "task": "embedding"},
        {**ARTIFACT, "artifact_id": "clusters", "task": "clustering"},
        {**ARTIFACT, "artifact_id": "refiner", "task": "segmentation"},
    ]
    app.state.orchestrator.artifact = lambda artifact_id: next(
        (row for row in app.state.orchestrator.artifacts() if row["artifact_id"] == artifact_id), None
    )
    with TestClient(app) as client:
        models = client.get("/v2/inference/models").json()["models"]
    assert {row["task"] for row in models} == {"classification", "embedding", "clustering", "segmentation"}
