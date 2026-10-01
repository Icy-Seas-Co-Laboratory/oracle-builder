"""Public, catalog-addressed inference gateway for the resident V2 runtime."""

from __future__ import annotations

import asyncio
import re
from typing import Any, Callable
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from oracle_builder.inference.transport_v2 import V2_NPZ_MEDIA_TYPE

from oracle_builder.orchestration.service import Orchestrator


MAX_FRAME_BYTES = 32 * 1024 * 1024
NPZ_MEDIA_TYPE_V2 = V2_NPZ_MEDIA_TYPE
_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")


def register_inference_v2_routes(
    app: FastAPI,
    orchestrator: Orchestrator,
    *,
    runtime_url: str | None,
    runtime_token: str | None,
    role_for_authorization: Callable[[str | None], str | None],
    role_tokens_configured: bool,
) -> None:
    """Expose one public boundary; runtime URLs and file paths stay private."""
    runtime_url = runtime_url.rstrip("/") if runtime_url else None
    if runtime_url:
        parsed_url = urlsplit(runtime_url)
        if parsed_url.scheme != "http" or parsed_url.hostname not in {"127.0.0.1", "localhost", "::1"} or parsed_url.path not in {"", "/"} or parsed_url.query or parsed_url.fragment or parsed_url.username or parsed_url.password:
            raise ValueError("Resident inference runtime URL must be a loopback HTTP origin")
    # Reuse loopback connections across frames; opening a TCP client for each
    # request would erode the benefit of a warm resident model.
    app.state.inference_http_client = httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=5.0), trust_env=False)

    def eligible(artifact_id: str, expected_fingerprint: str | None = None) -> dict[str, Any]:
        artifact = orchestrator.artifact(artifact_id)
        if artifact is None:
            raise HTTPException(404, "Model artifact was not found")
        if artifact.get("artifact_type") not in {"model_run", "model_product"} or artifact.get("lifecycle") != "sealed" or artifact.get("status") != "complete":
            raise HTTPException(409, "Inference requires a sealed, complete model artifact")
        if str(artifact.get("task") or "").strip().lower() not in {"classification", "embedding", "clustering", "segmentation"}:
            raise HTTPException(409, "Interactive V2 requires a supported image or mask model")
        fingerprint = str(artifact.get("fingerprint_sha256") or "").lower()
        if not _FINGERPRINT.fullmatch(fingerprint):
            raise HTTPException(409, "Model artifact has no valid sealed fingerprint")
        if expected_fingerprint and expected_fingerprint.lower() != fingerprint:
            raise HTTPException(409, "Model artifact fingerprint changed")
        return artifact

    def public_model(artifact: dict[str, Any]) -> dict[str, Any]:
        return {
            "artifact_id": artifact["artifact_id"],
            "fingerprint_sha256": artifact["fingerprint_sha256"],
            "name": artifact.get("name"),
            "task": artifact.get("task"),
            "status": artifact.get("status"),
        }

    def runtime_target(artifact_id: str, action: str) -> str:
        if not runtime_url or not runtime_token:
            raise HTTPException(503, "Resident inference runtime is not configured")
        return f"{runtime_url}/internal/v2/models/{artifact_id}:{action}"

    async def forward(artifact: dict[str, Any], action: str, payload: bytes | None = None, *, method: str = "POST") -> httpx.Response:
        target = runtime_target(str(artifact["artifact_id"]), action)
        headers = {
            "Authorization": f"Bearer {runtime_token}",
            "X-Artifact-Fingerprint": str(artifact["fingerprint_sha256"]),
        }
        if payload is not None:
            headers["Content-Type"] = NPZ_MEDIA_TYPE_V2
        try:
            return await app.state.inference_http_client.request(method, target, content=payload, headers=headers)
        except httpx.RequestError as exc:
            raise HTTPException(503, "Resident inference runtime is unavailable") from exc

    @app.get("/v2/inference/models")
    def list_models() -> dict[str, Any]:
        models = []
        for artifact in orchestrator.artifacts():
            try:
                eligible(str(artifact["artifact_id"]))
            except HTTPException:
                continue
            models.append(public_model(artifact))
        return {"models": models}

    @app.get("/v2/inference/models/{artifact_id}")
    async def model(request: Request, artifact_id: str) -> dict[str, Any]:
        # First detail access may verify/materialize a large artifact. Keep
        # this heavier read behind the same role boundary as inference.
        if role_tokens_configured and role_for_authorization(request.headers.get("authorization")) not in {"analyst", "operator", "admin"}:
            raise HTTPException(401, "An authorized API role token is required")
        artifact = eligible(artifact_id)
        reply = await forward(artifact, "describe", method="GET")
        if reply.status_code != 200:
            raise HTTPException(reply.status_code, "Model capabilities are unavailable")
        return {**public_model(artifact), "capabilities": reply.json()["capabilities"]}

    @app.post("/v2/inference/models/{artifact_id}:warm")
    async def warm_model(artifact_id: str, x_artifact_fingerprint: str | None = Header(default=None)) -> Response:
        artifact = eligible(artifact_id, x_artifact_fingerprint)
        reply = await forward(artifact, "warm")
        return Response(reply.content, status_code=reply.status_code, media_type=reply.headers.get("content-type", "application/json"))

    @app.post("/v2/inference/models/{artifact_id}:predict")
    async def predict_model(request: Request, artifact_id: str, x_artifact_fingerprint: str | None = Header(default=None)) -> Response:
        artifact = eligible(artifact_id, x_artifact_fingerprint)
        if request.headers.get("content-type", "").split(";", 1)[0].strip() != NPZ_MEDIA_TYPE_V2:
            raise HTTPException(415, f"Expected {NPZ_MEDIA_TYPE_V2}")
        try:
            content_length = int(request.headers.get("content-length", "0") or "0")
        except ValueError:
            raise HTTPException(400, "Content-Length must be an integer") from None
        if content_length > MAX_FRAME_BYTES:
            raise HTTPException(413, "Inference frame exceeds maximum size")
        chunks: list[bytes] = []
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_FRAME_BYTES:
                raise HTTPException(413, "Inference frame exceeds maximum size")
            chunks.append(chunk)
        payload = b"".join(chunks)
        reply = await forward(artifact, "predict", payload)
        return Response(reply.content, status_code=reply.status_code, media_type=reply.headers.get("content-type", NPZ_MEDIA_TYPE_V2))

    @app.websocket("/v2/inference/models/{artifact_id}/stream")
    async def stream_model(websocket: WebSocket, artifact_id: str) -> None:
        # HTTP middleware does not run on WebSockets; apply its role boundary here.
        if role_tokens_configured and role_for_authorization(websocket.headers.get("authorization")) not in {"analyst", "operator", "admin"}:
            await websocket.close(code=1008, reason="An authorized API role token is required")
            return
        try:
            artifact = eligible(artifact_id, websocket.headers.get("x-artifact-fingerprint"))
            runtime_target(artifact_id, "predict")
        except HTTPException as exc:
            await websocket.close(code=1008 if exc.status_code < 500 else 1013, reason=str(exc.detail))
            return
        await websocket.accept()
        slots = asyncio.Semaphore(8)
        send_lock = asyncio.Lock()
        tasks: set[asyncio.Task[None]] = set()
        frame_sequence = 0

        async def handle(payload: bytes, sequence: int) -> None:
            try:
                reply = await forward(artifact, "predict", payload)
                async with send_lock:
                    if reply.status_code == 200:
                        await websocket.send_bytes(reply.content)
                    else:
                        await websocket.send_json({"event": "error", "frame_sequence": sequence, "status": reply.status_code, "detail": reply.text[:1000]})
            except HTTPException as exc:
                async with send_lock:
                    await websocket.send_json({"event": "error", "frame_sequence": sequence, "status": exc.status_code, "detail": exc.detail})
            except Exception:
                # A disconnect can race a completed runtime request. Do not
                # leak internals or leave a detached task exception behind.
                pass
            finally:
                slots.release()

        try:
            while True:
                await slots.acquire()
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    slots.release()
                    break
                sequence = frame_sequence
                frame_sequence += 1
                payload = message.get("bytes")
                if payload is None or not payload or len(payload) > MAX_FRAME_BYTES:
                    slots.release()
                    async with send_lock:
                        await websocket.send_json({"event": "error", "frame_sequence": sequence, "status": 413 if payload else 415, "detail": "Expected a bounded binary V2 inference frame"})
                    continue
                task = asyncio.create_task(handle(payload, sequence))
                tasks.add(task)
                task.add_done_callback(tasks.discard)
        except WebSocketDisconnect:
            pass
        finally:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
