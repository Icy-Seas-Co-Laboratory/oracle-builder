from __future__ import annotations

import hmac
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, Request, Response
from starlette.concurrency import run_in_threadpool

from oracle_builder.api.registry import InferenceModelRegistry, ModelNotFoundError
from oracle_builder.inference.transport import (
    InferenceTransportError,
    NPZ_MEDIA_TYPE,
    decode_inference_request,
    encode_inference_result_set,
)


def create_app(
    registry: InferenceModelRegistry,
    *,
    auth_token: str | None = None,
    preload: bool = True,
    max_payload_bytes: int = 256 * 1024 * 1024,
    max_items: int = 128,
) -> FastAPI:
    if max_payload_bytes < 1 or max_items < 1:
        raise ValueError("Inference payload and item limits must be positive")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if preload:
            registry.preload()
        yield

    app = FastAPI(title="Oracle Builder Inference API", version="1.0.0", lifespan=lifespan)
    app.state.registry = registry

    def authorize(authorization: str | None) -> None:
        if auth_token is None:
            return
        expected = f"Bearer {auth_token}"
        if authorization is None or not hmac.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="Invalid Oracle Builder bearer token")

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready() -> dict[str, Any]:
        if not registry.healthy:
            raise HTTPException(status_code=503, detail="No healthy inference models are registered")
        return {"status": "ready", "registered_models": registry.registered_count}

    @app.get("/v1/models")
    def models(
        task: str | None = Query(default=None),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(authorization)
        return {"models": registry.describe(task=task)}

    @app.get("/v1/models/{selector}/evidence/{item_id}")
    def evidence_item(
        selector: str,
        item_id: str,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(authorization)
        try:
            return registry.evidence_item(selector, item_id)
        except ModelNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Evidence exemplar was not found") from exc

    @app.post("/v1/models/{selector}:predict")
    async def predict(
        selector: str,
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> Response:
        authorize(authorization)
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type != NPZ_MEDIA_TYPE:
            raise HTTPException(status_code=415, detail=f"Expected Content-Type {NPZ_MEDIA_TYPE}")
        try:
            inference_request = decode_inference_request(
                await request.body(),
                max_payload_bytes=max_payload_bytes,
                max_items=max_items,
            )
        except (InferenceTransportError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            result_set = await run_in_threadpool(
                registry.predict,
                selector,
                inference_request.items,
            )
            result_set.parameters.update(
                {
                    "transport_request_id": inference_request.request_id,
                    "model_selector": selector,
                }
            )
            return Response(
                content=encode_inference_result_set(result_set),
                media_type=NPZ_MEDIA_TYPE,
                headers={"X-Oracle-Request-ID": inference_request.request_id},
            )
        except ModelNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown inference model: {selector}") from exc

    return app


def app_from_environment() -> FastAPI:
    registry = InferenceModelRegistry()
    registrations = os.environ.get("ORACLE_BUILDER_MODELS", "")
    for entry in filter(None, (value.strip() for value in registrations.split(","))):
        if "=" not in entry:
            raise ValueError("ORACLE_BUILDER_MODELS entries must use alias=/path/to/run syntax")
        alias, path = entry.split("=", 1)
        registry.register(alias, path)
    return create_app(
        registry,
        auth_token=os.environ.get("ORACLE_BUILDER_API_TOKEN"),
        preload=os.environ.get("ORACLE_BUILDER_PRELOAD", "true").lower() not in {"0", "false", "no"},
        max_payload_bytes=int(os.environ.get("ORACLE_BUILDER_MAX_PAYLOAD_BYTES", 256 * 1024 * 1024)),
        max_items=int(os.environ.get("ORACLE_BUILDER_MAX_ITEMS", 128)),
    )


app = app_from_environment()
