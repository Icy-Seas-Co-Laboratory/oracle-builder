from __future__ import annotations

import hmac
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from oracle_builder.api.registry import InferenceModelRegistry, ModelNotFoundError
from oracle_builder.api.compute import ComputeRequestError, ComputeService, JobAction
from oracle_builder.inference.transport import (
    InferenceTransportError,
    NPZ_MEDIA_TYPE,
    decode_inference_request,
    encode_inference_result_set,
)


class ComputeJobRequest(BaseModel):
    """An immutable execution specification created by the orchestrator."""

    job_id: str = Field(description="UUID issued by the orchestrator")
    action: JobAction
    parameters: dict[str, Any] = Field(default_factory=dict)
    resources: dict[str, Any] = Field(default_factory=dict)


def create_app(
    registry: InferenceModelRegistry,
    *,
    compute: ComputeService | None = None,
    auth_token: str | None = None,
    preload: bool = True,
    max_payload_bytes: int = 256 * 1024 * 1024,
    root_path: str = "",
) -> FastAPI:
    if max_payload_bytes < 1:
        raise ValueError("Inference payload limit must be positive")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if preload:
            registry.preload()
        try:
            yield
        finally:
            registry.close()
            if compute is not None:
                compute.close()

    normalized_root_path = "" if root_path.strip() in {"", "/"} else f"/{root_path.strip().strip('/')}"
    app = FastAPI(
        title="Oracle Builder Compute and Inference API",
        version="1.0.0",
        lifespan=lifespan,
        root_path=normalized_root_path,
    )
    app.state.registry = registry
    app.state.compute = compute

    def authorize(authorization: str | None) -> None:
        if auth_token is None:
            return
        expected = f"Bearer {auth_token}"
        if authorization is None or not hmac.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="Invalid Oracle Builder bearer token")

    def require_compute() -> ComputeService:
        if compute is None:
            raise HTTPException(status_code=503, detail="Compute execution is disabled on this oracle-serve instance")
        return compute

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready() -> dict[str, Any]:
        if not registry.healthy and compute is None:
            raise HTTPException(status_code=503, detail="No healthy inference models are registered")
        return {
            "status": "ready",
            "registered_models": registry.registered_count,
            "compute_enabled": compute is not None,
        }

    @app.get("/compute/workers")
    def compute_workers(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        authorize(authorization)
        return {"workers": require_compute().workers()}

    @app.get("/compute/status")
    def compute_status(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        authorize(authorization)
        return require_compute().status()

    @app.post("/compute/jobs", status_code=202)
    def submit_compute_job(
        body: ComputeJobRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(authorization)
        try:
            return require_compute().submit(
                job_id=body.job_id,
                action=body.action,
                parameters=body.parameters,
                resources=body.resources,
            )
        except ComputeRequestError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/compute/jobs/{job_id}")
    def compute_job(job_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        authorize(authorization)
        try:
            return require_compute().get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Compute job was not found") from exc

    @app.get("/compute/jobs/{job_id}/events")
    def compute_job_events(
        job_id: str,
        after: int = Query(default=0, ge=0),
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(authorization)
        try:
            return require_compute().events(job_id, after=after)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Compute job was not found") from exc

    @app.post("/compute/jobs/{job_id}/cancel")
    def cancel_compute_job(job_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        authorize(authorization)
        try:
            return require_compute().cancel(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Compute job was not found") from exc

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
            max_items = registry.max_items_for(selector)
            inference_request = decode_inference_request(
                await request.body(),
                max_payload_bytes=max_payload_bytes,
                max_items=max_items,
            )
        except ModelNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown inference model: {selector}") from exc
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
                    "execution": result_set.execution,
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
    registry = InferenceModelRegistry(
        serving_max_batch_size=int(os.environ.get("ORACLE_BUILDER_SERVING_MAX_BATCH_SIZE", 256)),
        serving_max_wait_ms=int(os.environ.get("ORACLE_BUILDER_SERVING_MAX_WAIT_MS", 8)),
        serving_queue_capacity=int(os.environ.get("ORACLE_BUILDER_SERVING_QUEUE_CAPACITY", 1024)),
    )
    roots = os.environ.get("ORACLE_BUILDER_MODELS_ROOT", "")
    for root in filter(None, (value.strip() for value in roots.split(os.pathsep))):
        registry.register_root(root)
    registrations = os.environ.get("ORACLE_BUILDER_MODELS", "")
    for entry in filter(None, (value.strip() for value in registrations.split(","))):
        if "=" not in entry:
            raise ValueError("ORACLE_BUILDER_MODELS entries must use alias=/path/to/run syntax")
        alias, path = entry.split("=", 1)
        registry.register(alias, path)
    return create_app(
        registry,
        compute=ComputeService(
            max_queue_size=int(os.environ.get("ORACLE_BUILDER_COMPUTE_QUEUE_SIZE", 128)),
            worker_id=os.environ.get("ORACLE_BUILDER_WORKER_ID", "local"),
        ) if os.environ.get("ORACLE_BUILDER_COMPUTE_ENABLED", "true").lower() not in {"0", "false", "no"} else None,
        auth_token=os.environ.get("ORACLE_BUILDER_API_TOKEN"),
        preload=os.environ.get("ORACLE_BUILDER_PRELOAD", "true").lower() not in {"0", "false", "no"},
        max_payload_bytes=int(os.environ.get("ORACLE_BUILDER_MAX_PAYLOAD_BYTES", 256 * 1024 * 1024)),
        root_path=os.environ.get("ORACLE_BUILDER_ROOT_PATH", ""),
    )


app = app_from_environment()
