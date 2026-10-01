"""Private HTTP sidecar for the resident V2 inference runtime.

The orchestration API is the public boundary.  This app is deliberately an
internal, loopback-only component: it receives an opaque catalog id and its
already-pinned fingerprint, resolves that identity through the local catalog,
and serves bounded binary V2 requests from the resident model cache.
"""

from __future__ import annotations

import argparse
import hmac
import logging
import os
import re
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response

from oracle_builder.inference.resident import (
    ResidentArtifactChangedError,
    ResidentInferenceRuntime,
    ResidentModelNotFoundError,
    ResidentRuntimeCapacityError,
    ResidentRuntimeError,
)
from oracle_builder.inference.image_contract_v2 import capabilities_for_artifact, request_task_for_model
from oracle_builder.inference.transport_v2 import (
    V2_NPZ_MEDIA_TYPE,
    V2InferenceOutputLimitError,
    V2InferenceContractError,
    V2InferenceTransportError,
    decode_v2_inference_request,
    encode_v2_inference_result_set,
    project_v2_result_set,
)
from oracle_builder.orchestration.service import Orchestrator


_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")
_LOG = logging.getLogger(__name__)
DEFAULT_MAX_REQUEST_BYTES = 32 * 1024 * 1024


def _detail(status_code: int, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail=message)


class _ArtifactAdmission:
    """Materialize catalog-approved models once for the resident process."""

    def __init__(
        self,
        orchestrator: Orchestrator,
        runtime: ResidentInferenceRuntime,
        materialized_root: Path,
    ) -> None:
        self.orchestrator = orchestrator
        self.runtime = runtime
        self.materialized_root = materialized_root.resolve()
        self.materialized_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._admitted: dict[str, Any] = {}
        self._capabilities: dict[str, dict[str, Any]] = {}

    def admit(self, artifact_id: str, expected_fingerprint: str):
        artifact = self.orchestrator.artifact(artifact_id)
        if artifact is None:
            raise _detail(404, "Model artifact was not found")
        if artifact.get("lifecycle") != "sealed" or artifact.get("status") != "complete":
            raise _detail(409, "Inference requires a sealed, complete model artifact")
        if artifact.get("artifact_type") not in {"model_run", "model_product"}:
            raise _detail(422, "Inference requires a cataloged model artifact")
        catalog_fingerprint = str(artifact.get("fingerprint_sha256") or "").lower()
        if not _FINGERPRINT.fullmatch(catalog_fingerprint):
            raise _detail(409, "Model artifact has no valid immutable fingerprint")
        if not _FINGERPRINT.fullmatch(expected_fingerprint) or not hmac.compare_digest(
            catalog_fingerprint, expected_fingerprint
        ):
            raise _detail(409, "Model artifact fingerprint does not match the catalog")

        # Materialization copies the artifact with a content-integrity check.
        # The runtime then verifies the run's own seal before loading it.
        with self._lock:
            model = self._admitted.get(catalog_fingerprint)
            if model is None:
                target = self.materialized_root / catalog_fingerprint
                try:
                    if target.exists():
                        if target.is_symlink() or not target.is_dir():
                            raise ValueError("resident artifact cache contains an invalid entry")
                    else:
                        ref = self.orchestrator._model_artifact_ref_for_inference(artifact)
                        self.orchestrator.artifact_store.materialize(ref, target)
                    model = self.runtime.register(
                        target, artifact_fingerprint=catalog_fingerprint
                    )
                except HTTPException:
                    raise
                except (OSError, ValueError, ResidentArtifactChangedError) as exc:
                    _LOG.exception("Resident model integrity admission failed")
                    raise _detail(409, "Model artifact failed integrity admission") from exc
                self._admitted[catalog_fingerprint] = model
        if model.artifact_id != artifact_id:
            raise _detail(409, "Materialized model identity does not match the catalog artifact")
        if not hmac.compare_digest(model.artifact_fingerprint, catalog_fingerprint):
            raise _detail(409, "Materialized model fingerprint does not match the catalog")
        return model

    def capabilities(self, model: Any) -> dict[str, Any]:
        with self._lock:
            cached = self._capabilities.get(model.artifact_fingerprint)
            if cached is None:
                cached = capabilities_for_artifact(model.run_dir, model.task)
                self._capabilities[model.artifact_fingerprint] = cached
            return cached


def create_inference_runtime_app(
    orchestrator: Orchestrator,
    runtime: ResidentInferenceRuntime,
    internal_token: str,
    *,
    materialized_root: str | Path | None = None,
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    max_items: int = 64,
) -> FastAPI:
    """Create the loopback-only serving app used by the public API proxy."""
    if not isinstance(internal_token, str) or not internal_token:
        raise ValueError("internal_token must be a non-empty bearer token")
    if not isinstance(max_request_bytes, int) or max_request_bytes < 1:
        raise ValueError("max_request_bytes must be a positive integer")
    if not isinstance(max_items, int) or max_items < 1:
        raise ValueError("max_items must be a positive integer")
    cache_root = Path(materialized_root or (orchestrator.database.parent / "inference-runtime-artifacts"))
    admission = _ArtifactAdmission(orchestrator, runtime, cache_root)
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        # Let in-flight lane work finish before the process exits and release
        # cached model resources for embedders that construct this app in tests.
        close = getattr(runtime, "close", None)
        if callable(close):
            close()

    app = FastAPI(
        title="Oracle Builder resident inference runtime", docs_url=None,
        redoc_url=None, lifespan=lifespan,
    )
    app.state.inference_runtime = runtime
    app.state.inference_admission = admission

    def authorize(authorization: str | None) -> None:
        scheme, _, presented = (authorization or "").partition(" ")
        if scheme.lower() != "bearer" or not presented or not hmac.compare_digest(presented, internal_token):
            raise _detail(401, "Invalid internal inference runtime token")

    async def body(request: Request) -> bytes:
        raw_length = request.headers.get("content-length")
        if raw_length:
            try:
                if int(raw_length) > max_request_bytes:
                    raise _detail(413, "Inference request exceeds the configured byte limit")
            except ValueError:
                raise _detail(400, "Content-Length must be an integer") from None
        chunks: list[bytes] = []
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > max_request_bytes:
                raise _detail(413, "Inference request exceeds the configured byte limit")
            chunks.append(chunk)
        return b"".join(chunks)

    def admitted_model(artifact_id: str, fingerprint: str):
        return admission.admit(artifact_id, fingerprint.lower())

    def admitted_capabilities(artifact_id: str, fingerprint: str):
        model = admitted_model(artifact_id, fingerprint)
        try:
            return model, admission.capabilities(model)
        except (OSError, ValueError) as exc:
            _LOG.exception("Resident model capability inspection failed")
            raise _detail(409, "Model serving contract is unavailable") from exc

    @app.get("/internal/v2/health")
    def health() -> dict[str, str]:
        # This endpoint intentionally reveals no model or configuration data,
        # so the local launcher can use it as a loopback readiness probe.
        return {"status": "ok"}

    @app.post("/internal/v2/models/{artifact_id}:warm")
    async def warm(
        artifact_id: str,
        authorization: str | None = Header(default=None),
        x_artifact_fingerprint: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(authorization)
        if not x_artifact_fingerprint:
            raise _detail(400, "X-Artifact-Fingerprint is required")
        model, capabilities = await run_in_threadpool(admitted_capabilities, artifact_id, x_artifact_fingerprint)
        try:
            result = await run_in_threadpool(runtime.warm, model.artifact_fingerprint)
        except ResidentRuntimeCapacityError as exc:
            raise _detail(503, str(exc)) from exc
        except (ResidentRuntimeError, ValueError) as exc:
            _LOG.exception("Resident model warmup failed")
            raise _detail(409, "Model warmup failed") from exc
        return {"model": model.to_dict(), "capabilities": capabilities, "runtime": result}

    @app.get("/internal/v2/models/{artifact_id}:describe")
    async def describe(
        artifact_id: str,
        authorization: str | None = Header(default=None),
        x_artifact_fingerprint: str | None = Header(default=None),
    ) -> dict[str, Any]:
        authorize(authorization)
        if not x_artifact_fingerprint:
            raise _detail(400, "X-Artifact-Fingerprint is required")
        model, capabilities = await run_in_threadpool(admitted_capabilities, artifact_id, x_artifact_fingerprint)
        return {"model": model.to_dict(), "capabilities": capabilities}

    @app.post("/internal/v2/models/{artifact_id}:predict")
    async def predict(
        artifact_id: str,
        request: Request,
        authorization: str | None = Header(default=None),
        x_artifact_fingerprint: str | None = Header(default=None),
    ) -> Response:
        authorize(authorization)
        if not x_artifact_fingerprint:
            raise _detail(400, "X-Artifact-Fingerprint is required")
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type != V2_NPZ_MEDIA_TYPE:
            raise _detail(415, f"Content-Type must be {V2_NPZ_MEDIA_TYPE}")
        payload = await body(request)
        try:
            parsed = await run_in_threadpool(
                decode_v2_inference_request, payload,
                max_payload_bytes=max_request_bytes, max_items=max_items,
            )
        except V2InferenceTransportError as exc:
            raise _detail(422, str(exc)) from exc
        model, capabilities = await run_in_threadpool(admitted_capabilities, artifact_id, x_artifact_fingerprint)
        model_task = str(model.task).strip().lower()
        expected_task = request_task_for_model(model_task)
        if parsed.task != expected_task:
            raise _detail(422, f"Request task {parsed.task!r} is incompatible with model task {model.task!r}")
        supported = frozenset(capabilities["possible_outputs"])
        selected = parsed.response_options.outputs
        if selected is not None and selected - supported:
            raise _detail(422, f"Requested outputs are unsupported by this artifact: {', '.join(sorted(selected - supported))}")
        try:
            result_set = await run_in_threadpool(runtime.predict, model.artifact_fingerprint, parsed.items)
            projected = await run_in_threadpool(
                project_v2_result_set, result_set, parsed.response_options, parsed.task,
                supported_outputs=supported,
                default_outputs=frozenset(capabilities["default_outputs"]),
            )
            encoded = await run_in_threadpool(encode_v2_inference_result_set, projected)
        except V2InferenceOutputLimitError as exc:
            raise _detail(413, str(exc)) from exc
        except V2InferenceContractError as exc:
            _LOG.exception("Resident model violated its V2 output contract")
            raise _detail(500, "Model output violated its serving contract") from exc
        except V2InferenceTransportError as exc:
            raise _detail(422, str(exc)) from exc
        except ResidentRuntimeCapacityError as exc:
            raise _detail(503, str(exc)) from exc
        except (ResidentModelNotFoundError, ResidentRuntimeError, ValueError) as exc:
            _LOG.exception("Resident model inference failed")
            raise _detail(409, "Model inference failed") from exc
        if len(encoded) > max_request_bytes:
            raise _detail(413, "Inference response exceeds the configured byte limit")
        return Response(content=encoded, media_type=V2_NPZ_MEDIA_TYPE)

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, error: HTTPException) -> JSONResponse:
        return JSONResponse({"detail": error.detail}, status_code=error.status_code)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the private Oracle Builder resident inference runtime.")
    parser.add_argument("--database", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--artifact-root")
    parser.add_argument("--materialized-root")
    parser.add_argument("--internal-token", default=os.environ.get("ORACLE_INFERENCE_RUNTIME_TOKEN"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8111)
    parser.add_argument("--max-request-mib", type=int, default=32)
    parser.add_argument("--max-items", type=int, default=64)
    parser.add_argument("--max-loaded-models", type=int, default=2)
    parser.add_argument("--classification-max-batch-size", type=int, default=64)
    parser.add_argument("--classification-max-wait-ms", type=int, default=8)
    parser.add_argument("--queue-capacity", type=int, default=1024)
    args = parser.parse_args()
    if args.host != "127.0.0.1":
        parser.error("the resident inference runtime must bind to 127.0.0.1")
    if not args.internal_token:
        parser.error("set --internal-token or ORACLE_INFERENCE_RUNTIME_TOKEN")
    if args.max_request_mib < 1:
        parser.error("--max-request-mib must be positive")
    orchestrator = Orchestrator(
        args.database,
        workspace_root=args.workspace_root,
        artifact_root=args.artifact_root,
    )
    runtime = ResidentInferenceRuntime(
        max_loaded_models=args.max_loaded_models,
        classification_max_batch_size=args.classification_max_batch_size,
        classification_max_wait_ms=args.classification_max_wait_ms,
        queue_capacity=args.queue_capacity,
    )
    app = create_inference_runtime_app(
        orchestrator,
        runtime,
        args.internal_token,
        materialized_root=args.materialized_root,
        max_request_bytes=args.max_request_mib * 1024 * 1024,
        max_items=args.max_items,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    main()
