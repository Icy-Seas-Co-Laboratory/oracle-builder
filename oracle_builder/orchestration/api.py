from __future__ import annotations

from contextlib import asynccontextmanager
from collections import defaultdict, deque
from hashlib import sha256
import hmac
import re
from typing import Any, Mapping
import uuid
import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect

from oracle_data_contracts.artifacts import ArtifactRef
from oracle_builder.orchestration.service import Orchestrator


class DatasetIngestRequest(BaseModel):
    path: str


class UploadSessionRequest(BaseModel):
    """Metadata for a browser-resumable file transfer; bytes use part routes."""
    model_config = ConfigDict(extra="forbid")
    kind: str
    filename: str = Field(min_length=1, max_length=255)
    size_bytes: int = Field(ge=1)


class ScanRequest(BaseModel):
    root: str


class WorkerPoolEnqueueRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    worker_pool_id: str


class RecipeRequest(BaseModel):
    name: str
    config_path: str
    description: str = ""


class PlannedRunUpdateRequest(BaseModel):
    name: str | None = None
    resources: dict[str, Any] = Field(default_factory=dict)
    config_overrides: dict[str, Any] = Field(default_factory=dict)


class ModelPreviewRequest(BaseModel):
    architecture: str
    dataset_id: str
    overrides: dict[str, Any] = Field(default_factory=dict)


class ComparisonRequest(BaseModel):
    name: str
    artifact_ids: list[str]
    description: str = ""


class ComparisonGroupMemberRequest(BaseModel):
    artifact_id: str
    relationship_label: str | None = None
    note: str = ""


class ComparisonGroupRequest(BaseModel):
    name: str
    members: list[ComparisonGroupMemberRequest]
    description: str = ""
    relationship_label: str = "related"
    baseline_artifact_id: str | None = None


class ArtifactTagRequest(BaseModel):
    tags: list[str] = Field(default_factory=list)


class ArtifactCatalogQueryRequest(BaseModel):
    filters: dict[str, Any] = Field(default_factory=dict)
    sort: str = "updated_at"
    order: str = "desc"
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=100, ge=1, le=500)


class ArtifactCatalogReindexRequest(BaseModel):
    """Optional registered artifact subset; omitted means all catalog rows."""
    artifact_ids: list[str] | None = None


class ArtifactReplicaOperationRequest(BaseModel):
    """Operator-only replica action over stable ArtifactRef values."""
    model_config = ConfigDict(extra="forbid")
    refs: list[dict[str, Any]] = Field(min_length=1, max_length=100)


class ArtifactTagAssignmentRequest(BaseModel):
    artifact_ids: list[str]
    tags: list[str] = Field(default_factory=list)


class TagRequest(BaseModel):
    name: str
    color: str | None = None


class ModelDraftRequest(BaseModel):
    name: str = "Untitled model"
    description: str = ""
    config: dict[str, Any] = Field(default_factory=dict)
    layout: dict[str, Any] = Field(default_factory=dict)
    source_artifact_id: str | None = None


class ModelDraftUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    config: dict[str, Any] | None = None
    layout: dict[str, Any] | None = None


class ModelDraftCloneRequest(BaseModel):
    name: str | None = None


class ModelDefinitionRequest(BaseModel):
    name: str
    template_id: str
    description: str = ""
    config: dict[str, Any] = Field(default_factory=dict)


class ModelDefinitionUpdateRequest(BaseModel):
    expected_revision: int = Field(ge=1)
    name: str | None = None
    description: str | None = None
    config: dict[str, Any] | None = None


class ModelDefinitionDuplicateRequest(BaseModel):
    revision: int | None = Field(default=None, ge=1)
    name: str | None = None
    lineage_kind: str = "duplicate"


class ValidatedQueueRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    dataset_id: str
    worker_pool_id: str
    revision: int | None = Field(default=None, ge=1)
    description: str = ""
    resources: dict[str, Any] = Field(default_factory=dict)
    initialization: dict[str, Any] = Field(default_factory=dict)
    batch_size_mode: str = "manual"
    batch_size: int | None = Field(default=None, ge=1)
    maximum_batch_size: int = Field(default=256, ge=1)
    epochs: int = Field(default=10, ge=1, description="Run-specific training epoch budget sealed with this queued run.")


class InferenceRunRequest(BaseModel):
    """Typed, path-free request for durable batch inference."""
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=240)
    model_artifact_id: str
    dataset_id: str
    worker_pool_id: str
    split: str = "all"
    prediction_set: str | None = Field(default=None, max_length=120)
    resources: dict[str, Any] = Field(default_factory=dict)
    shard_size: int | None = Field(default=1024, ge=1, le=100000)


class QueueStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    worker_pool_id: str
    queued_run_ids: list[str] = Field(default_factory=list)
    all_ready: bool = False


class QueueVerifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    queued_run_ids: list[str] = Field(min_length=1, max_length=100)
    start_after_verification: bool = False


class QueuedRunBatchSizeRequest(BaseModel):
    batch_size: int = Field(ge=1)


class QueueMaintenanceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    worker_pool_id: str | None = None


class DraftTrainingPlanRequest(BaseModel):
    name: str
    dataset_id: str
    description: str = ""
    resources: dict[str, Any] = Field(default_factory=dict)
    training_overrides: dict[str, Any] = Field(default_factory=dict)
    initialization: dict[str, Any] = Field(default_factory=dict)


class TrainingCatalogScanRequest(BaseModel):
    root_id: str | None = None


class TrainingCatalogCompareRequest(BaseModel):
    catalog_ids: list[str]


class WorkerPoolRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    allowed_actions: list[str]
    max_workers: int | None = Field(default=None, ge=1)


class WorkerDeploymentRequest(BaseModel):
    """Profile reference only; paths, secrets, images and commands are server configuration."""
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=120)
    pool_id: str = Field(min_length=1, max_length=160)
    profile_id: str = Field(min_length=1, max_length=160)


class WorkerDeploymentStopRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timeout_seconds: float = Field(default=10.0, ge=0, le=120)


class WorkerRegistrationRequest(BaseModel):
    """An untrusted worker registration; never accepts launch commands."""
    model_config = ConfigDict(extra="forbid")
    pool_id: str
    name: str
    registration_token: str
    endpoint: str | None = None
    capabilities: dict[str, Any] = Field(default_factory=dict)
    worker_id: str | None = None


class WorkerLeaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ttl_seconds: int = Field(default=60, ge=5, le=3600)


class WorkerPollRequest(WorkerLeaseRequest):
    capabilities: dict[str, Any] | None = None


class WorkerLeaseRenewRequest(WorkerLeaseRequest):
    lease_token: str


class WorkerLeaseReleaseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lease_token: str
    outcome: str = "released"


class ArtifactDeliveryGrantRequest(BaseModel):
    """Lease-bound request for one sealed work-unit input archive."""

    model_config = ConfigDict(extra="forbid")
    lease_token: str = Field(min_length=1, max_length=512)
    ref: dict[str, Any]
    ttl_seconds: int = Field(default=300, ge=5, le=600)


class WorkerLeaseAcknowledgeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lease_token: str = Field(min_length=1, max_length=512)


class WorkerOutputUploadCreateRequest(WorkerLeaseAcknowledgeRequest):
    """Metadata only; archive bytes are uploaded through bounded part routes."""
    model_config = ConfigDict(extra="forbid")
    archive_size: int = Field(ge=1)
    archive_sha256: str = Field(min_length=64, max_length=64)
    part_size: int = Field(ge=1)


class WorkerLeaseEventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    event_id: str
    event_type: str = Field(alias="type")
    message: str
    data: dict[str, Any] = Field(default_factory=dict)


class WorkerLeaseEventsRequest(WorkerLeaseAcknowledgeRequest):
    events: list[WorkerLeaseEventRequest] = Field(min_length=1, max_length=50)


class WorkerLeaseCompleteRequest(WorkerLeaseAcknowledgeRequest):
    outcome: str
    error: str | None = Field(default=None, max_length=2000)


class WorkerLeaseResumeOutputRequest(WorkerLeaseRequest):
    """Worker identity, rather than a persisted raw lease secret, authorizes recovery."""
    model_config = ConfigDict(extra="forbid")
    worker_id: str = Field(min_length=1, max_length=160)


class WorkerLeaseCancelledRequest(WorkerLeaseAcknowledgeRequest):
    message: str | None = Field(default=None, max_length=2000)


class WorkerHeartbeatRequest(WorkerLeaseAcknowledgeRequest):
    ttl_seconds: int = Field(default=30, ge=2, le=3600)
    telemetry: dict[str, Any] = Field(default_factory=dict)


class WorkerCommandPollRequest(WorkerLeaseAcknowledgeRequest):
    wait_seconds: float = Field(default=25, ge=0, le=30)


class WorkerCommandAckRequest(WorkerLeaseAcknowledgeRequest):
    command_id: str
    status: str
    result: dict[str, Any] = Field(default_factory=dict)


class JobCommandRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: str
    reason: str = Field(default="", max_length=2000)
    command_id: str | None = None


_ROLE_RANK = {"analyst": 1, "operator": 2, "admin": 3}
_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


def role_token_digests(configured: Mapping[str, str] | None) -> dict[str, str]:
    """Validate a role-to-SHA256 configuration without handling raw secrets.

    Values are accepted as 64 lowercase/uppercase hex characters, optionally
    prefixed with ``sha256:``.  Keeping plaintext tokens out of configuration
    makes accidental process/environment disclosure less damaging.
    """
    result: dict[str, str] = {}
    for role, value in (configured or {}).items():
        if role not in _ROLE_RANK:
            raise ValueError(f"unsupported API role {role!r}")
        digest = value.removeprefix("sha256:").lower() if isinstance(value, str) else ""
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"API token digest for role {role!r} must be a SHA-256 hex digest")
        result[role] = digest
    return result


def create_app(
    orchestrator: Orchestrator, *, role_tokens: Mapping[str, str] | None = None,
    anonymous_get_limit: int = 120, anonymous_get_window_seconds: float = 60.0,
) -> FastAPI:
    """Create the control-plane API with optional configured operator roles.

    Supplying ``role_tokens`` enables the production boundary.  The optional
    form is retained only for embedded/local test construction; the launcher
    requires a configured role-token map outside explicit development mode.
    """
    if anonymous_get_limit < 1 or anonymous_get_window_seconds <= 0:
        raise ValueError("anonymous GET rate limit and window must be positive")
    configured_role_tokens = role_token_digests(role_tokens)
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Recovery remains synchronous so the readiness report is meaningful;
        # recurring remote reconciliation is subsequently durable/scheduled.
        app.state.startup_reconciliation = await run_in_threadpool(orchestrator.reconcile_startup)
        # Provider handles are intentionally ephemeral.  Reconciliation only
        # queries deterministic provider identities and marks unknown rather
        # than adopting an arbitrary process after restart.
        app.state.startup_deployments = await run_in_threadpool(orchestrator.reconcile_worker_deployments)
        orchestrator.start_operation_runner()
        yield
        orchestrator.stop_operation_runner()

    app = FastAPI(title="Oracle Builder Orchestrator API", version="0.1.0", lifespan=lifespan)
    app.state.orchestrator = orchestrator
    app.state.api_role_tokens_configured = bool(configured_role_tokens)
    app.state.anonymous_get_requests: dict[str, deque[float]] = defaultdict(deque)

    def api_role(authorization: str | None) -> str | None:
        if not authorization or not authorization.startswith("Bearer "):
            return None
        token = authorization.removeprefix("Bearer ").strip()
        if not token:
            return None
        digest = sha256(token.encode("utf-8")).hexdigest()
        for role, expected in configured_role_tokens.items():
            if hmac.compare_digest(digest, expected):
                return role
        return None

    def is_worker_boundary(path: str) -> bool:
        # Worker credentials are scoped and validated by their dedicated lease
        # handlers.  Never interpret one as an operator browser/CLI token.
        return path == "/v1/workers:register" or path.startswith("/v1/workers/") or path.startswith("/v1/worker-leases/") or path.startswith("/v1/worker-artifacts/")

    def is_analytical_request(path: str) -> bool:
        return path in {"/v1/model-previews", "/v1/artifacts/catalog/query"} or path.endswith(":preflight")

    @app.middleware("http")
    async def role_auth_and_anonymous_read_limit(request: Request, call_next):
        path, method = request.url.path, request.method.upper()
        authorization = request.headers.get("authorization")
        role = api_role(authorization)
        request.state.api_role = role
        if method in _SAFE_METHODS:
            if configured_role_tokens and role is None:
                # Simple per-process protection for intentionally anonymous
                # catalog/health reads in token-configured deployments. An
                # explicitly unauthenticated local development app has no
                # production trust boundary to protect and must not throttle
                # its own Web GUI's initial catalog fan-out. A reverse proxy
                # remains responsible for distributed rate limiting and source
                # address correctness in production.
                source = request.client.host if request.client else "unknown"
                now = asyncio.get_running_loop().time()
                bucket = app.state.anonymous_get_requests[source]
                cutoff = now - anonymous_get_window_seconds
                while bucket and bucket[0] <= cutoff:
                    bucket.popleft()
                if len(bucket) >= anonymous_get_limit:
                    return JSONResponse({"detail": "Anonymous read rate limit exceeded"}, status_code=429,
                                        headers={"Retry-After": str(max(1, int(anonymous_get_window_seconds)))})
                bucket.append(now)
            return await call_next(request)
        if is_worker_boundary(path):
            return await call_next(request)
        if configured_role_tokens:
            required_rank = _ROLE_RANK["analyst"] if is_analytical_request(path) else _ROLE_RANK["operator"]
            actual_rank = _ROLE_RANK.get(role or "", 0)
            if actual_rank < required_rank:
                return JSONResponse({"detail": "An authorized API role token is required"}, status_code=401)
        response = await call_next(request)
        # Durable auditing is supplied by the Orchestrator when available. Do
        # not log request bodies, bearer tokens, artifact grants, or paths.
        if response.status_code < 400 and method in {"POST", "PUT", "PATCH", "DELETE"} and hasattr(orchestrator, "record_audit_event"):
            try:
                await run_in_threadpool(
                    orchestrator.record_audit_event, actor_role=role or "local-unconfigured",
                    method=method, path=path, outcome="succeeded", request_id=request.headers.get("x-request-id"),
                )
            except Exception:
                # An audit failure must not cause a completed operation to be
                # reattempted. The service implementation records its own
                # availability diagnostics.
                pass
        return response

    def required(value: dict[str, Any] | None, label: str) -> dict[str, Any]:
        if value is None:
            raise HTTPException(status_code=404, detail=f"{label} was not found")
        return value

    def worker_bearer(authorization: str | None) -> str:
        """Extract the worker credential without accepting alternate schemes."""
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Worker bearer token is required")
        token = authorization.removeprefix("Bearer ").strip()
        if not token:
            raise HTTPException(status_code=401, detail="Worker bearer token is required")
        return token

    @app.get("/health/live")
    def live() -> dict[str, str]: return {"status": "ok"}

    @app.get("/health/ready")
    def ready() -> dict[str, Any]:
        pools = orchestrator.worker_pools()
        workers = orchestrator.registered_workers()
        active = sum(worker.get("state") in {"idle", "leased"} for worker in workers)
        return {
            "status": "ready" if active else "degraded",
            "database": "ready",
            "worker_pools": {"configured": len(pools), "workers": len(workers), "active": active},
            "startup_reconciliation": getattr(app.state, "startup_reconciliation", None),
        }

    @app.get("/v1/operations")
    def operations(status: str | None = Query(default=None), limit: int = Query(default=100, ge=1, le=500)) -> dict[str, Any]:
        return {"operations": orchestrator.operations(status=status, limit=limit)}

    @app.get("/v1/operations/{operation_id}")
    def operation(operation_id: str) -> dict[str, Any]:
        return required(orchestrator.operation(operation_id), "Operation")

    @app.get("/v1/system/logs")
    async def server_logs(tail_lines: int = Query(default=400, ge=1, le=2_000)) -> dict[str, Any]:
        return await run_in_threadpool(orchestrator.server_logs, tail_lines=tail_lines)

    @app.post("/v1/workers:reconcile/schedule", status_code=202)
    async def schedule_worker_reconciliation() -> dict[str, Any]:
        """Schedule durable pull-worker liveness/publication reconciliation."""
        return {"operation": await run_in_threadpool(orchestrator.create_operation, "worker_reconciliation")}

    @app.get("/v1/events")
    async def operation_event_stream(request: Request, after: int | None = Query(default=None, ge=0), operation_id: str | None = Query(default=None)) -> StreamingResponse:
        """Resumable SSE sourced entirely from the durable event ledger."""
        # Native EventSource reconnects send Last-Event-ID, while non-browser
        # clients may use the explicit query cursor.  Honor both forms.
        try:
            cursor = after if after is not None else max(0, int(request.headers.get("last-event-id", "0")))
        except ValueError:
            cursor = 0
        async def stream():
            nonlocal cursor
            while not await request.is_disconnected():
                events = await run_in_threadpool(
                    orchestrator.operation_events, after=cursor, operation_id=operation_id
                )
                if events:
                    for event in events:
                        cursor = event["sequence"]
                        yield f"id: {cursor}\nevent: {event['event_type']}\ndata: {json.dumps(event, separators=(',', ':'))}\n\n"
                else:
                    yield ": keepalive\n\n"
                    await asyncio.sleep(1)
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # Pull workers have a separate admission boundary from lifecycle-managed
    # local workers.  These endpoints expose sealed work units only; neither
    # registration nor a lease can carry a command, local path, or provider.
    @app.get("/v1/worker-pools")
    def worker_pools() -> dict[str, Any]:
        return {"pools": orchestrator.worker_pools()}

    @app.post("/v1/worker-pools", status_code=201)
    def create_worker_pool(body: WorkerPoolRequest) -> dict[str, Any]:
        try:
            return orchestrator.create_worker_pool(**body.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/worker-pools/{pool_id}")
    def worker_pool(pool_id: str) -> dict[str, Any]:
        return required(orchestrator.worker_pool(pool_id), "Worker pool")

    @app.get("/v1/workers")
    def registered_workers(pool_id: str | None = Query(default=None)) -> dict[str, Any]:
        return {"workers": orchestrator.registered_workers(pool_id=pool_id)}

    @app.get("/v1/worker-deployments")
    def worker_deployments() -> dict[str, Any]:
        return {"deployments": orchestrator.worker_deployments()}

    @app.post("/v1/worker-deployments", status_code=201)
    def create_worker_deployment(body: WorkerDeploymentRequest) -> dict[str, Any]:
        try:
            return orchestrator.create_worker_deployment(**body.model_dump())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Worker pool was not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/worker-deployments/{deployment_id}")
    def worker_deployment(deployment_id: str) -> dict[str, Any]:
        return required(orchestrator.worker_deployment(deployment_id), "Worker deployment")

    @app.post("/v1/worker-deployments/{deployment_id}:start")
    def start_worker_deployment(deployment_id: str) -> dict[str, Any]:
        try:
            return orchestrator.start_worker_deployment(deployment_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Worker deployment was not found") from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/worker-deployments/{deployment_id}:stop")
    def stop_worker_deployment(deployment_id: str, body: WorkerDeploymentStopRequest) -> dict[str, Any]:
        try:
            return orchestrator.stop_worker_deployment(deployment_id, timeout_seconds=body.timeout_seconds)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Worker deployment was not found") from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/worker-deployments:reconcile")
    def reconcile_worker_deployments() -> dict[str, Any]:
        return {"deployments": orchestrator.reconcile_worker_deployments()}

    @app.post("/v1/workers:register", status_code=201)
    def register_worker(body: WorkerRegistrationRequest) -> dict[str, Any]:
        try:
            # A supplied worker_id is intentionally rejected at the API
            # boundary: identity is server-issued, so a client cannot adopt
            # another worker's durable record during registration.
            if body.worker_id is not None:
                raise ValueError("worker_id is server-issued and may not be supplied during registration")
            return orchestrator.register_worker(**body.model_dump(exclude={"worker_id"}))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Worker pool was not found") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/workers/{worker_id}")
    def registered_worker(worker_id: str) -> dict[str, Any]:
        return required(orchestrator.registered_worker(worker_id), "Registered worker")

    @app.post("/v1/workers/{worker_id}:lease", response_model=None)
    def acquire_worker_lease(
        worker_id: str, body: WorkerPollRequest,
        authorization: str | None = Header(default=None),
    ) -> Any:
        try:
            lease = orchestrator.acquire_next_worker_lease(
                worker_id=worker_id, worker_token=worker_bearer(authorization), ttl_seconds=body.ttl_seconds, capabilities=body.capabilities,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Registered worker was not found") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if lease is None:
            return Response(status_code=204)
        # Keep the worker response intentionally narrow.  The lease record
        # carries only renewal/release metadata; the executable contract is a
        # separate, path-free sealed WorkUnit.
        work_unit = lease.pop("work_unit")
        return {"lease": lease, "work_unit": work_unit}

    @app.get("/v1/worker-leases")
    def worker_leases(worker_id: str | None = Query(default=None)) -> dict[str, Any]:
        return {"leases": orchestrator.worker_leases(worker_id=worker_id)}

    @app.get("/v1/workers/{worker_id}/heartbeat")
    def worker_heartbeat_status(worker_id: str) -> dict[str, Any]:
        return {"heartbeat": orchestrator.worker_heartbeat_status(worker_id)}

    @app.get("/v1/jobs/{job_id}/commands")
    def job_commands(job_id: str) -> dict[str, Any]:
        return {"commands": orchestrator.job_commands(job_id)}

    @app.get("/v1/execution-runs/{run_id}")
    def execution_run(run_id: str) -> dict[str, Any]:
        return {"run": required(orchestrator.execution_run(run_id), "Execution run")}

    @app.post("/v1/jobs/{job_id}/commands", status_code=202)
    def request_job_command(job_id: str, body: JobCommandRequest) -> dict[str, Any]:
        try:
            return {"command": orchestrator.request_job_command(job_id, **body.model_dump())}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job was not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    async def worker_control_request(lease_id, authorization, body, method):
        try:
            lease = required(orchestrator.worker_lease(lease_id), "Worker lease")
            return await run_in_threadpool(method, lease_id=lease_id, worker_id=lease["worker_id"],
                worker_token=worker_bearer(authorization), **body.model_dump())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Worker lease or command was not found") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/worker-leases/{lease_id}:heartbeat")
    async def worker_lease_heartbeat(lease_id: str, body: WorkerHeartbeatRequest, authorization: str | None = Header(default=None)):
        return await worker_control_request(lease_id, authorization, body, orchestrator.heartbeat_worker_lease)

    @app.post("/v1/worker-leases/{lease_id}/commands:poll")
    async def worker_lease_commands(lease_id: str, body: WorkerCommandPollRequest, authorization: str | None = Header(default=None)):
        return await worker_control_request(lease_id, authorization, body, orchestrator.poll_worker_commands)

    @app.post("/v1/worker-leases/{lease_id}/commands:acknowledge")
    async def worker_command_acknowledge(lease_id: str, body: WorkerCommandAckRequest, authorization: str | None = Header(default=None)):
        return await worker_control_request(lease_id, authorization, body, orchestrator.acknowledge_worker_command)

    @app.get("/v1/worker-leases/{lease_id}")
    def worker_lease(lease_id: str) -> dict[str, Any]:
        return required(orchestrator.worker_lease(lease_id), "Worker lease")

    @app.post("/v1/worker-leases/{lease_id}:renew")
    def renew_worker_lease(
        lease_id: str, body: WorkerLeaseRenewRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        try:
            lease = required(orchestrator.worker_lease(lease_id), "Worker lease")
            # The per-lease bearer is intentionally not sufficient by itself:
            # control requests also prove possession of the registered
            # worker's credential, binding the request to the lease owner.
            orchestrator.authenticate_worker(
                worker_id=str(lease["worker_id"]), worker_token=worker_bearer(authorization),
            )
            return orchestrator.renew_worker_lease(lease_id=lease_id, **body.model_dump())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Worker lease was not found") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/worker-leases/{lease_id}:release")
    def release_worker_lease(
        lease_id: str, body: WorkerLeaseReleaseRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        try:
            lease = required(orchestrator.worker_lease(lease_id), "Worker lease")
            orchestrator.authenticate_worker(
                worker_id=str(lease["worker_id"]), worker_token=worker_bearer(authorization),
            )
            return orchestrator.release_worker_lease(lease_id=lease_id, **body.model_dump())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Worker lease was not found") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    def active_worker_lease(
        lease_id: str, *, authorization: str | None, lease_token: str,
    ) -> dict[str, Any]:
        """Authenticate both bearers before an artifact is disclosed.

        ``worker_id`` is read from the server's public lease record, never
        supplied by the caller.  The service rechecks that binding atomically
        along with active/expiry state and the sealed WorkUnit.
        """
        lease = required(orchestrator.worker_lease(lease_id), "Worker lease")
        return orchestrator.authenticate_active_worker_lease(
            lease_id=lease_id,
            worker_id=str(lease["worker_id"]),
            worker_token=worker_bearer(authorization),
            lease_token=lease_token,
        )

    @app.post("/v1/worker-leases/{lease_id}/cancellation")
    def worker_lease_cancellation(
        lease_id: str,
        body: WorkerLeaseAcknowledgeRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Let only the active lease owner observe a pull-job cancellation."""
        try:
            context = active_worker_lease(lease_id, authorization=authorization, lease_token=body.lease_token)
            return orchestrator.worker_lease_cancellation(
                lease_id=lease_id, worker_id=str(context["lease"]["worker_id"]),
                worker_token=worker_bearer(authorization), lease_token=body.lease_token,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Worker lease was not found") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/worker-leases/{lease_id}:cancelled")
    def acknowledge_worker_lease_cancellation(
        lease_id: str,
        body: WorkerLeaseCancelledRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        try:
            context = active_worker_lease(lease_id, authorization=authorization, lease_token=body.lease_token)
            return orchestrator.cancel_worker_lease(
                lease_id=lease_id, worker_id=str(context["lease"]["worker_id"]),
                worker_token=worker_bearer(authorization), lease_token=body.lease_token,
                message=body.message,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Worker lease was not found") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/worker-leases/{lease_id}:artifact-grants", status_code=201)
    def issue_artifact_delivery_grant(
        lease_id: str,
        body: ArtifactDeliveryGrantRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Issue a short-lived, one-use download authority for a lease input."""
        try:
            context = active_worker_lease(
                lease_id, authorization=authorization, lease_token=body.lease_token,
            )
            ref = ArtifactRef.from_dict(body.ref)
            unit = context["work_unit"]
            permitted = list((unit.get("inputs") or {}).values())
            if unit.get("configuration") is not None:
                permitted.append(unit["configuration"])
            # Staging is intentionally absent: the worker must publish via a
            # later staged-output protocol, never download a server path.
            if ref.to_dict() not in permitted:
                raise PermissionError("Artifact is not an input to this leased work unit")
            lease = context["lease"]
            grant = orchestrator.artifact_store.issue_materialization_grant(
                ref, lease_id=lease_id, job_id=str(lease["job_id"]), ttl_seconds=body.ttl_seconds,
            )
            return {
                "grant": grant.delivery_dict(),
                "download_path": f"/v1/worker-leases/{lease_id}/artifact-grants/{grant.grant_id}:download",
            }
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Worker lease or artifact was not found") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/worker-leases/{lease_id}/artifact-grants/{grant_id}:download")
    def download_artifact_delivery_grant(
        lease_id: str,
        grant_id: str,
        authorization: str | None = Header(default=None),
        x_oracle_lease_token: str | None = Header(default=None),
        x_oracle_artifact_grant: str | None = Header(default=None),
    ) -> StreamingResponse:
        """Stream a gzip artifact archive after worker, lease, and grant auth."""
        if not x_oracle_lease_token or not x_oracle_artifact_grant:
            raise HTTPException(status_code=401, detail="Lease and artifact grant credentials are required")
        try:
            context = active_worker_lease(
                lease_id, authorization=authorization, lease_token=x_oracle_lease_token,
            )
            lease = context["lease"]
            stream = orchestrator.artifact_store.stream_grant_archive(
                x_oracle_artifact_grant,
                lease_id=lease_id,
                job_id=str(lease["job_id"]),
                expected_grant_id=grant_id,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Worker lease or artifact was not found") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return StreamingResponse(
            stream,
            media_type="application/gzip",
            headers={
                "Cache-Control": "no-store",
                "Content-Disposition": f'attachment; filename="oracle-artifact-{grant_id}.tar.gz"',
            },
        )

    @app.post("/v1/worker-leases/{lease_id}:acknowledge")
    def acknowledge_worker_lease(
        lease_id: str,
        body: WorkerLeaseAcknowledgeRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        try:
            context = active_worker_lease(lease_id, authorization=authorization, lease_token=body.lease_token)
            return orchestrator.acknowledge_worker_lease(
                lease_id=lease_id,
                worker_id=str(context["lease"]["worker_id"]),
                worker_token=worker_bearer(authorization),
                lease_token=body.lease_token,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Worker lease was not found") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/worker-leases/{lease_id}:events")
    def append_worker_lease_events(
        lease_id: str,
        body: WorkerLeaseEventsRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        try:
            context = active_worker_lease(lease_id, authorization=authorization, lease_token=body.lease_token)
            worker_id = str(context["lease"]["worker_id"])
            worker_token = worker_bearer(authorization)
            events = [
                orchestrator.append_worker_lease_event(
                    lease_id=lease_id, worker_id=worker_id, worker_token=worker_token,
                    lease_token=body.lease_token, event_id=event.event_id,
                    event_type=event.event_type, message=event.message, data=event.data,
                )
                for event in body.events
            ]
            return {"events": events}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Worker lease was not found") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.put("/v1/worker-leases/{lease_id}/staging")
    async def upload_worker_lease_staging(
        lease_id: str,
        request: Request,
        authorization: str | None = Header(default=None),
        x_oracle_lease_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Spool a worker archive locally, then extract it into owned staging.

        The worker never supplies a filename or destination.  Spooling avoids
        retaining a potentially multi-gigabyte archive in the API process;
        the service receives only this API-owned temporary file.
        """
        if not x_oracle_lease_token:
            raise HTTPException(status_code=401, detail="Lease credential is required")
        content_length = request.headers.get("content-length")
        try:
            if content_length and int(content_length) > orchestrator.upload_limit_bytes:
                raise HTTPException(status_code=413, detail="Output archive exceeds the upload limit")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="Content-Length must be an integer") from exc
        temporary = (
            Path(orchestrator.artifact_root) / "worker-upload-spool" /
            f"{uuid.uuid4().hex}.archive"
        )
        written = 0
        try:
            temporary.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("xb") as handle:
                async for chunk in request.stream():
                    written += len(chunk)
                    if written > orchestrator.upload_limit_bytes:
                        raise HTTPException(status_code=413, detail="Output archive exceeds the upload limit")
                    await run_in_threadpool(handle.write, chunk)
            context = active_worker_lease(
                lease_id, authorization=authorization, lease_token=x_oracle_lease_token,
            )
            return await run_in_threadpool(
                orchestrator.upload_worker_lease_output_archive,
                lease_id=lease_id,
                worker_id=str(context["lease"]["worker_id"]),
                worker_token=worker_bearer(authorization),
                lease_token=x_oracle_lease_token,
                archive=temporary,
            )
        except HTTPException:
            raise
        except ClientDisconnect as exc:
            # A worker can retry through the durable multipart protocol; a
            # dropped legacy stream must not surface as an unhandled ASGI
            # exception or be misreported as execution failure.
            raise HTTPException(status_code=499, detail="Client disconnected during output upload") from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Worker lease was not found") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            temporary.unlink(missing_ok=True)

    @app.post("/v1/worker-leases/{lease_id}/output-uploads", status_code=201)
    def create_worker_output_upload(
        lease_id: str,
        body: WorkerOutputUploadCreateRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        """Create or resume a lease-bound, multipart output transfer."""
        try:
            context = active_worker_lease(lease_id, authorization=authorization, lease_token=body.lease_token)
            return orchestrator.create_worker_output_upload(
                lease_id=lease_id, worker_id=str(context["lease"]["worker_id"]),
                worker_token=worker_bearer(authorization), lease_token=body.lease_token,
                archive_size=body.archive_size, archive_sha256=body.archive_sha256, part_size=body.part_size,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Worker lease was not found") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/worker-output-uploads/{upload_id}")
    def get_worker_output_upload(
        upload_id: str,
        authorization: str | None = Header(default=None),
        x_oracle_lease_token: str | None = Header(default=None),
    ) -> dict[str, Any]:
        if not x_oracle_lease_token:
            raise HTTPException(status_code=401, detail="Lease credential is required")
        try:
            # The service obtains the lease from the opaque upload id, so a
            # caller cannot use this route to probe another worker's uploads.
            return orchestrator.worker_output_upload(
                upload_id=upload_id, worker_id="", worker_token=worker_bearer(authorization), lease_token=x_oracle_lease_token,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Output upload was not found") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/jobs/{job_id}/output-upload")
    def job_output_upload(job_id: str) -> dict[str, Any]:
        # This intentionally exposes only transfer counters/state already
        # visible in job events—never upload paths, digests, or credentials.
        return {"upload": orchestrator.job_worker_output_upload(job_id)}

    @app.put("/v1/worker-output-uploads/{upload_id}/parts/{part_number}")
    async def put_worker_output_upload_part(
        upload_id: str, part_number: int, request: Request,
        authorization: str | None = Header(default=None),
        x_oracle_lease_token: str | None = Header(default=None),
        x_oracle_part_sha256: str | None = Header(default=None),
        content_range: str | None = Header(default=None),
    ) -> dict[str, Any]:
        if not x_oracle_lease_token or not x_oracle_part_sha256:
            raise HTTPException(status_code=401, detail="Lease credential and part digest are required")
        temporary: Path | None = None
        try:
            # Resolve the layout before accepting bytes. This bounds an
            # interrupted or malicious request at one declared part.
            worker_token = worker_bearer(authorization)
            status = orchestrator.worker_output_upload(
                upload_id=upload_id, worker_id="", worker_token=worker_token, lease_token=x_oracle_lease_token,
            )
            if part_number < 0 or part_number >= int(status["part_count"]):
                raise HTTPException(status_code=422, detail="part_number is outside this upload")
            expected_size = int(status["part_size"])
            if part_number == int(status["part_count"]) - 1:
                expected_size = int(status["archive_size"]) - part_number * int(status["part_size"])
            expected_range = f"bytes {part_number * int(status['part_size'])}-{part_number * int(status['part_size']) + expected_size - 1}/{status['archive_size']}"
            if content_range != expected_range:
                raise HTTPException(status_code=422, detail="Content-Range does not match the declared part layout")
            content_length = request.headers.get("content-length")
            if content_length is not None and int(content_length) != expected_size:
                raise HTTPException(status_code=422, detail="Content-Length does not match the declared part layout")
            spool = orchestrator._worker_output_upload_root() / "incoming"
            spool.mkdir(mode=0o700, exist_ok=True)
            temporary = spool / f"{uuid.uuid4().hex}.part"
            written = 0
            with temporary.open("xb") as handle:
                async for chunk in request.stream():
                    written += len(chunk)
                    if written > expected_size:
                        raise HTTPException(status_code=413, detail="Output part exceeds its declared size")
                    await run_in_threadpool(handle.write, chunk)
            if written != expected_size:
                raise HTTPException(status_code=422, detail="Output part is shorter than its declared size")
            # Resolve worker id only after the service has authenticated the
            # bearer against the upload's lease; it is never client supplied.
            lease = orchestrator.worker_lease(str(status["lease_id"]))
            if lease is None: raise KeyError(upload_id)
            return await run_in_threadpool(
                orchestrator.upload_worker_output_part,
                upload_id=upload_id, worker_id=str(lease["worker_id"]), worker_token=worker_token,
                lease_token=x_oracle_lease_token, part_number=part_number,
                part_sha256=x_oracle_part_sha256, part=temporary,
            )
        except ClientDisconnect as exc:
            # A dropped remote connection is ordinary retryable transport
            # failure. Do not turn it into an ASGI traceback or alter the job.
            raise HTTPException(status_code=499, detail="Client disconnected before output part completed") from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Output upload was not found") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finally:
            if temporary is not None: temporary.unlink(missing_ok=True)

    @app.post("/v1/worker-output-uploads/{upload_id}:finalize")
    def finalize_worker_output_upload(
        upload_id: str, body: WorkerLeaseAcknowledgeRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        try:
            worker_token = worker_bearer(authorization)
            status = orchestrator.worker_output_upload(upload_id=upload_id, worker_id="", worker_token=worker_token, lease_token=body.lease_token)
            lease = orchestrator.worker_lease(str(status["lease_id"]))
            if lease is None: raise KeyError(upload_id)
            return orchestrator.finalize_worker_output_upload(
                upload_id=upload_id, worker_id=str(lease["worker_id"]), worker_token=worker_token, lease_token=body.lease_token,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Output upload was not found") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/worker-leases/{lease_id}:defer-output")
    def defer_worker_output_publication(
        lease_id: str, body: WorkerLeaseAcknowledgeRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        try:
            context = active_worker_lease(lease_id, authorization=authorization, lease_token=body.lease_token)
            return orchestrator.defer_worker_output_publication(
                lease_id=lease_id, worker_id=str(context["lease"]["worker_id"]),
                worker_token=worker_bearer(authorization), lease_token=body.lease_token,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Worker lease was not found") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/worker-leases/{lease_id}:resume-output")
    def resume_worker_output_publication(
        lease_id: str, body: WorkerLeaseResumeOutputRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        try:
            # Only the registered worker bearer is required here. The endpoint
            # rotates a new lease token after proving the lease belongs to it.
            return orchestrator.resume_worker_output_publication(
                lease_id=lease_id, worker_id=body.worker_id,
                worker_token=worker_bearer(authorization), ttl_seconds=body.ttl_seconds,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Worker lease was not found") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/worker-leases/{lease_id}:complete")
    def complete_worker_lease(
        lease_id: str,
        body: WorkerLeaseCompleteRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        if body.outcome not in {"succeeded", "failed"}:
            raise HTTPException(status_code=422, detail="outcome must be succeeded or failed")
        try:
            lease = required(orchestrator.worker_lease(lease_id), 'Worker lease')
            return orchestrator.complete_worker_lease(
                lease_id=lease_id,
                worker_id=str(lease["worker_id"]),
                worker_token=worker_bearer(authorization),
                lease_token=body.lease_token,
                success=body.outcome == "succeeded",
                message=body.error or ("Worker completed execution" if body.outcome == "succeeded" else "Worker execution failed"),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Worker lease was not found") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/files/roots")
    def file_roots() -> dict[str, Any]: return {"roots": orchestrator.file_roots()}

    @app.get("/v1/files")
    def files(root: str = Query(default="workspace"), path: str = Query(default="")) -> dict[str, Any]:
        try: return orchestrator.files(root, path or ".")
        except KeyError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, ValueError) as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/uploads/sessions", status_code=201)
    def create_upload_session(body: UploadSessionRequest) -> dict[str, Any]:
        try:
            return orchestrator.create_upload_session(**body.model_dump())
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/uploads/sessions/{upload_id}")
    def upload_session(upload_id: str) -> dict[str, Any]:
        try:
            return orchestrator.upload_session(upload_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.put("/v1/uploads/sessions/{upload_id}/part")
    async def upload_session_part(upload_id: str, request: Request) -> dict[str, Any]:
        """Receive exactly one fixed-size range without buffering it in RAM."""
        header = request.headers.get("content-range", "")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", header)
        if match is None:
            raise HTTPException(status_code=422, detail="Content-Range must be bytes start-end/total")
        start, end, total = (int(value) for value in match.groups())
        if end < start:
            raise HTTPException(status_code=422, detail="Content-Range end must not precede start")
        try:
            prepared = orchestrator.prepare_upload_session_part(
                upload_id, offset_bytes=start, size_bytes=end - start + 1, total_bytes=total,
            )
            if prepared["idempotent"]:
                return {"upload": orchestrator.upload_session(upload_id), "idempotent": True}
            temporary = Path(str(prepared["temporary_path"]))
            written, digest = 0, sha256()
            with temporary.open("r+b") as handle:
                handle.seek(start)
                async for chunk in request.stream():
                    written += len(chunk)
                    if written > end - start + 1:
                        raise HTTPException(status_code=422, detail="Upload chunk exceeds its declared range")
                    digest.update(chunk)
                    await run_in_threadpool(handle.write, chunk)
            expected_size = end - start + 1
            if written != expected_size:
                raise HTTPException(status_code=422, detail="Upload chunk did not match its declared range")
            upload = orchestrator.record_upload_session_part(
                upload_id, offset_bytes=start, size_bytes=written, sha256_hex=digest.hexdigest(),
            )
            return {"upload": upload, "idempotent": False}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except HTTPException:
            raise
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/uploads/sessions/{upload_id}:complete")
    def complete_upload_session(upload_id: str) -> dict[str, Any]:
        try:
            return orchestrator.complete_upload_session(upload_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete("/v1/uploads/sessions/{upload_id}")
    def cancel_upload_session(upload_id: str) -> dict[str, Any]:
        try:
            return orchestrator.cancel_upload_session(upload_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Register this compatibility route after the explicit session routes:
    # otherwise its broad two-segment pattern would treat ``id:complete`` as
    # an old ``kind/filename`` request.
    @app.post("/v1/uploads/{kind}/{filename}", status_code=201)
    async def upload(kind: str, filename: str, request: Request) -> dict[str, Any]:
        try:
            destination = orchestrator.upload_destination(kind, filename)
            if destination.exists():
                raise HTTPException(status_code=409, detail="An upload with this filename already exists")
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > orchestrator.upload_limit_bytes:
                raise HTTPException(status_code=413, detail="Upload exceeds configured size limit")
            temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")
            written = 0
            try:
                with temporary.open("xb") as handle:
                    async for chunk in request.stream():
                        written += len(chunk)
                        if written > orchestrator.upload_limit_bytes:
                            raise HTTPException(status_code=413, detail="Upload exceeds configured size limit")
                        await run_in_threadpool(handle.write, chunk)
                temporary.replace(destination)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
            return {"kind": kind, "filename": filename, "path": str(destination), "size_bytes": written}
        except HTTPException:
            raise
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/datasets")
    def datasets() -> dict[str, Any]: return {"datasets": orchestrator.datasets()}

    @app.get("/v1/training-catalog")
    def training_catalog(root_id: str | None = Query(default=None)) -> dict[str, Any]:
        return {"roots": orchestrator.training_catalog_roots_info(), "entries": orchestrator.training_catalog(root_id=root_id), "bundles": orchestrator.training_catalog_bundles(root_id=root_id)}

    @app.post("/v1/training-catalog:scan")
    def scan_training_catalog(body: TrainingCatalogScanRequest) -> dict[str, Any]:
        try: return orchestrator.scan_training_catalog(**body.model_dump())
        except KeyError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, ValueError) as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/training-catalog:scan/schedule", status_code=202)
    async def schedule_training_catalog_scan(body: TrainingCatalogScanRequest) -> dict[str, Any]:
        try: return {"operation": await run_in_threadpool(orchestrator.create_operation, "training_catalog_scan", body.model_dump())}
        except KeyError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/v1/training-catalog/{catalog_id}:freeze")
    def freeze_training_catalog_entry(catalog_id: str) -> dict[str, Any]:
        try: return orchestrator.freeze_training_catalog_entry(catalog_id)
        except KeyError as exc: raise HTTPException(status_code=404, detail="Training catalog entry was not found") from exc
        except (OSError, ValueError) as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/training-catalog/{catalog_id}")
    def training_catalog_entry(catalog_id: str) -> dict[str, Any]:
        entry = required(orchestrator.training_catalog_entry(catalog_id), "Training catalog entry")
        view = orchestrator._training_catalog_view(entry)
        return {"entry": view, "classes": view.get("classes", []), "dimensions": view.get("dimensions", {}), "warnings": view.get("warnings", [])}

    @app.get("/v1/training-catalog/{catalog_id}/previews")
    def training_catalog_previews(catalog_id: str, label: str | None = Query(default=None), offset: int = Query(default=0, ge=0), limit: int = Query(default=24, ge=1, le=100)) -> dict[str, Any]:
        try: return orchestrator.training_catalog_previews(catalog_id, label=label, offset=offset, limit=limit)
        except KeyError as exc: raise HTTPException(status_code=404, detail="Training catalog entry was not found") from exc

    @app.get("/v1/training-catalog/{catalog_id}/previews/{item_id}")
    def training_catalog_preview_image(catalog_id: str, item_id: str, max_size: int = Query(default=320, ge=32, le=1024)) -> Response:
        try:
            return Response(orchestrator.training_catalog_preview_image(catalog_id, item_id, max_size=max_size), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})
        except (KeyError, FileNotFoundError) as exc: raise HTTPException(status_code=404, detail="Training catalog preview was not found") from exc
        except (OSError, ValueError) as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/training-catalog:compare")
    def compare_training_catalog(body: TrainingCatalogCompareRequest) -> dict[str, Any]:
        try: return orchestrator.compare_training_catalog(body.catalog_ids)
        except KeyError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/model-setups/{architecture}")
    def model_setup(architecture: str) -> dict[str, Any]:
        try: return orchestrator.model_setup(architecture)
        except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/config-schema")
    def configuration_schema() -> dict[str, Any]: return orchestrator.configuration_schema()

    @app.post("/v1/model-previews")
    def model_preview(body: ModelPreviewRequest) -> dict[str, Any]:
        try: return orchestrator.model_preview(**body.model_dump())
        except KeyError as exc: raise HTTPException(status_code=404, detail="Dataset was not found") from exc
        except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/datasets:ingest", status_code=201)
    def ingest_dataset(body: DatasetIngestRequest) -> dict[str, Any]:
        try: return orchestrator.ingest_dataset(body.path)
        except (OSError, ValueError) as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/datasets/{dataset_id}:download")
    def download_dataset(dataset_id: str) -> FileResponse:
        try:
            path = orchestrator.dataset_download_path(dataset_id)
            return FileResponse(path, media_type="application/x-sqlite3", filename=path.name)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail="Dataset file was not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/datasets/{dataset_id}")
    def dataset(dataset_id: str) -> dict[str, Any]: return required(orchestrator.dataset(dataset_id), "Dataset")

    @app.get("/v1/datasets/{dataset_id}/detail")
    def dataset_detail(dataset_id: str) -> dict[str, Any]:
        try: return orchestrator.dataset_detail(dataset_id)
        except KeyError as exc: raise HTTPException(status_code=404, detail="Dataset was not found") from exc
        except (OSError, ValueError) as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/datasets/{dataset_id}/previews")
    def dataset_previews(dataset_id: str, offset: int = Query(default=0, ge=0), limit: int = Query(default=24, ge=1, le=100)) -> dict[str, Any]:
        try: return orchestrator.dataset_previews(dataset_id, offset=offset, limit=limit)
        except KeyError as exc: raise HTTPException(status_code=404, detail="Dataset was not found") from exc
        except (OSError, ValueError) as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/datasets/{dataset_id}/previews/{item_id}")
    def dataset_preview_image(dataset_id: str, item_id: str, kind: str = Query(default="image"), max_size: int = Query(default=320, ge=32, le=1024)) -> Response:
        try:
            return Response(orchestrator.dataset_preview_image(dataset_id, item_id, kind=kind, max_size=max_size), media_type="image/jpeg",
                            headers={"Cache-Control": "private, max-age=3600"})
        except (KeyError, FileNotFoundError) as exc: raise HTTPException(status_code=404, detail="Dataset preview was not found") from exc
        except (OSError, ValueError) as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/catalog:scan")
    def scan(body: ScanRequest) -> dict[str, Any]:
        try: return orchestrator.scan(body.root)
        except (OSError, ValueError) as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/catalog:scan/schedule", status_code=202)
    async def schedule_scan(body: ScanRequest) -> dict[str, Any]:
        try: return {"operation": await run_in_threadpool(orchestrator.create_operation, "catalog_scan", body.model_dump())}
        except (OSError, ValueError) as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/artifacts")
    def artifacts() -> dict[str, Any]: return {"artifacts": orchestrator.artifacts()}

    @app.get("/v1/artifact-replicas")
    def artifact_replicas(status: str | None = Query(default=None), limit: int = Query(default=200, ge=1, le=500)) -> dict[str, Any]:
        try: return {"replications": orchestrator.artifact_replications(status=status, limit=limit)}
        except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/artifact-replicas:retry", status_code=202)
    async def retry_artifact_replicas(body: ArtifactReplicaOperationRequest) -> dict[str, Any]:
        try:
            refs = [ArtifactRef.from_dict(value).to_dict() for value in body.refs]
            return {"operation": await run_in_threadpool(orchestrator.create_operation, "artifact_replication", {"action": "retry", "refs": refs})}
        except (TypeError, ValueError) as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/artifact-replicas:verify", status_code=202)
    async def verify_artifact_replicas(body: ArtifactReplicaOperationRequest) -> dict[str, Any]:
        try:
            refs = [ArtifactRef.from_dict(value).to_dict() for value in body.refs]
            return {"operation": await run_in_threadpool(orchestrator.create_operation, "artifact_replication", {"action": "verify", "refs": refs})}
        except (TypeError, ValueError) as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/artifact-replicas:restore", status_code=202)
    async def restore_artifact_replicas(body: ArtifactReplicaOperationRequest) -> dict[str, Any]:
        try:
            refs = [ArtifactRef.from_dict(value).to_dict() for value in body.refs]
            return {"operation": await run_in_threadpool(orchestrator.create_operation, "artifact_replication", {"action": "restore", "refs": refs})}
        except (TypeError, ValueError) as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/artifacts/catalog")
    def artifact_catalog(filters: str | None = Query(default=None), sort: str = Query(default="updated_at"), order: str = Query(default="desc"), offset: int = Query(default=0, ge=0), limit: int = Query(default=100, ge=1, le=500)) -> dict[str, Any]:
        import json
        try:
            parsed = json.loads(filters) if filters else {}
            if not isinstance(parsed, dict): raise ValueError("filters must be a JSON object")
            return orchestrator.artifact_catalog_query(filters=parsed, sort=sort, order=order, offset=offset, limit=limit)
        except (json.JSONDecodeError, ValueError) as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/artifacts/filter-schema")
    def artifact_filter_schema() -> dict[str, Any]: return orchestrator.artifact_filter_schema()

    @app.post("/v1/artifacts/catalog/query")
    def query_artifact_catalog(body: ArtifactCatalogQueryRequest) -> dict[str, Any]:
        try: return orchestrator.artifact_catalog_query(**body.model_dump())
        except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/artifacts/catalog:reindex")
    def reindex_artifact_catalog(body: ArtifactCatalogReindexRequest | None = None) -> dict[str, Any]:
        # This route only refreshes database-owned catalog facts.  It accepts
        # IDs, never paths, so it cannot be used to inspect arbitrary files.
        return orchestrator.reindex_artifact_catalog(artifact_ids=body.artifact_ids if body else None)

    @app.post("/v1/artifacts/catalog:reindex/schedule", status_code=202)
    async def schedule_reindex_artifact_catalog(body: ArtifactCatalogReindexRequest | None = None) -> dict[str, Any]:
        return {"operation": await run_in_threadpool(
            orchestrator.create_operation, "artifact_reindex", {"artifact_ids": body.artifact_ids if body else None}
        )}

    @app.get("/v1/tags")
    def tags() -> dict[str, Any]: return {"tags": orchestrator.tags()}

    @app.get("/v1/artifact-tags")
    def artifact_tags_catalog() -> dict[str, Any]: return {"tags": orchestrator.tags()}

    @app.post("/v1/tags", status_code=201)
    def create_tag(body: TagRequest) -> dict[str, Any]:
        try: return orchestrator.create_tag(**body.model_dump())
        except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/artifact-tags/assign")
    def assign_artifact_tags(body: ArtifactTagAssignmentRequest) -> dict[str, Any]:
        if not body.artifact_ids: raise HTTPException(status_code=422, detail="Select at least one artifact")
        try:
            assignments = {artifact_id: orchestrator.set_artifact_tags(artifact_id, body.tags) for artifact_id in dict.fromkeys(body.artifact_ids)}
            return {"assignments": assignments}
        except KeyError as exc: raise HTTPException(status_code=404, detail="Artifact was not found") from exc
        except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/artifacts/{artifact_id}:download")
    async def download_artifact(artifact_id: str) -> StreamingResponse:
        """Download a portable, complete model artifact without exposing its path."""
        try:
            filename, stream = await run_in_threadpool(orchestrator.artifact_download_archive, artifact_id)
            return StreamingResponse(
                stream, media_type="application/gzip",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail="Artifact was not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/artifacts/{artifact_id}")
    def artifact(artifact_id: str) -> dict[str, Any]: return required(orchestrator.artifact(artifact_id), "Artifact")

    @app.get("/v1/artifacts/{artifact_id}/detail")
    def artifact_detail(artifact_id: str) -> dict[str, Any]:
        try: return orchestrator.artifact_detail(artifact_id)
        except KeyError as exc: raise HTTPException(status_code=404, detail="Artifact was not found") from exc

    @app.get("/v1/artifacts/{artifact_id}/architecture-view")
    def artifact_architecture_view(artifact_id: str) -> dict[str, Any]:
        try: return orchestrator.artifact_architecture_view(artifact_id)
        except KeyError as exc: raise HTTPException(status_code=404, detail="Artifact was not found") from exc

    @app.get("/v1/artifacts/{artifact_id}/tags")
    def artifact_tags(artifact_id: str) -> dict[str, Any]:
        if orchestrator.artifact(artifact_id) is None: raise HTTPException(status_code=404, detail="Artifact was not found")
        return {"tags": orchestrator.tags_for_artifact(artifact_id)}

    @app.put("/v1/artifacts/{artifact_id}/tags")
    def set_artifact_tags(artifact_id: str, body: ArtifactTagRequest) -> dict[str, Any]:
        try: return {"tags": orchestrator.set_artifact_tags(artifact_id, body.tags)}
        except KeyError as exc: raise HTTPException(status_code=404, detail="Artifact was not found") from exc
        except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/model-drafts")
    def model_drafts() -> dict[str, Any]: return {"drafts": orchestrator.model_drafts()}

    @app.post("/v1/model-drafts", status_code=201)
    def create_model_draft(body: ModelDraftRequest) -> dict[str, Any]:
        try: return orchestrator.create_model_draft(**body.model_dump())
        except KeyError as exc: raise HTTPException(status_code=404, detail="Source artifact was not found") from exc
        except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/model-drafts/{draft_id}:clone", status_code=201)
    def clone_model_draft(draft_id: str, body: ModelDraftCloneRequest) -> dict[str, Any]:
        try: return orchestrator.clone_model_draft(draft_id, **body.model_dump())
        except KeyError as exc: raise HTTPException(status_code=404, detail="Model draft was not found") from exc

    @app.get("/v1/model-drafts/{draft_id}:validate")
    def validate_model_draft(draft_id: str) -> dict[str, Any]:
        try: return orchestrator.validate_model_draft(draft_id)
        except KeyError as exc: raise HTTPException(status_code=404, detail="Model draft was not found") from exc

    @app.get("/v1/model-drafts/{draft_id}:preview")
    def preview_model_draft(draft_id: str, dataset_id: str | None = Query(default=None)) -> dict[str, Any]:
        try: return orchestrator.preview_model_draft(draft_id, dataset_id=dataset_id)
        except KeyError as exc: raise HTTPException(status_code=404, detail="Model draft or dataset was not found") from exc
        except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/model-drafts/{draft_id}:plan-training", status_code=201)
    def plan_draft_training(draft_id: str, body: DraftTrainingPlanRequest) -> dict[str, Any]:
        try: return orchestrator.plan_draft_training(draft_id, **body.model_dump())
        except KeyError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    # V2 model definitions replace drafts for new authoring workflows.  They
    # are complete, versioned configurations whose revisions can be pinned by
    # the queue without exposing mutable client state.
    @app.get("/v1/model-definition-templates")
    def model_definition_templates() -> dict[str, Any]:
        return {"templates": orchestrator.model_definition_templates()}

    @app.get("/v1/model-definitions")
    def model_definitions() -> dict[str, Any]:
        return {"definitions": orchestrator.model_definitions()}

    @app.post("/v1/model-definitions", status_code=201)
    def create_model_definition(body: ModelDefinitionRequest) -> dict[str, Any]:
        try: return orchestrator.create_model_definition(**body.model_dump())
        except KeyError as exc: raise HTTPException(status_code=404, detail="Model-definition template was not found") from exc
        except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/model-definitions/{definition_id}:duplicate", status_code=201)
    def duplicate_model_definition(definition_id: str, body: ModelDefinitionDuplicateRequest) -> dict[str, Any]:
        try: return orchestrator.duplicate_model_definition(definition_id, **body.model_dump())
        except KeyError as exc: raise HTTPException(status_code=404, detail="Model definition or revision was not found") from exc
        except ValueError as exc: raise HTTPException(status_code=409 if "conflict" in str(exc).lower() else 422, detail=str(exc)) from exc

    @app.get("/v1/model-definitions/{definition_id}/revisions")
    def model_definition_revisions(definition_id: str) -> dict[str, Any]:
        try: return {"revisions": orchestrator.model_definition_revisions(definition_id)}
        except KeyError as exc: raise HTTPException(status_code=404, detail="Model definition was not found") from exc

    @app.get("/v1/model-definitions/{definition_id}")
    def model_definition(definition_id: str, revision: int | None = Query(default=None, ge=1)) -> dict[str, Any]:
        return required(orchestrator.model_definition(definition_id, revision=revision), "Model definition")

    @app.patch("/v1/model-definitions/{definition_id}")
    def update_model_definition(definition_id: str, body: ModelDefinitionUpdateRequest) -> dict[str, Any]:
        try: return orchestrator.update_model_definition(definition_id, **body.model_dump(exclude_unset=True))
        except KeyError as exc: raise HTTPException(status_code=404, detail="Model definition was not found") from exc
        except ValueError as exc: raise HTTPException(status_code=409 if "conflict" in str(exc).lower() else 422, detail=str(exc)) from exc

    @app.delete("/v1/model-definitions/{definition_id}")
    def delete_model_definition(definition_id: str) -> dict[str, Any]:
        try: return orchestrator.delete_model_definition(definition_id)
        except KeyError as exc: raise HTTPException(status_code=404, detail="Model definition was not found") from exc
        except ValueError as exc: raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/model-definitions/{definition_id}:queue", status_code=201)
    @app.post("/v1/model-definitions/{definition_id}:validate-and-queue", status_code=201)
    def validate_and_queue_model_definition(definition_id: str, body: ValidatedQueueRunRequest) -> dict[str, Any]:
        try: return orchestrator.queue_model_definition_for_pool(definition_id, **body.model_dump())
        except KeyError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, ValueError, RuntimeError) as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/model-definitions/{definition_id}:validate-and-queue/schedule", status_code=202)
    async def schedule_validated_queue_model_definition(definition_id: str, body: ValidatedQueueRunRequest) -> dict[str, Any]:
        # This returns immediately.  Any automatic batch calibration runs on
        # the worker that eventually claims the sealed training work unit.
        parameters = {"definition_id": definition_id, **body.model_dump()}
        try: return {"operation": await run_in_threadpool(orchestrator.create_operation, "validated_queue_validation", parameters)}
        except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/queued-runs")
    def queued_runs() -> dict[str, Any]:
        return {"queued_runs": orchestrator.queued_runs()}

    @app.post("/v1/queued-runs:verify", status_code=202)
    def verify_queued_runs(body: QueueVerifyRequest) -> dict[str, Any]:
        try: return orchestrator.verify_queued_runs_for_pool(**body.model_dump())
        except KeyError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/queued-runs:start")
    def start_queued_runs(body: QueueStartRequest) -> dict[str, Any]:
        try: return orchestrator.authorize_queued_runs_for_pool(**body.model_dump())
        except KeyError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, RuntimeError) as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/queued-runs/{queued_run_id}:cancel")
    def cancel_queued_run(queued_run_id: str) -> dict[str, Any]:
        try: return orchestrator.cancel_queued_run(queued_run_id)
        except KeyError as exc: raise HTTPException(status_code=404, detail="Queued run was not found") from exc
        except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/queued-runs/{queued_run_id}:clear")
    def clear_terminal_queued_run(queued_run_id: str) -> dict[str, Any]:
        try: return orchestrator.archive_terminal_queued_run(queued_run_id)
        except KeyError as exc: raise HTTPException(status_code=404, detail="Queued run was not found") from exc
        except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/queued-runs:clear")
    def clear_queued_runs(body: QueueMaintenanceRequest) -> dict[str, Any]:
        try: return orchestrator.clear_queued_runs(**body.model_dump())
        except KeyError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/v1/inference-runs")
    def inference_runs(limit: int = Query(default=100, ge=1, le=500)) -> dict[str, Any]:
        return {"inference_runs": orchestrator.inference_runs(limit=limit)}

    @app.post("/v1/inference-runs", status_code=201)
    def create_inference_run(body: InferenceRunRequest) -> dict[str, Any]:
        try: return {"inference_run": orchestrator.create_inference_run(**body.model_dump())}
        except KeyError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, ValueError) as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/inference-runs/{inference_run_id}:start")
    def start_inference_run(inference_run_id: str) -> dict[str, Any]:
        try: return {"inference_run": orchestrator.start_inference_run(inference_run_id)}
        except KeyError as exc: raise HTTPException(status_code=404, detail="Inference run was not found") from exc
        except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/inference-runs/{inference_run_id}:cancel")
    def cancel_inference_run(inference_run_id: str) -> dict[str, Any]:
        try: return {"inference_run": orchestrator.cancel_inference_run(inference_run_id)}
        except KeyError as exc: raise HTTPException(status_code=404, detail="Inference run was not found") from exc
        except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/inference-runs/{inference_run_id}:download")
    async def download_inference_result(inference_run_id: str) -> StreamingResponse:
        try:
            filename, stream = await run_in_threadpool(orchestrator.inference_result_download_archive, inference_run_id)
            return StreamingResponse(stream, media_type="application/gzip", headers={"Content-Disposition": f'attachment; filename="{filename}"'})
        except (KeyError, FileNotFoundError) as exc: raise HTTPException(status_code=404, detail="Inference result was not found") from exc
        except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/inference-runs/{inference_run_id}")
    def inference_run(inference_run_id: str) -> dict[str, Any]:
        return {"inference_run": required(orchestrator.inference_run(inference_run_id), "Inference run")}

    # Keep the bare parameter route after action routes: Starlette treats
    # ``id:validate`` as a valid path parameter otherwise.
    @app.get("/v1/model-drafts/{draft_id}")
    def model_draft(draft_id: str) -> dict[str, Any]: return required(orchestrator.model_draft(draft_id), "Model draft")

    @app.patch("/v1/model-drafts/{draft_id}")
    def update_model_draft(draft_id: str, body: ModelDraftUpdateRequest) -> dict[str, Any]:
        try: return orchestrator.update_model_draft(draft_id, **body.model_dump(exclude_unset=True))
        except KeyError as exc: raise HTTPException(status_code=404, detail="Model draft was not found") from exc
        except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/artifacts/{artifact_id}/history")
    def artifact_history(artifact_id: str, limit: int = Query(default=500, ge=1, le=2000)) -> dict[str, Any]:
        try: return orchestrator.artifact_history(artifact_id, limit=limit)
        except KeyError as exc: raise HTTPException(status_code=404, detail="Artifact was not found") from exc

    @app.get("/v1/artifacts/{artifact_id}/evidence")
    def artifact_evidence(artifact_id: str) -> dict[str, Any]:
        try: return orchestrator.artifact_evidence(artifact_id)
        except KeyError as exc: raise HTTPException(status_code=404, detail="Artifact was not found") from exc

    @app.get("/v1/artifacts/{artifact_id}/evidence/files/{relative_path:path}")
    def artifact_evidence_file(artifact_id: str, relative_path: str) -> FileResponse:
        try: return FileResponse(orchestrator.artifact_evidence_file(artifact_id, relative_path))
        except (KeyError, FileNotFoundError) as exc: raise HTTPException(status_code=404, detail="Evidence file was not found") from exc
        except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/recipes")
    def recipes() -> dict[str, Any]: return {"recipes": orchestrator.recipes()}

    @app.post("/v1/recipes", status_code=201)
    def create_recipe(body: RecipeRequest) -> dict[str, Any]:
        try: return orchestrator.create_recipe(**body.model_dump())
        except (OSError, ValueError) as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/recipes/{recipe_id}")
    def recipe(recipe_id: str) -> dict[str, Any]: return required(orchestrator.recipe(recipe_id), "Recipe")

    @app.get("/v1/experiments")
    def experiments() -> dict[str, Any]: return {"experiments": orchestrator.experiments()}

    @app.post("/v1/experiments:train")
    def retired_training_experiment() -> None:
        """Explicitly reject the former recipe/specification intake lane.

        New training begins with a versioned model definition and
        ``:validate-and-queue``.  Returning Gone instead of silently accepting
        this payload keeps old automation from creating work that no worker is
        expected to execute.
        """
        raise HTTPException(
            status_code=410,
            detail="Training experiments are retired; queue a model definition through :validate-and-queue.",
        )

    @app.post("/v1/model-imports")
    def retired_model_import() -> None:
        """External path-based model import belonged to the retired push lane."""
        raise HTTPException(
            status_code=410,
            detail="Model import is retired; upload or retrieve sealed artifacts through the artifact catalog.",
        )

    @app.get("/v1/experiments/{experiment_id}")
    def experiment(experiment_id: str) -> dict[str, Any]: return required(orchestrator.experiment(experiment_id), "Experiment")

    @app.get("/v1/experiments/{experiment_id}/results")
    def experiment_results(experiment_id: str) -> dict[str, Any]:
        try: return orchestrator.experiment_results(experiment_id)
        except KeyError as exc: raise HTTPException(status_code=404, detail="Experiment was not found") from exc

    @app.get("/v1/comparisons")
    def comparisons() -> dict[str, Any]: return {"comparisons": orchestrator.comparisons()}

    @app.post("/v1/comparisons", status_code=201)
    def create_comparison(body: ComparisonRequest) -> dict[str, Any]:
        try: return orchestrator.create_comparison(**body.model_dump())
        except KeyError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/comparisons/{comparison_id}")
    def comparison(comparison_id: str) -> dict[str, Any]: return required(orchestrator.comparison(comparison_id), "Comparison")

    @app.get("/v1/comparison-groups")
    def comparison_groups() -> dict[str, Any]: return {"comparison_groups": orchestrator.comparison_groups()}

    @app.post("/v1/comparison-groups", status_code=201)
    def create_comparison_group(body: ComparisonGroupRequest) -> dict[str, Any]:
        try: return orchestrator.create_comparison_group(**body.model_dump())
        except KeyError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/comparison-groups/{comparison_group_id}")
    def comparison_group(comparison_group_id: str) -> dict[str, Any]:
        return required(orchestrator.comparison_group(comparison_group_id), "Comparison group")


    @app.get("/v1/specifications")
    def specifications(experiment_id: str | None = Query(default=None)) -> dict[str, Any]:
        return {"specifications": orchestrator.specifications(experiment_id)}

    @app.patch("/v1/specifications/{specification_id}")
    def update_planned_specification(specification_id: str, body: PlannedRunUpdateRequest) -> dict[str, Any]:
        try: return orchestrator.update_planned_specification(specification_id, name=body.name, resources=body.resources, config_overrides=body.config_overrides)
        except KeyError as exc: raise HTTPException(status_code=404, detail="Run specification was not found") from exc
        except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/specifications/{specification_id}")
    def specification_detail(specification_id: str) -> dict[str, Any]:
        try: return orchestrator.specification_detail(specification_id)
        except KeyError as exc: raise HTTPException(status_code=404, detail="Run specification was not found") from exc
        except (OSError, ValueError) as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/specifications/{specification_id}:enqueue", status_code=201)
    def enqueue_for_worker_pool(specification_id: str, body: WorkerPoolEnqueueRequest) -> dict[str, Any]:
        try:
            return orchestrator.enqueue_specification_for_pool(specification_id, body.worker_pool_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run specification or worker pool was not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/jobs")
    def jobs(refresh: bool = Query(default=False)) -> dict[str, Any]:
        if refresh:
            raise HTTPException(
                status_code=410,
                detail="Legacy push-job refresh is retired; inspect pull-worker leases and job events instead.",
            )
        return {"jobs": orchestrator.jobs()}

    @app.get("/v1/jobs/{job_id}")
    def job(job_id: str) -> dict[str, Any]: return required(orchestrator.job(job_id), "Job")

    @app.get("/v1/jobs/{job_id}/events")
    def job_events(job_id: str) -> dict[str, Any]:
        required(orchestrator.job(job_id), "Job")
        return {"events": orchestrator.job_events(job_id)}

    @app.get("/v1/jobs/{job_id}/timing")
    def job_timing(job_id: str) -> dict[str, Any]:
        try:
            return orchestrator.job_timing(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job was not found") from exc

    @app.post("/v1/jobs/{job_id}:cancel")
    def cancel_pull_job(job_id: str) -> dict[str, Any]:
        try:
            return orchestrator.request_pull_job_cancellation(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job was not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app
