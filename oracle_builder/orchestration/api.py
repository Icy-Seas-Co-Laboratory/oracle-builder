from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
import uuid

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from oracle_builder.orchestration.service import Orchestrator


class DatasetIngestRequest(BaseModel):
    path: str


class ScanRequest(BaseModel):
    root: str


class DispatchRequest(BaseModel):
    endpoint_id: str


class RecipeRequest(BaseModel):
    name: str
    config_path: str
    description: str = ""


class TrainingExperimentRequest(BaseModel):
    name: str
    dataset_id: str
    recipe_ids: list[str]
    seeds: list[int]
    description: str = ""
    resources: dict[str, Any] = Field(default_factory=dict)
    config_overrides: dict[str, Any] = Field(default_factory=dict)


class PlannedRunUpdateRequest(BaseModel):
    name: str | None = None
    resources: dict[str, Any] = Field(default_factory=dict)
    config_overrides: dict[str, Any] = Field(default_factory=dict)


class ModelPreviewRequest(BaseModel):
    architecture: str
    dataset_id: str
    overrides: dict[str, Any] = Field(default_factory=dict)


class ModelImportRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    name: str
    model_path: str
    info_path: str
    dataset_id: str | None = None
    description: str = ""
    resources: dict[str, Any] = Field(default_factory=dict)


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
    name: str
    dataset_id: str
    endpoint_id: str
    revision: int | None = Field(default=None, ge=1)
    description: str = ""
    resources: dict[str, Any] = Field(default_factory=dict)
    initialization: dict[str, Any] = Field(default_factory=dict)
    batch_size_mode: str = "manual"
    batch_size: int | None = Field(default=None, ge=1)
    maximum_batch_size: int = Field(default=256, ge=1)


class QueueStartRequest(BaseModel):
    endpoint_id: str
    queued_run_ids: list[str] = Field(default_factory=list)
    all_ready: bool = False


class QueueMaintenanceRequest(BaseModel):
    endpoint_id: str | None = None


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


def create_app(orchestrator: Orchestrator) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.startup_reconciliation = orchestrator.reconcile_startup()
        yield

    app = FastAPI(title="Oracle Builder Orchestrator API", version="0.1.0", lifespan=lifespan)
    app.state.orchestrator = orchestrator

    def required(value: dict[str, Any] | None, label: str) -> dict[str, Any]:
        if value is None:
            raise HTTPException(status_code=404, detail=f"{label} was not found")
        return value

    @app.get("/health/live")
    def live() -> dict[str, str]: return {"status": "ok"}

    @app.get("/health/ready")
    def ready() -> dict[str, Any]:
        endpoints = orchestrator.compute_endpoints()
        available = sum(endpoint["status"] == "ready" for endpoint in endpoints)
        return {"status": "ready" if available else "degraded", "database": "ready", "compute_endpoints": {"configured": len(endpoints), "ready": available}, "startup_reconciliation": getattr(app.state, "startup_reconciliation", None)}

    @app.get("/v1/compute/endpoints")
    def compute_endpoints(refresh: bool = Query(default=False)) -> dict[str, Any]:
        return {"endpoints": orchestrator.compute_endpoints(refresh=refresh)}

    @app.get("/v1/files/roots")
    def file_roots() -> dict[str, Any]: return {"roots": orchestrator.file_roots()}

    @app.get("/v1/files")
    def files(root: str = Query(default="workspace"), path: str = Query(default="")) -> dict[str, Any]:
        try: return orchestrator.files(root, path or ".")
        except KeyError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, ValueError) as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

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
                        handle.write(chunk)
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

    @app.get("/v1/artifacts")
    def artifacts() -> dict[str, Any]: return {"artifacts": orchestrator.artifacts()}

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

    @app.post("/v1/model-definitions/{definition_id}:validate-and-queue", status_code=201)
    def validate_and_queue_model_definition(definition_id: str, body: ValidatedQueueRunRequest) -> dict[str, Any]:
        try: return orchestrator.validate_and_queue_model_definition(definition_id, **body.model_dump())
        except KeyError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, ValueError, RuntimeError) as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/queued-runs")
    def queued_runs() -> dict[str, Any]:
        return {"queued_runs": orchestrator.queued_runs()}

    @app.post("/v1/queued-runs:start")
    def start_queued_runs(body: QueueStartRequest) -> dict[str, Any]:
        try: return orchestrator.authorize_queued_runs(**body.model_dump())
        except KeyError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, RuntimeError) as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/queued-runs/{queued_run_id}:cancel")
    def cancel_queued_run(queued_run_id: str) -> dict[str, Any]:
        try: return orchestrator.cancel_queued_run(queued_run_id)
        except KeyError as exc: raise HTTPException(status_code=404, detail="Queued run was not found") from exc
        except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/queued-runs:clear")
    def clear_queued_runs(body: QueueMaintenanceRequest) -> dict[str, Any]:
        try: return orchestrator.clear_queued_runs(**body.model_dump())
        except KeyError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/v1/jobs:reset-stuck")
    def reset_stuck_jobs(body: QueueMaintenanceRequest) -> dict[str, Any]:
        try: return orchestrator.reset_stuck_jobs(**body.model_dump())
        except KeyError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc

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

    @app.post("/v1/experiments:train", status_code=201)
    def create_training_experiment(body: TrainingExperimentRequest) -> dict[str, Any]:
        try: return orchestrator.create_training_experiment(**body.model_dump())
        except KeyError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, ValueError) as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/model-imports", status_code=201)
    def create_model_import(body: ModelImportRequest) -> dict[str, Any]:
        try: return orchestrator.create_model_import(**body.model_dump())
        except KeyError as exc: raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (OSError, ValueError) as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

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

    @app.get("/v1/specifications/{specification_id}/preflight")
    def preflight(specification_id: str, endpoint_id: str = Query()) -> dict[str, Any]:
        try: return orchestrator.preflight(specification_id, endpoint_id)
        except KeyError as exc: raise HTTPException(status_code=404, detail="Run specification or compute endpoint was not found") from exc

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

    @app.post("/v1/specifications/{specification_id}:dispatch", status_code=202)
    def dispatch(specification_id: str, body: DispatchRequest) -> dict[str, Any]:
        try: return orchestrator.dispatch(specification_id, body.endpoint_id)
        except KeyError as exc: raise HTTPException(status_code=404, detail="Run specification was not found") from exc
        except (ValueError, RuntimeError) as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/jobs")
    def jobs(refresh: bool = Query(default=False)) -> dict[str, Any]:
        if refresh:
            orchestrator.reconcile_active_jobs()
        return {"jobs": orchestrator.jobs()}

    @app.get("/v1/jobs/{job_id}")
    def job(job_id: str) -> dict[str, Any]: return required(orchestrator.job(job_id), "Job")

    @app.get("/v1/jobs/{job_id}/events")
    def job_events(job_id: str) -> dict[str, Any]:
        required(orchestrator.job(job_id), "Job")
        return {"events": orchestrator.job_events(job_id)}

    @app.get("/v1/jobs/{job_id}/training-status")
    def job_training_status(job_id: str) -> dict[str, Any]:
        try:
            return orchestrator.job_training_status(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job was not found") from exc

    @app.post("/v1/jobs/{job_id}:reconcile")
    def reconcile(job_id: str) -> dict[str, Any]:
        try: return orchestrator.reconcile_job(job_id)
        except KeyError as exc: raise HTTPException(status_code=404, detail="Job was not found") from exc
        except RuntimeError as exc: raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/v1/jobs/{job_id}:pause")
    def pause_job(job_id: str) -> dict[str, Any]:
        try: return orchestrator.control_job(job_id, "pause")
        except KeyError as exc: raise HTTPException(status_code=404, detail="Job was not found") from exc
        except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc: raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/v1/jobs/{job_id}:resume")
    def resume_job(job_id: str) -> dict[str, Any]:
        try: return orchestrator.control_job(job_id, "resume")
        except KeyError as exc: raise HTTPException(status_code=404, detail="Job was not found") from exc
        except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc: raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/v1/jobs/{job_id}:cancel")
    def cancel_job(job_id: str) -> dict[str, Any]:
        try: return orchestrator.control_job(job_id, "cancel")
        except KeyError as exc: raise HTTPException(status_code=404, detail="Job was not found") from exc
        except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc: raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app
