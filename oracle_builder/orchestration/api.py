from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
import uuid

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
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


def create_app(orchestrator: Orchestrator) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
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
        return {"status": "ready" if available else "degraded", "database": "ready", "compute_endpoints": {"configured": len(endpoints), "ready": available}}

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

    @app.post("/v1/datasets:ingest", status_code=201)
    def ingest_dataset(body: DatasetIngestRequest) -> dict[str, Any]:
        try: return orchestrator.ingest_dataset(body.path)
        except (OSError, ValueError) as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/datasets/{dataset_id}")
    def dataset(dataset_id: str) -> dict[str, Any]: return required(orchestrator.dataset(dataset_id), "Dataset")

    @app.post("/v1/catalog:scan")
    def scan(body: ScanRequest) -> dict[str, Any]:
        try: return orchestrator.scan(body.root)
        except (OSError, ValueError) as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/artifacts")
    def artifacts() -> dict[str, Any]: return {"artifacts": orchestrator.artifacts()}

    @app.get("/v1/artifacts/{artifact_id}")
    def artifact(artifact_id: str) -> dict[str, Any]: return required(orchestrator.artifact(artifact_id), "Artifact")

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


    @app.get("/v1/specifications")
    def specifications(experiment_id: str | None = Query(default=None)) -> dict[str, Any]:
        return {"specifications": orchestrator.specifications(experiment_id)}

    @app.get("/v1/specifications/{specification_id}/preflight")
    def preflight(specification_id: str, endpoint_id: str = Query()) -> dict[str, Any]:
        try: return orchestrator.preflight(specification_id, endpoint_id)
        except KeyError as exc: raise HTTPException(status_code=404, detail="Run specification or compute endpoint was not found") from exc

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

    @app.post("/v1/jobs/{job_id}:reconcile")
    def reconcile(job_id: str) -> dict[str, Any]:
        try: return orchestrator.reconcile_job(job_id)
        except KeyError as exc: raise HTTPException(status_code=404, detail="Job was not found") from exc
        except RuntimeError as exc: raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app
