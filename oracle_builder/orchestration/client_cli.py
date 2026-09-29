"""Thin, transport-only command line client for the Oracle Orchestrator.

This module deliberately contains no training, inference, artifact sealing, or
output-path logic.  Those decisions belong to the control plane; the only
local file access supported here is streaming an explicitly selected source
file to its owned upload endpoint.
"""
from __future__ import annotations

import argparse
import uuid
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, BinaryIO, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from oracle_data_contracts.artifacts import ArtifactRef


DEFAULT_URL = "http://127.0.0.1:8110"


class OrchestratorClient:
    """Small stdlib HTTP client, kept separate to make the CLI testable."""

    def __init__(self, base_url: str, *, timeout: float = 30.0, token: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.token = token

    def request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | bytes | BinaryIO | None = None,
        query: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        if query:
            query_string = urlencode({key: value for key, value in query.items() if value is not None}, doseq=True)
            path = f"{path}?{query_string}"
        data: bytes | BinaryIO | None
        request_headers = {"Accept": "application/json", **(headers or {})}
        if self.token:
            request_headers.setdefault("Authorization", f"Bearer {self.token}")
        if isinstance(body, dict):
            data = json.dumps(body).encode("utf-8")
            request_headers["Content-Type"] = "application/json"
        else:
            data = body
            if body is not None:
                request_headers.setdefault("Content-Type", "application/octet-stream")
        request = Request(f"{self.base_url}{path}", data=data, headers=request_headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout) as response:  # nosec B310 -- explicit operator URL
                raw = response.read()
                return json.loads(raw) if raw else None
        except HTTPError as exc:
            payload = exc.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(payload).get("detail", payload)
            except json.JSONDecodeError:
                detail = payload
            raise RuntimeError(f"Orchestrator returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach Orchestrator at {self.base_url}: {exc.reason}") from exc

    def upload(self, kind: str, source: Path) -> Any:
        # The server, not the client, chooses the durable staging directory.
        with source.open("rb") as handle:
            return self.request(
                "POST", f"/v1/uploads/{quote(kind)}/{quote(source.name)}", body=handle,
                headers={"Content-Length": str(source.stat().st_size)},
            )

    def download(self, path: str, destination: Path) -> dict[str, Any]:
        """Stream a server-owned artifact to an explicit local destination."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        headers = {"Accept": "application/octet-stream"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = Request(f"{self.base_url}{path}", headers=headers, method="GET")
        try:
            with urlopen(request, timeout=self.timeout) as response, destination.open("wb") as handle:  # nosec B310 -- explicit operator URL
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Orchestrator returned HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach Orchestrator at {self.base_url}: {exc.reason}") from exc
        return {"path": str(destination), "bytes": destination.stat().st_size}


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oracle", description="Control Oracle Builder through the Orchestrator API.")
    parser.add_argument("--url", default=os.environ.get("ORACLE_ORCHESTRATOR_URL", DEFAULT_URL), help="Orchestrator base URL (or ORACLE_ORCHESTRATOR_URL).")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds.")
    parser.add_argument("--token", default=os.environ.get("ORACLE_ORCHESTRATOR_TOKEN"),
                        help="Operator API token (or ORACLE_ORCHESTRATOR_TOKEN).")
    commands = parser.add_subparsers(dest="area", required=True)

    commands.add_parser("health", help="Read control-plane readiness.")

    dataset = commands.add_parser("dataset", help="Manage server-owned dataset records.")
    ds = dataset.add_subparsers(dest="command", required=True)
    ds.add_parser("list")
    ingest = ds.add_parser("ingest", help="Register an already staged, server-visible dataset path.")
    ingest.add_argument("path", help="Path previously returned by `oracle dataset upload` or approved by the server.")
    upload = ds.add_parser("upload", help="Upload a local SQLite dataset into Orchestrator-owned staging.")
    upload.add_argument("source", type=Path)
    upload.add_argument("--ingest", action="store_true", help="Register the upload immediately after the server accepts it.")

    recipe = commands.add_parser("recipe", help="Manage recipe references held by the Orchestrator.")
    rs = recipe.add_subparsers(dest="command", required=True)
    rs.add_parser("list")
    recipe_upload = rs.add_parser("upload", help="Upload a local TOML recipe into Orchestrator-owned staging.")
    recipe_upload.add_argument("source", type=Path)
    recipe_upload.add_argument("--create", action="store_true", help="Register the server-returned config reference immediately.")
    recipe_upload.add_argument("--name", help="Required with --create.")
    recipe_upload.add_argument("--description", default="")
    create_recipe = rs.add_parser("create")
    create_recipe.add_argument("--name", required=True)
    create_recipe.add_argument("--config-path", required=True, help="Approved server-visible config path.")
    create_recipe.add_argument("--description", default="")

    definition = commands.add_parser("definition", help="Inspect definitions and seal portable training work.")
    ds = definition.add_subparsers(dest="command", required=True)
    ds.add_parser("list")
    queue = ds.add_parser("queue", help="Add a definition to the queue without verifying or starting training.")
    queue.add_argument("definition_id")
    queue.add_argument("--name", required=True)
    queue.add_argument("--dataset", required=True, dest="dataset_id")
    queue.add_argument("--worker-pool", required=True, dest="worker_pool_id")
    queue.add_argument("--revision", type=_positive)
    queue.add_argument("--description", default="")
    queue.add_argument("--gpu-count", type=int, default=0)
    queue.add_argument("--batch-size", type=_positive)
    queue.add_argument("--auto-batch-size", action="store_true")
    queue.add_argument("--maximum-batch-size", type=_positive, default=256)
    queue.add_argument("--epochs", type=_positive, default=10)

    queue_control = commands.add_parser("queue", help="Verify queued runs, then explicitly start training.")
    qs = queue_control.add_subparsers(dest="command", required=True)
    qs.add_parser("list")
    verify = qs.add_parser("verify")
    verify.add_argument("queued_run_ids", nargs="+")
    verify.add_argument("--start", action="store_true", help="Start each run only after successful verification.")
    start = qs.add_parser("start")
    start.add_argument("queued_run_ids", nargs="+")
    start.add_argument("--worker-pool", required=True, dest="worker_pool_id")

    specification = commands.add_parser("spec", help="Inspect and queue immutable specifications for worker pools.")
    ss = specification.add_subparsers(dest="command", required=True)
    spec_list = ss.add_parser("list")
    spec_list.add_argument("--experiment")
    for name in ("show", "enqueue"):
        command = ss.add_parser(name)
        command.add_argument("specification_id")
        if name == "enqueue":
            command.add_argument("--worker-pool", required=True, dest="worker_pool_id")

    job = commands.add_parser("job", help="Inspect and control jobs through the Orchestrator.")
    js = job.add_subparsers(dest="command", required=True)
    js.add_parser("list")
    for name in ("show", "events"):
        command = js.add_parser(name)
        command.add_argument("job_id")
    command = js.add_parser("commands", help="Inspect durable worker-control receipts.")
    command.add_argument("job_id")
    command = js.add_parser("command", help="Request durable worker control.")
    command.add_argument("job_id")
    command.add_argument("action", choices=("pause", "yield", "stop_now", "restart", "resume"))
    command.add_argument("--reason")

    execution = commands.add_parser("execution", help="Inspect a resumable execution run.")
    es = execution.add_subparsers(dest="command", required=True)
    command = es.add_parser("show")
    command.add_argument("run_id")

    artifact = commands.add_parser("artifact", help="Inspect cataloged artifacts; artifact bytes remain server-owned.")
    ars = artifact.add_subparsers(dest="command", required=True)
    ars.add_parser("list")
    artifact_show = ars.add_parser("show")
    artifact_show.add_argument("artifact_id")
    artifact_replicas = ars.add_parser("replicas", help="Inspect durable S3-compatible replica state.")
    artifact_replicas.add_argument("--status", choices=("pending", "replicated", "failed"))
    for name in ("replica-retry", "replica-verify", "replica-restore"):
        command = ars.add_parser(name, help="Operate replicas by location-free ArtifactRef.")
        command.add_argument("ref", nargs="+", type=ArtifactRef.parse)

    worker = commands.add_parser("worker", help="Operate registered pull-worker pools.")
    ws = worker.add_subparsers(dest="command", required=True)
    for name, path in (("pools", None), ("list", None), ("leases", None)):
        ws.add_parser(name)
    pool_create = ws.add_parser("pool-create")
    pool_create.add_argument("--name", required=True)
    pool_create.add_argument("--action", required=True, action="append", dest="allowed_actions")
    pool_create.add_argument("--max-workers", type=_positive)
    deployment = commands.add_parser("deployment", help="Operate fixed-profile worker deployments.")
    dps = deployment.add_subparsers(dest="command", required=True)
    dps.add_parser("list")
    deployment_create = dps.add_parser("create")
    deployment_create.add_argument("--name", required=True)
    deployment_create.add_argument("--worker-pool", required=True, dest="pool_id")
    deployment_create.add_argument("--profile", required=True, dest="profile_id", help="Operator-configured deployment profile ID.")
    for name in ("show", "start", "stop"):
        command = dps.add_parser(name)
        command.add_argument("deployment_id")
        if name == "stop": command.add_argument("--timeout-seconds", type=float, default=10.0)
    dps.add_parser("reconcile")

    inference = commands.add_parser("inference", help="Submit and inspect durable batch inference runs.")
    ins = inference.add_subparsers(dest="command", required=True)
    ins.add_parser("list")
    inference_submit = ins.add_parser("submit", help="Create an immutable inference request.")
    inference_submit.add_argument("--name", required=True)
    inference_submit.add_argument("--model", required=True, dest="model_artifact_id")
    inference_submit.add_argument("--dataset", required=True, dest="dataset_id")
    inference_submit.add_argument("--worker-pool", required=True, dest="worker_pool_id")
    inference_submit.add_argument("--split", choices=("all", "train", "validation", "test"), default="all")
    inference_submit.add_argument("--prediction-set", help="Optional versioned prediction-set name.")
    inference_submit.add_argument("--resources", default="{}", help="JSON object of bounded resource requests.")
    inference_submit.add_argument("--start", action="store_true", help="Authorize the new run immediately.")
    inference_submit.add_argument("--wait", action="store_true", help="Start and wait for a terminal durable state.")
    inference_submit.add_argument("--poll-seconds", type=float, default=2.0)
    for name in ("show", "start", "cancel", "download"):
        command = ins.add_parser(name)
        command.add_argument("inference_run_id")
        if name == "download": command.add_argument("--output", required=True, type=Path)
        if name == "start":
            command.add_argument("--wait", action="store_true")
            command.add_argument("--poll-seconds", type=float, default=2.0)
    return parser


def _request_for(args: argparse.Namespace, client: OrchestratorClient) -> Any:
    area, command = args.area, getattr(args, "command", None)
    if area == "health":
        return client.request("GET", "/health/ready")
    if area == "dataset":
        if command == "list": return client.request("GET", "/v1/datasets")
        if command == "ingest": return client.request("POST", "/v1/datasets:ingest", body={"path": args.path})
        if not args.source.is_file(): raise ValueError(f"Local upload source is not a file: {args.source}")
        uploaded = client.upload("datasets", args.source)
        if args.ingest:
            return {"upload": uploaded, "dataset": client.request("POST", "/v1/datasets:ingest", body={"path": uploaded["path"]})}
        return uploaded
    if area == "recipe":
        if command == "list": return client.request("GET", "/v1/recipes")
        if command == "upload":
            if not args.source.is_file(): raise ValueError(f"Local upload source is not a file: {args.source}")
            if args.create and not args.name: raise ValueError("--name is required with recipe upload --create")
            uploaded = client.upload("configs", args.source)
            if args.create:
                return {"upload": uploaded, "recipe": client.request("POST", "/v1/recipes", body={"name": args.name, "config_path": uploaded["path"], "description": args.description})}
            return uploaded
        return client.request("POST", "/v1/recipes", body={"name": args.name, "config_path": args.config_path, "description": args.description})
    if area == "definition":
        if command == "list": return client.request("GET", "/v1/model-definitions")
        if args.gpu_count < 0:
            raise ValueError("--gpu-count must be non-negative")
        body = {
            "name": args.name,
            "dataset_id": args.dataset_id,
            "worker_pool_id": args.worker_pool_id,
            "description": args.description,
            "resources": {"gpu_count": args.gpu_count},
            "epochs": args.epochs,
        }
        if args.revision is not None:
            body["revision"] = args.revision
        if args.auto_batch_size:
            if args.batch_size is not None:
                raise ValueError("--batch-size cannot be combined with --auto-batch-size")
            body.update(batch_size_mode="auto", maximum_batch_size=args.maximum_batch_size)
        elif args.batch_size is not None:
            body["batch_size"] = args.batch_size
        return client.request("POST", f"/v1/model-definitions/{quote(args.definition_id)}:queue", body=body)
    if area == "queue":
        if command == "list": return client.request("GET", "/v1/queued-runs")
        if command == "verify": return client.request("POST", "/v1/queued-runs:verify", body={"queued_run_ids": args.queued_run_ids, "start_after_verification": args.start})
        if command == "start": return client.request("POST", "/v1/queued-runs:start", body={"queued_run_ids": args.queued_run_ids, "worker_pool_id": args.worker_pool_id})
    if area == "spec":
        if command == "list": return client.request("GET", "/v1/specifications", query={"experiment_id": args.experiment})
        if command == "show": return client.request("GET", f"/v1/specifications/{quote(args.specification_id)}")
        if command == "enqueue": return client.request("POST", f"/v1/specifications/{quote(args.specification_id)}:enqueue", body={"worker_pool_id": args.worker_pool_id})
        raise AssertionError(f"Unsupported specification command: {command}")
    if area == "job":
        if command == "list": return client.request("GET", "/v1/jobs")
        if command == "commands": return client.request("GET", f"/v1/jobs/{quote(args.job_id)}/commands")
        if command == "command": return client.request("POST", f"/v1/jobs/{quote(args.job_id)}/commands", body={"action": args.action, "command_id": str(uuid.uuid4()), **({"reason": args.reason} if args.reason else {})})
        suffix = "/events" if command == "events" else ""
        return client.request("GET", f"/v1/jobs/{quote(args.job_id)}{suffix}")
    if area == "execution":
        return client.request("GET", f"/v1/execution-runs/{quote(args.run_id)}")
    if area == "artifact":
        if command == "list": return client.request("GET", "/v1/artifacts")
        if command == "show": return client.request("GET", f"/v1/artifacts/{quote(args.artifact_id)}")
        if command == "replicas": return client.request("GET", "/v1/artifact-replicas", query={"status": args.status})
        actions = {"replica-retry": "retry", "replica-verify": "verify", "replica-restore": "restore"}
        if command in actions:
            return client.request("POST", f"/v1/artifact-replicas:{actions[command]}", body={"refs": [ref.to_dict() for ref in args.ref]})
    if area == "worker":
        listings = {"pools": "/v1/worker-pools", "list": "/v1/workers", "leases": "/v1/worker-leases"}
        if command in listings: return client.request("GET", listings[command])
        if command == "pool-create": return client.request("POST", "/v1/worker-pools", body={"name": args.name, "allowed_actions": args.allowed_actions, "max_workers": args.max_workers})
    if area == "deployment":
        if command == "list": return client.request("GET", "/v1/worker-deployments")
        if command == "create": return client.request("POST", "/v1/worker-deployments", body={"name": args.name, "pool_id": args.pool_id, "profile_id": args.profile_id})
        if command == "show": return client.request("GET", f"/v1/worker-deployments/{quote(args.deployment_id)}")
        if command == "start": return client.request("POST", f"/v1/worker-deployments/{quote(args.deployment_id)}:start")
        if command == "stop": return client.request("POST", f"/v1/worker-deployments/{quote(args.deployment_id)}:stop", body={"timeout_seconds": args.timeout_seconds})
        if command == "reconcile": return client.request("POST", "/v1/worker-deployments:reconcile")
    if area == "inference":
        if command == "list": return client.request("GET", "/v1/inference-runs")
        if command == "show": return client.request("GET", f"/v1/inference-runs/{quote(args.inference_run_id)}")
        if command == "cancel": return client.request("POST", f"/v1/inference-runs/{quote(args.inference_run_id)}:cancel")
        if command == "download":
            return client.download(f"/v1/inference-runs/{quote(args.inference_run_id)}:download", args.output)
        if command == "submit":
            try:
                resources = json.loads(args.resources)
            except json.JSONDecodeError as exc:
                raise ValueError("--resources must be a JSON object") from exc
            if not isinstance(resources, dict):
                raise ValueError("--resources must be a JSON object")
            created = client.request("POST", "/v1/inference-runs", body={
                "name": args.name, "model_artifact_id": args.model_artifact_id,
                "dataset_id": args.dataset_id, "worker_pool_id": args.worker_pool_id,
                "split": args.split, "prediction_set": args.prediction_set, "resources": resources,
            })
            if not (args.start or args.wait): return created
            run_id = _inference_run_id(created)
            started = client.request("POST", f"/v1/inference-runs/{quote(run_id)}:start")
            if not args.wait: return {"inference_run": created, "start": started}
            return _wait_for_inference(client, run_id, args.poll_seconds)
        if command == "start":
            started = client.request("POST", f"/v1/inference-runs/{quote(args.inference_run_id)}:start")
            return _wait_for_inference(client, args.inference_run_id, args.poll_seconds) if args.wait else started
    raise AssertionError(f"Unsupported command: {area} {command}")


def _inference_run_id(value: Any) -> str:
    """Accept the compact and envelope response forms used by API clients."""
    if isinstance(value, dict):
        for key in ("inference_run_id", "run_id", "id"):
            if value.get(key): return str(value[key])
        for key in ("inference_run", "run"):
            nested = value.get(key)
            if isinstance(nested, dict):
                try: return _inference_run_id(nested)
                except ValueError: pass
    raise ValueError("Orchestrator response did not include an inference run id")


def _wait_for_inference(client: OrchestratorClient, run_id: str, poll_seconds: float) -> Any:
    if poll_seconds <= 0: raise ValueError("--poll-seconds must be positive")
    terminal = {"completed", "published", "failed", "cancelled", "canceled"}
    while True:
        run = client.request("GET", f"/v1/inference-runs/{quote(run_id)}")
        status = str(run.get("status", run.get("state", ""))).lower() if isinstance(run, dict) else ""
        if status in terminal: return run
        time.sleep(poll_seconds)


def main(argv: Sequence[str] | None = None, *, client_factory=OrchestratorClient) -> int:
    args = build_parser().parse_args(argv)
    try:
        try:
            client = client_factory(args.url, timeout=args.timeout, token=args.token)
        except TypeError as exc:
            # Narrow compatibility for injected pre-token test transports.
            # Production uses OrchestratorClient above, which always accepts it.
            if client_factory is OrchestratorClient:
                raise
            client = client_factory(args.url, timeout=args.timeout)
        result = _request_for(args, client)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
