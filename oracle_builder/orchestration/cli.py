from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import uvicorn

from oracle_builder.orchestration.api import create_app, role_token_digests
from oracle_builder.orchestration.service import Orchestrator
from oracle_builder.orchestration.storage import LocalArtifactStore, S3ReplicatedArtifactStore
from oracle_builder.orchestration.lifecycle import (
    DockerPullWorkerProfile,
    DockerPullWorkerProvider,
    PullWorkerDeploymentProfile,
)


def load_deployment_profiles(path: str | None) -> tuple[dict[str, PullWorkerDeploymentProfile], dict[str, object]]:
    """Load operator-owned deployment profiles without writing them to SQLite.

    The compact JSON format is intentionally a deployment file rather than an
    API document.  A Docker profile nests its image/network/mount policy under
    ``docker``; clients can later select only the top-level ``profile_id``.
    """
    if not path:
        return {}, {}
    source = Path(path).expanduser().resolve()
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read deployment profiles: {exc}") from exc
    entries = document.get("profiles") if isinstance(document, dict) else None
    if not isinstance(entries, list):
        raise ValueError("deployment profile file must contain a profiles array")
    profiles: dict[str, PullWorkerDeploymentProfile] = {}
    providers: dict[str, object] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each deployment profile must be an object")
        allowed = {"profile_id", "provider", "orchestrator_url", "credentials_root", "scratch_root", "executors", "registration_token_file", "capabilities", "docker"}
        unknown = set(entry) - allowed
        if unknown:
            raise ValueError(f"deployment profile has unsupported fields: {', '.join(sorted(unknown))}")
        try:
            profile = PullWorkerDeploymentProfile(
                profile_id=entry["profile_id"], provider=entry["provider"], orchestrator_url=entry["orchestrator_url"],
                credentials_root=entry["credentials_root"], scratch_root=entry["scratch_root"],
                executors=tuple(entry["executors"]), registration_token_file=entry.get("registration_token_file"),
                capabilities=dict(entry.get("capabilities", {})),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid deployment profile: {exc}") from exc
        if profile.profile_id in profiles:
            raise ValueError(f"duplicate deployment profile: {profile.profile_id}")
        if profile.provider == "docker":
            docker = entry.get("docker")
            if not isinstance(docker, dict):
                raise ValueError("Docker deployment profiles require a docker object")
            allowed_docker = {"image", "network", "docker_executable", "resource_args"}
            if set(docker) - allowed_docker:
                raise ValueError("Docker profile has unsupported fields")
            try:
                docker_profile = DockerPullWorkerProfile(
                    image=docker["image"], network=docker.get("network", "bridge"),
                    credentials_root=profile.credentials_root, scratch_root=profile.scratch_root,
                    registration_token_file=profile.registration_token_file,
                    docker_executable=docker.get("docker_executable", "docker"),
                    resource_args=tuple(docker.get("resource_args", ())),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid Docker deployment profile: {exc}") from exc
            # A provider is keyed by profile ID internally, then mapped to a
            # deployment's opaque provider key below.  This permits multiple
            # Docker safety policies in one orchestrator process.
            provider_key = f"docker:{profile.profile_id}"
            providers[provider_key] = DockerPullWorkerProvider(docker_profile)
            profile = PullWorkerDeploymentProfile(
                profile_id=profile.profile_id, provider=provider_key, orchestrator_url=profile.orchestrator_url,
                credentials_root=profile.credentials_root, scratch_root=profile.scratch_root,
                executors=profile.executors, registration_token_file=profile.registration_token_file,
                capabilities=profile.capabilities,
            )
        elif profile.provider != "local-process":
            raise ValueError("deployment profile provider must be local-process or docker")
        profiles[profile.profile_id] = profile
    return profiles, providers


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Oracle Builder orchestration API.")
    parser.add_argument("--database", required=True, help="Central SQLite orchestration database.")
    parser.add_argument("--workspace-root", required=True, help="Root containing approved working inputs and recipes.")
    parser.add_argument("--artifact-root", help="Owned runtime staging root for uploads, evaluations, imported products, and packages.")
    parser.add_argument("--s3-bucket", help="Optional S3-compatible immutable replica bucket for sealed artifacts.")
    parser.add_argument("--s3-prefix", default=os.environ.get("ORACLE_ARTIFACT_S3_PREFIX", "oracle-builder"),
                        help="Object key prefix used with --s3-bucket (default: oracle-builder).")
    parser.add_argument("--s3-endpoint-url", default=os.environ.get("ORACLE_ARTIFACT_S3_ENDPOINT_URL"),
                        help="Optional S3-compatible endpoint, for example a MinIO URL.")
    parser.add_argument("--s3-region", default=os.environ.get("AWS_REGION"),
                        help="Optional S3 region; credentials remain in the standard AWS provider chain.")
    parser.add_argument("--runs-root", help="Canonical model-run directory; defaults to WORKSPACE_ROOT/runs.")
    parser.add_argument("--datasets-root", help="Canonical training-dataset directory; defaults to WORKSPACE_ROOT/datasets.")
    parser.add_argument("--log-root", help="Directory containing allow-listed orchestrator and Web GUI diagnostic logs; defaults beside the database.")
    parser.add_argument("--browse-root", action="append", default=[], help="Additional allow-listed root exposed to the file explorer.")
    parser.add_argument("--training-catalog-root", action="append", default=[], help="Read-only allow-listed root to scan for available training sets; may be repeated.")
    parser.add_argument("--upload-limit-mib", type=int, default=10_240, help="Maximum uploaded file size in MiB.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8110)
    parser.add_argument("--role-tokens-sha256", default=os.environ.get("ORACLE_ORCHESTRATOR_ROLE_TOKENS_SHA256"),
                        help="JSON role-to-SHA256 token map (or ORACLE_ORCHESTRATOR_ROLE_TOKENS_SHA256).")
    parser.add_argument("--allow-unauthenticated-mutations", action="store_true",
                        help="Development-only: do not require configured API role tokens.")
    parser.add_argument("--worker-deployment-profiles", default=os.environ.get("ORACLE_WORKER_DEPLOYMENT_PROFILES"),
                        help="Operator-owned JSON deployment profile file (or ORACLE_WORKER_DEPLOYMENT_PROFILES).")
    args = parser.parse_args()
    try:
        configured_roles = json.loads(args.role_tokens_sha256) if args.role_tokens_sha256 else {}
        if not isinstance(configured_roles, dict):
            raise ValueError("role token configuration must be a JSON object")
        configured_roles = role_token_digests(configured_roles)
    except (json.JSONDecodeError, ValueError) as exc:
        parser.error(f"invalid --role-tokens-sha256: {exc}")
    if not configured_roles and not args.allow_unauthenticated_mutations:
        parser.error("configure --role-tokens-sha256 (hashed tokens only) or explicitly use --allow-unauthenticated-mutations for local development")
    artifact_root = os.path.abspath(os.path.expanduser(args.artifact_root)) if args.artifact_root else os.path.join(os.path.dirname(os.path.abspath(args.database)), "oracle-artifacts")
    artifact_store = None
    if args.s3_bucket:
        try:
            import boto3
        except ImportError:
            parser.error("--s3-bucket requires the optional dependency: pip install 'oracle-builder[storage-s3]'")
        client_options = {key: value for key, value in {
            "endpoint_url": args.s3_endpoint_url,
            "region_name": args.s3_region,
        }.items() if value}
        artifact_store = S3ReplicatedArtifactStore(
            LocalArtifactStore(os.path.join(artifact_root, "artifact-store")),
            bucket=args.s3_bucket,
            prefix=args.s3_prefix,
            client=boto3.client("s3", **client_options),
        )
    try:
        deployment_profiles, deployment_providers = load_deployment_profiles(args.worker_deployment_profiles)
    except ValueError as exc:
        parser.error(str(exc))
    orchestrator = Orchestrator(
        args.database, workspace_root=args.workspace_root, artifact_root=args.artifact_root, runs_root=args.runs_root,
        datasets_root=args.datasets_root, browse_roots=args.browse_root,
        training_catalog_roots=args.training_catalog_root or None, log_root=args.log_root,
        upload_limit_bytes=args.upload_limit_mib * 1024 * 1024, artifact_store=artifact_store,
        deployment_profiles=deployment_profiles, deployment_providers=deployment_providers,
    )
    uvicorn.run(create_app(orchestrator, role_tokens=configured_roles), host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    main()
