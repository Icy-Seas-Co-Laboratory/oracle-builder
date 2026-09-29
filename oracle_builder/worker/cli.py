"""Command line entry point for the stateless Oracle pull worker.

The worker is an outbound-only lease client. It deliberately has no HTTP
server, model registry, compute queue, or Oracle Serve compatibility mode.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import uuid
from collections.abc import Sequence

from oracle_builder.worker.pull import (DeterministicPackagingExecutor, ExecutorRegistry,
                                        InferenceExecutor,
                                        OrchestratorPullClient, PullProtocolError,
                                        PullWorkerLoop, PullWorkerRuntime, TrainingExecutor,
                                        load_worker_credentials, recover_output_upload)


def _positive_int(value: str) -> int:
    try: parsed = int(value)
    except ValueError as exc: raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1: raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oracle-worker", description="Run a stateless Oracle pull worker.")
    parser.add_argument("--orchestrator-url")
    parser.add_argument("--pool-id")
    parser.add_argument("--name", default=os.environ.get("ORACLE_BUILDER_WORKER_NAME", "oracle-worker"))
    parser.add_argument("--registration-token", default=os.environ.get("ORACLE_BUILDER_WORKER_REGISTRATION_TOKEN"))
    parser.add_argument("--endpoint")
    parser.add_argument("--capability", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--credentials-file", default=os.environ.get("ORACLE_BUILDER_PULL_CREDENTIALS_FILE"))
    parser.add_argument("--scratch-root", default=os.environ.get("ORACLE_BUILDER_WORKER_SCRATCH_ROOT"))
    parser.add_argument("--executor", action="append", choices=("package", "train", "infer"))
    parser.add_argument("--recover-output", metavar="RECOVERY_DIR", help="Resume a retained multipart output upload using existing worker credentials.")
    parser.add_argument("--recover-lease-token", help="Optional active lease token for legacy/manual recovery; normally a fresh token is issued from worker identity.")
    parser.add_argument("--max-jobs", type=_positive_int)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--lease-ttl-seconds", type=_positive_int, default=30)
    parser.add_argument("--poll-interval-seconds", type=float, default=2.0)
    parser.add_argument(
        "--artifact-timeout-seconds", type=_positive_int, default=900,
        help="maximum time to wait for large artifact delivery or output upload (default: 900)",
    )
    return parser


def _capabilities(values: Sequence[str], parser: argparse.ArgumentParser) -> dict[str, object]:
    result: dict[str, object] = {}
    for value in values:
        if "=" not in value: parser.error("--capability must use KEY=VALUE syntax")
        key, raw = (part.strip() for part in value.split("=", 1))
        if not key or not raw: parser.error("--capability must use non-empty KEY=VALUE syntax")
        result[key] = [item.strip() for item in raw.split(",") if item.strip()] if key == "gpu_ids" else int(raw) if raw.isdecimal() else raw
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.poll_interval_seconds <= 0: parser.error("--poll-interval-seconds must be positive")
    if args.recover_output:
        if not args.orchestrator_url or not args.credentials_file:
            parser.error("--recover-output requires --orchestrator-url and --credentials-file")
        try:
            credentials = load_worker_credentials(args.credentials_file)
            if credentials is None:
                raise PullProtocolError("worker credentials file does not exist")
            recovery_root = Path(args.recover_output).expanduser().resolve()
            metadata = json.loads((recovery_root / "recovery.json").read_text(encoding="utf-8"))
            lease_id = metadata.get("lease_id") if isinstance(metadata, dict) else None
            if not isinstance(lease_id, str) or not lease_id:
                raise PullProtocolError("output recovery metadata is invalid")
            client = OrchestratorPullClient(args.orchestrator_url)
            lease_token = args.recover_lease_token
            if not lease_token:
                resumed = client.resume_output_publication(
                    lease_id=lease_id, worker_id=credentials.worker_id,
                    worker_token=credentials.worker_token,
                )
                lease_token = resumed.get("lease_token") if isinstance(resumed, dict) else None
                if not isinstance(lease_token, str) or not lease_token:
                    raise PullProtocolError("output recovery did not receive a fresh lease token")
            recover_output_upload(client, recovery_dir=recovery_root, worker_token=credentials.worker_token,
                                  lease_token=lease_token, remove_on_success=False)
            client.complete(lease_id=lease_id, worker_token=credentials.worker_token,
                            lease_token=lease_token, outcome="succeeded")
            shutil.rmtree(recovery_root)
        except (OSError, ValueError, PullProtocolError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        print(json.dumps({"recovered_output": str(args.recover_output)}, sort_keys=True, separators=(",", ":")))
        return
    if not args.orchestrator_url or not args.pool_id or not args.credentials_file or not args.scratch_root or not args.executor:
        parser.error("--orchestrator-url, --pool-id, --credentials-file, --scratch-root, and --executor are required")
    capabilities = _capabilities(args.capability, parser)
    registry = ExecutorRegistry()
    for executor in args.executor:
        registry.register(executor, (
            DeterministicPackagingExecutor() if executor == "package" else
            TrainingExecutor() if executor == "train" else InferenceExecutor()
        ))
    capabilities["actions"] = list(registry.actions)
    capabilities["worker_control_v1"] = True
    if "infer" in registry.actions:
        capabilities["inference_shards_v1"] = True
    capabilities["execution_instance_id"] = str(uuid.uuid4())
    if "train" in registry.actions:
        capabilities["training_verification_v1"] = True
        capabilities["work_unit_v2"] = True
        capabilities["training_segments_v1"] = True
        capabilities["execution_instance_id"] = str(uuid.uuid4())
    client = OrchestratorPullClient(
        args.orchestrator_url, artifact_timeout_seconds=args.artifact_timeout_seconds,
    )
    runtime = PullWorkerRuntime(client, scratch_root=args.scratch_root, executor=registry, supervised=True)
    loop = PullWorkerLoop(client, runtime=runtime, pool_id=args.pool_id, name=args.name,
                          credentials_path=args.credentials_file, registration_token=args.registration_token,
                          capabilities=capabilities, endpoint=args.endpoint,
                          lease_ttl_seconds=args.lease_ttl_seconds, poll_interval_seconds=args.poll_interval_seconds)
    try: completed = loop.run(max_jobs=args.max_jobs, once=args.once)
    except (PullProtocolError, ValueError) as exc: parser.error(str(exc))
    print(json.dumps({"completed_jobs": completed}, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":  # pragma: no cover
    main()
