from __future__ import annotations

import json
import fcntl
import os
from subprocess import DEVNULL, run
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import tensorflow as tf


@dataclass(frozen=True)
class DistributionInfo:
    requested_strategy: str
    resolved_strategy: str
    replicas: int
    devices: list[str]
    global_batch_size: int
    per_replica_batch_size: int
    cross_device_ops: str
    gpu_lease_path: str | None = None


def _device_name(value: str) -> str:
    value = str(value).strip()
    if value.startswith("/"):
        return value
    return f"/{value.upper()}"


def _cross_device_ops(name: str):
    normalized = name.lower()
    if normalized == "auto":
        return None
    if normalized == "nccl":
        return tf.distribute.NcclAllReduce()
    if normalized in {"hierarchical_copy", "hierarchical-copy"}:
        return tf.distribute.HierarchicalCopyAllReduce()
    raise ValueError(
        "distribution.cross_device_ops must be auto, nccl, or hierarchical_copy"
    )


_GPU_LEASES: dict[str, Any] = {}


def _busy_gpu_indices() -> set[int]:
    """Best-effort NVIDIA occupancy query; leases handle the race we control."""
    try:
        result = run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"],
            capture_output=True, text=True, check=False, timeout=2, stdin=DEVNULL,
        )
        if result.returncode:
            return set()
        uuid_result = run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, check=False, timeout=2, stdin=DEVNULL,
        )
        mapping = {
            parts[1].strip(): int(parts[0].strip())
            for line in uuid_result.stdout.splitlines()
            if len(parts := line.split(",", 1)) == 2
        }
        return {mapping[value.strip()] for value in result.stdout.splitlines() if value.strip() in mapping}
    except (FileNotFoundError, OSError):
        return set()


def _reserve_gpu(index: int, settings: dict[str, Any]) -> str | None:
    directory = Path(settings.get("gpu_lease_directory", "/tmp/oracle-builder-gpu-leases"))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"gpu-{index}.lock"
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid": os.getpid()}) + "\n")
    handle.flush()
    _GPU_LEASES[str(path)] = handle
    return str(path)


def _select_auto_gpu(available: list[str], settings: dict[str, Any]) -> tuple[str | None, str | None]:
    busy = _busy_gpu_indices()
    for device in available:
        index = int(device.rsplit(":", 1)[1])
        if bool(settings.get("require_unused_gpu", True)) and index in busy:
            continue
        lease = _reserve_gpu(index, settings)
        if lease is not None:
            return device, lease
    if bool(settings.get("allow_busy_fallback", False)):
        for device in available:
            lease = _reserve_gpu(int(device.rsplit(":", 1)[1]), settings)
            if lease is not None:
                return device, lease
    return None, None


def select_distribution_strategy(
    config: dict[str, Any],
) -> tuple[tf.distribute.Strategy, DistributionInfo]:
    settings = config.get("distribution", {})
    requested = str(settings.get("strategy", "auto")).lower()
    if requested == "none":
        requested = "single"
    if requested not in {"auto", "single", "mirrored", "cpu"}:
        raise ValueError(
            "distribution.strategy must be auto, single, mirrored, or cpu"
        )

    physical_gpus = tf.config.list_physical_devices("GPU")
    if bool(settings.get("memory_growth", True)):
        for device in physical_gpus:
            try:
                tf.config.experimental.set_memory_growth(device, True)
            except RuntimeError:
                # Device initialization may already have occurred during environment inspection.
                pass
    logical_gpus = tf.config.list_logical_devices("GPU")
    available_gpu_names = [f"/GPU:{index}" for index, _device in enumerate(logical_gpus)]
    requested_devices = [
        _device_name(device) for device in settings.get("devices", [])
    ]
    selected_devices = requested_devices or available_gpu_names
    unknown = set(requested_devices) - set(available_gpu_names)
    if unknown:
        raise ValueError(
            f"Requested distribution devices are unavailable: {sorted(unknown)}; "
            f"available GPUs: {available_gpu_names}"
        )

    lease_path = None
    if requested == "auto" and not requested_devices and available_gpu_names:
        selected, lease_path = _select_auto_gpu(available_gpu_names, settings)
        if selected is None:
            raise RuntimeError("No unused GPU is available for this Oracle Builder run")
        selected_devices = [selected]
    elif requested == "single" and selected_devices:
        selected_devices = selected_devices[:1]
    should_mirror = requested == "mirrored"
    cross_name = str(settings.get("cross_device_ops", "auto")).lower()
    if should_mirror and len(selected_devices) < 2:
        if requested == "mirrored" and not bool(
            settings.get("fallback_to_single", True)
        ):
            raise RuntimeError(
                "MirroredStrategy requested but fewer than two GPUs are available"
            )
        should_mirror = False

    if should_mirror:
        cross_ops = _cross_device_ops(cross_name)
        kwargs = {"devices": selected_devices}
        if cross_ops is not None:
            kwargs["cross_device_ops"] = cross_ops
        strategy = tf.distribute.MirroredStrategy(**kwargs)
        resolved = "mirrored"
    elif requested == "cpu":
        strategy = tf.distribute.OneDeviceStrategy("/CPU:0")
        resolved = "cpu"
    else:
        strategy = tf.distribute.OneDeviceStrategy(selected_devices[0]) if selected_devices else tf.distribute.get_strategy()
        resolved = "single"

    replicas = int(strategy.num_replicas_in_sync)
    global_batch = int(config["data"].get("batch_size", 16))
    if global_batch % replicas:
        raise ValueError(
            f"Global data.batch_size={global_batch} must be divisible by "
            f"{replicas} synchronized replicas"
        )
    try:
        worker_devices = [
            str(device) for device in strategy.extended.worker_devices
        ]
    except (AttributeError, RuntimeError):
        if resolved == "mirrored":
            worker_devices = list(selected_devices)
        elif resolved == "cpu":
            worker_devices = ["/CPU:0"]
        else:
            worker_devices = [available_gpu_names[0]] if available_gpu_names else ["/CPU:0"]
    info = DistributionInfo(
        requested_strategy=requested,
        resolved_strategy=resolved,
        replicas=replicas,
        devices=worker_devices,
        global_batch_size=global_batch,
        per_replica_batch_size=global_batch // replicas,
        cross_device_ops=cross_name,
        gpu_lease_path=lease_path,
    )
    return strategy, info


def write_distribution_info(
    info: DistributionInfo,
    run_dir: str | Path,
) -> None:
    from oracle_builder.artifacts.layout import RunLayout

    RunLayout(run_dir).distribution.write_text(
        json.dumps(asdict(info), indent=2, sort_keys=True) + "\n"
    )
