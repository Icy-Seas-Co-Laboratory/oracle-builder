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


def _gpu_loads() -> dict[int, tuple[int, int]]:
    """Return ``{index: (MiB used, utilization %)}`` when NVIDIA tools exist."""
    try:
        result = run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False, timeout=2, stdin=DEVNULL,
        )
        if result.returncode:
            return {}
        loads = {}
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 3:
                loads[int(parts[0])] = (int(float(parts[1])), int(float(parts[2])))
        return loads
    except (FileNotFoundError, OSError):
        return {}


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
    loads = _gpu_loads()
    max_memory = int(settings.get("gpu_light_share_memory_mb", 1024))
    max_utilization = int(settings.get("gpu_light_share_utilization_percent", 15))
    if bool(settings.get("require_unused_gpu", False)):
        max_memory = 0
        max_utilization = 0
    candidates = sorted(
        available,
        key=lambda device: loads.get(int(device.rsplit(":", 1)[1]), (0, 0)),
    )
    for device in candidates:
        index = int(device.rsplit(":", 1)[1])
        memory_used, utilization = loads.get(index, (0, 0))
        if (memory_used > max_memory or utilization > max_utilization) and not bool(
            settings.get("allow_busy_fallback", False)
        ):
            continue
        lease = _reserve_gpu(index, settings)
        if lease is not None:
            return device, lease
    if bool(settings.get("allow_busy_fallback", False)):
        for device in candidates:
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
