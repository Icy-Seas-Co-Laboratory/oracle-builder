from __future__ import annotations

import json
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

    should_mirror = requested == "mirrored" or (
        requested == "auto" and len(selected_devices) > 1
    )
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
        strategy = tf.distribute.get_strategy()
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
