from __future__ import annotations

from types import SimpleNamespace

import pytest

from oracle_builder.config import validate_config
from oracle_builder.training.distribution import select_distribution_strategy


def base_config():
    return {
        "run": {"task": "classification", "model": "simple_cnn"},
        "data": {"input_shape": [16, 16, 1], "num_classes": 2, "batch_size": 8},
        "training": {"loss": "sparse_categorical_crossentropy"},
        "distribution": {
            "strategy": "auto",
            "devices": [],
            "cross_device_ops": "auto",
            "fallback_to_single": True,
            "memory_growth": False,
        },
    }


def test_auto_strategy_reports_global_and_per_replica_batch_sizes():
    strategy, info = select_distribution_strategy(base_config())

    assert info.replicas == strategy.num_replicas_in_sync
    assert info.global_batch_size == 8
    assert info.per_replica_batch_size == 8 // info.replicas
    assert info.resolved_strategy in {"single", "mirrored"}


def test_cpu_strategy_can_be_requested_explicitly():
    config = base_config()
    config["distribution"]["strategy"] = "cpu"

    strategy, info = select_distribution_strategy(config)

    assert strategy.num_replicas_in_sync == 1
    assert info.resolved_strategy == "cpu"
    assert any("CPU" in device.upper() for device in info.devices)


def test_auto_uses_one_unused_gpu_for_multiple_gpus(monkeypatch, tmp_path):
    fake_devices = [
        SimpleNamespace(name="/device:GPU:0"),
        SimpleNamespace(name="/device:GPU:1"),
    ]
    fake_strategy = SimpleNamespace(
        num_replicas_in_sync=1,
        extended=SimpleNamespace(worker_devices=("/device:GPU:0",)),
    )
    monkeypatch.setattr(
        "oracle_builder.training.distribution.tf.config.list_physical_devices",
        lambda kind: fake_devices if kind == "GPU" else [],
    )
    monkeypatch.setattr(
        "oracle_builder.training.distribution.tf.config.list_logical_devices",
        lambda kind: fake_devices if kind == "GPU" else [],
    )
    one_device_calls = []
    monkeypatch.setattr(
        "oracle_builder.training.distribution.tf.distribute.OneDeviceStrategy",
        lambda device: one_device_calls.append(device) or fake_strategy,
    )
    monkeypatch.setattr("oracle_builder.training.distribution._busy_gpu_indices", lambda: set())
    config = base_config()
    config["distribution"]["gpu_lease_directory"] = str(tmp_path)

    strategy, info = select_distribution_strategy(config)

    assert strategy is fake_strategy
    assert info.resolved_strategy == "single"
    assert info.replicas == 1
    assert info.per_replica_batch_size == 8
    assert one_device_calls == ["/GPU:0"]
    assert info.gpu_lease_path is not None


def test_global_batch_must_be_divisible_by_replica_count(monkeypatch):
    config = base_config()
    config["data"]["batch_size"] = 7
    config["distribution"]["strategy"] = "mirrored"
    fake_devices = [
        SimpleNamespace(name="/device:GPU:0"),
        SimpleNamespace(name="/device:GPU:1"),
    ]
    fake_strategy = SimpleNamespace(
        num_replicas_in_sync=2,
        extended=SimpleNamespace(worker_devices=("/device:GPU:0", "/device:GPU:1")),
    )
    monkeypatch.setattr(
        "oracle_builder.training.distribution.tf.config.list_physical_devices",
        lambda kind: fake_devices,
    )
    monkeypatch.setattr(
        "oracle_builder.training.distribution.tf.config.list_logical_devices",
        lambda kind: fake_devices,
    )
    monkeypatch.setattr(
        "oracle_builder.training.distribution.tf.distribute.MirroredStrategy",
        lambda **_kwargs: fake_strategy,
    )

    with pytest.raises(ValueError, match="must be divisible"):
        select_distribution_strategy(config)


@pytest.mark.parametrize("strategy", ["many_gpus", "distributed"])
def test_invalid_distribution_strategy_is_rejected(strategy):
    config = base_config()
    config["distribution"]["strategy"] = strategy

    with pytest.raises(ValueError, match="distribution.strategy"):
        validate_config(config)
