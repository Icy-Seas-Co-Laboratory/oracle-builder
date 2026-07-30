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


def test_auto_uses_mirrored_strategy_for_multiple_gpus(monkeypatch):
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
        lambda kind: fake_devices if kind == "GPU" else [],
    )
    monkeypatch.setattr(
        "oracle_builder.training.distribution.tf.config.list_logical_devices",
        lambda kind: fake_devices if kind == "GPU" else [],
    )
    mirrored_calls = []
    monkeypatch.setattr(
        "oracle_builder.training.distribution.tf.distribute.MirroredStrategy",
        lambda **kwargs: mirrored_calls.append(kwargs) or fake_strategy,
    )

    strategy, info = select_distribution_strategy(base_config())

    assert strategy is fake_strategy
    assert info.resolved_strategy == "mirrored"
    assert info.replicas == 2
    assert info.per_replica_batch_size == 4
    assert mirrored_calls[0]["devices"] == ["/GPU:0", "/GPU:1"]


def test_global_batch_must_be_divisible_by_replica_count(monkeypatch):
    config = base_config()
    config["data"]["batch_size"] = 7
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
