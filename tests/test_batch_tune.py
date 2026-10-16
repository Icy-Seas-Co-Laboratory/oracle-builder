from __future__ import annotations

from oracle_builder.training import batch_tune
import pytest


def test_memory_errors_recognized_without_oom_in_message():
    class ResourceExhaustedError(Exception):
        pass

    assert batch_tune._is_memory_error(ResourceExhaustedError("Exception in BatchNormalization.call()"))
    assert batch_tune._is_memory_error(RuntimeError("failed to allocate memory [Op:SquaredDifference]"))
    assert batch_tune._is_memory_error(MemoryError())
    assert not batch_tune._is_memory_error(ValueError("Invalid input shape"))


def test_auto_tune_recovers_from_tensorflow_allocation_error(monkeypatch):
    class ResourceExhaustedError(Exception):
        pass

    monkeypatch.setattr("oracle_builder.config.resolve_config", lambda *args: {})
    def probe(config, input_path, batch_size):
        if batch_size > 4:
            raise ResourceExhaustedError("Exception encountered when calling BatchNormalization.call().")
        return {"status": "passed", "observed_batch_size": batch_size}

    monkeypatch.setattr(batch_tune, "_probe", probe)
    report = batch_tune.tune("config.toml", "data.sqlite", maximum=32)
    assert report["ready"]
    assert report["largest_verified_batch_size"] == 4
    assert report["recommended_batch_size"] == 3
    assert any(attempt["status"] == "out_of_memory" for attempt in report["attempts"])


def test_auto_tune_does_not_hide_non_memory_failures(monkeypatch):
    monkeypatch.setattr("oracle_builder.config.resolve_config", lambda *args: {})
    def probe(*args):
        raise ValueError("Invalid input shape")
    monkeypatch.setattr(batch_tune, "_probe", probe)
    with pytest.raises(RuntimeError, match="Invalid input shape"):
        batch_tune.tune("config.toml", "data.sqlite")


def test_auto_tune_binary_searches_memory_boundary_and_verifies_safe_result(monkeypatch):
    monkeypatch.setattr(
        "oracle_builder.config.resolve_config",
        lambda *_args, **_kwargs: {"run": {"model": "simple_cnn"}, "data": {"batch_size": 16}},
    )
    probed: list[int] = []

    def probe(_config, _input_path, batch_size: int):
        probed.append(batch_size)
        if batch_size > 12:
            raise RuntimeError("ResourceExhaustedError: OOM")
        return {"status": "passed", "observed_batch_size": batch_size}

    monkeypatch.setattr(batch_tune, "_probe", probe)
    report = batch_tune.tune("definition.toml", "frozen.sqlite", maximum=32, safety_factor=0.8)

    assert report["ready"] is True
    assert report["largest_verified_batch_size"] == 12
    assert report["recommended_batch_size"] == 9
    assert probed[-1] == 9


def test_auto_tune_selects_first_power_of_two_in_vram_target_band(monkeypatch):
    monkeypatch.setattr(
        "oracle_builder.config.resolve_config",
        lambda *_args, **_kwargs: {"run": {"model": "simple_cnn"}, "data": {"batch_size": 16}},
    )
    probed: list[int] = []

    def probe(_config, _input_path, batch_size: int):
        probed.append(batch_size)
        return {
            "status": "passed",
            "observed_batch_size": batch_size,
            "memory": {"peak_bytes": batch_size * 100 * 1024 * 1024, "source": "tensorflow_allocator"},
        }

    monkeypatch.setattr(batch_tune, "_probe", probe)
    report = batch_tune.tune(
        "definition.toml", "frozen.sqlite", maximum=32,
        vram_total_mib=1024, target_vram_min=0.30, target_vram_max=0.80,
    )

    assert report["tuning_strategy"] == "vram_target_power_of_two"
    assert report["recommended_batch_size"] == 4
    assert report["selected_peak_memory_mib"] == 400
    assert probed == [1, 2, 4]


def test_auto_tune_keeps_previous_power_of_two_when_next_probe_exceeds_vram_ceiling(monkeypatch):
    monkeypatch.setattr(
        "oracle_builder.config.resolve_config",
        lambda *_args, **_kwargs: {"run": {"model": "simple_cnn"}, "data": {"batch_size": 16}},
    )

    def probe(_config, _input_path, batch_size: int):
        return {
            "status": "passed", "observed_batch_size": batch_size,
            "memory": {"peak_bytes": batch_size * 220 * 1024 * 1024, "source": "tensorflow_allocator"},
        }

    monkeypatch.setattr(batch_tune, "_probe", probe)
    report = batch_tune.tune(
        "definition.toml", "frozen.sqlite", maximum=32,
        vram_total_mib=1024, target_vram_min=0.60, target_vram_max=0.80,
    )

    assert report["recommended_batch_size"] == 2
    assert report["attempts"][-1]["batch_size"] == 4
    assert report["attempts"][-1]["vram_fraction"] > 0.80


def test_auto_tune_preserves_replica_batch_multiple(monkeypatch):
    monkeypatch.setattr('oracle_builder.config.resolve_config', lambda *args: {})
    seen = []
    def probe(config, input_path, batch_size):
        assert batch_size % 4 == 0
        seen.append(batch_size)
        if batch_size > 20:
            raise RuntimeError('OOM')
        return {'status': 'passed', 'observed_batch_size': batch_size}
    monkeypatch.setattr(batch_tune, '_probe', probe)
    report = batch_tune.tune('config.toml', 'data.sqlite', minimum=4, maximum=64)
    assert report['recommended_batch_size'] == 16
    assert report['largest_verified_batch_size'] == 20
    assert seen[-1] == 16
