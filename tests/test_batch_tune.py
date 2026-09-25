from __future__ import annotations

from oracle_builder.training import batch_tune


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
