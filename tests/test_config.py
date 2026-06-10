from __future__ import annotations

from pathlib import Path

from oracle_builder.config import resolve_config


def test_resolve_config_adds_defaults_and_paths(tmp_path: Path):
    config_path = tmp_path / "config.toml"
    input_path = tmp_path / "data.sqlite"
    run_dir = tmp_path / "runs" / "test"
    config_path.write_text(
        """
[run]
task = "classification"
model = "simple_cnn"

[data]
input_shape = [16, 16, 1]
num_classes = 2

[training]
loss = "sparse_categorical_crossentropy"
"""
    )
    input_path.write_text("")
    config = resolve_config(config_path, input_path, run_dir)
    assert config["data"]["batch_size"] == 16
    assert config["run"]["seed"] == 123
    assert config["paths"]["run_dir"] == str(run_dir.resolve())

