from __future__ import annotations

import json
import zipfile

from oracle_builder.orchestration.service import Orchestrator


def test_legacy_inspection_reads_sealed_counts_and_stem(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    with zipfile.ZipFile(model_dir / "final.keras", "w") as archive:
        archive.writestr("config.json", json.dumps({"config": {"layers": [
            {"class_name": "InputLayer", "config": {"name": "image"}},
            {"class_name": "Conv2D", "config": {"name": "stem_conv", "filters": 32, "kernel_size": [3, 3], "strides": [2, 2]}},
        ]}}))

    result = Orchestrator._legacy_model_inspection(
        model_dir, "Total params: 2,234\nTrainable params: 1,200\nNon-trainable params: 34\n"
    )

    assert result is not None
    assert result["parameter_count"] == 1234
    assert result["trainable_parameter_count"] == 1200
    assert result["non_trainable_parameter_count"] == 34
    assert result["optimizer_parameter_count"] == 1000
    assert result["layer_count"] == 2
    assert result["layers"][1]["filters"] == 32
    assert result["layers"][1]["strides"] == [2, 2]
