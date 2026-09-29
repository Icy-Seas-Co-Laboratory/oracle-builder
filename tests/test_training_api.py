from __future__ import annotations

import pytest

import model_training


def test_typed_training_request_rejects_ambiguous_resume_arguments() -> None:
    with pytest.raises(ValueError, match="resume uses"):
        model_training.TrainingRequest(resume="runs/old", output="new")


def test_cli_is_a_thin_adapter_to_typed_training_request(monkeypatch) -> None:
    captured = {}

    def fake_train(request):
        captured["request"] = request
        return 17

    monkeypatch.setattr(model_training, "train", fake_train)
    assert model_training.main(["--config", "recipe.toml", "--input", "dataset.sqlite", "--output", "baseline", "--runs-dir", "runs"]) == 17
    assert captured["request"] == model_training.TrainingRequest(
        config="recipe.toml", input="dataset.sqlite", output="baseline", runs_dir="runs",
    )
