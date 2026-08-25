from __future__ import annotations

import json

import pytest

from oracle_builder.clustering import cli


def test_encoder_run_infers_fit_mode(monkeypatch, capsys):
    captured = {}

    def fake_fit(config, input_path, encoder_run, *, reopen_and_reseal):
        captured.update(
            config=config,
            input_path=input_path,
            encoder_run=encoder_run,
            reopen_and_reseal=reopen_and_reseal,
        )
        return {"attached": True}

    monkeypatch.setattr(cli, "fit_clustering_evidence_from_encoder", fake_fit)

    assert cli.main(
        [
            "--config", "clustering.toml",
            "--input", "rois.sqlite",
            "--encoder-run", "runs/roi-clusters",
            "--reopen-reseal",
        ]
    ) == 0

    assert captured == {
        "config": "clustering.toml",
        "input_path": "rois.sqlite",
        "encoder_run": "runs/roi-clusters",
        "reopen_and_reseal": True,
    }
    assert json.loads(capsys.readouterr().out) == {"attached": True}


def test_train_mode_reports_missing_output_without_name_error(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["--config", "clustering.toml", "--input", "rois.sqlite"])

    assert error.value.code == 2
    assert "--output is required in train mode" in capsys.readouterr().err
