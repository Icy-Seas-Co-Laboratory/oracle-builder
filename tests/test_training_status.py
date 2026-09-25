import io
import json

from oracle_builder.training.callbacks import build_callbacks
from oracle_builder.training.status import RichTrainingStatusCallback, _sparkline, _trend


def test_text_status_emits_one_compact_line_per_epoch_without_batch_noise():
    stream = io.StringIO()
    callback = RichTrainingStatusCallback(
        phase="SSL · BYOL", epochs=2, display="text", stream=stream
    )
    callback.set_params({"steps": 3})
    callback.on_train_begin()
    callback.on_epoch_begin(0)
    callback.on_train_batch_end(0, {"loss": 2.0, "variance_loss": 1.0})
    callback.on_train_batch_end(1, {"loss": 1.0, "variance_loss": 0.5})
    callback.on_epoch_end(0, {"loss": 1.0, "val_loss": 1.2, "accuracy": 0.8})
    callback.on_train_end()

    output = stream.getvalue()
    assert output.count("\n") == 2
    assert "epoch 1/2 started" in output
    assert "epoch 1/2 completed" in output
    assert "loss=1" in output
    assert "val_loss=1.2" in output
    assert "variance_loss" not in output


def test_supervised_callback_factory_uses_rich_status_by_default(tmp_path):
    callbacks = build_callbacks(
        {
            "training": {"epochs": 3},
            "recovery": {"enabled": False},
        },
        tmp_path,
        tmp_path / "training.sqlite",
        "run-1",
    )

    status = next(item for item in callbacks if isinstance(item, RichTrainingStatusCallback))
    assert status.phase == "Supervised training"
    assert status.epochs == 3
    assert status.display == "rich"


def test_status_retains_validation_metrics_and_metric_history_between_epochs():
    callback = RichTrainingStatusCallback(phase="Classification", epochs=3, display="off")
    callback.set_params({"steps": 2})
    callback.on_train_begin()
    callback.on_epoch_begin(0)
    callback.on_epoch_end(0, {"loss": 1.0, "val_loss": 1.2, "accuracy": 0.7, "val_accuracy": 0.6})
    callback.on_epoch_begin(1)
    callback.on_train_batch_end(0, {"loss": 0.8, "accuracy": 0.8})

    assert callback._metrics["val_loss"] == 1.2
    assert callback._metrics["val_accuracy"] == 0.6
    assert "loss" not in callback._metrics
    assert callback._history["loss"] == [1.0]


def test_status_distinguishes_first_batch_loading_from_optimizer_execution():
    callback = RichTrainingStatusCallback(phase="Classification", epochs=1, display="off")
    callback.set_params({"steps": 1})
    callback.on_epoch_begin(0)
    assert callback._batch_status == "Preparing first batch"

    callback.on_input_batch_loading(0)
    assert callback._batch_status == "Loading batch 1 from input pipeline"

    callback.on_train_batch_begin(0)
    assert callback._batch_status == "Computing first optimizer update (initial shape may compile)"

    callback.on_train_batch_end(0)
    assert callback._batch_status == "Completed batch 1"


def test_sparkline_and_direction_make_static_and_changing_metrics_visible():
    assert _sparkline([0.5, 0.5, 0.5]) == "▅▅▅"
    assert _trend([0.5, 0.5]) == "→"
    assert _trend([0.5, 0.6]) == "↗"
    assert _trend([0.6, 0.5]) == "↘"


def test_rich_board_prioritizes_live_and_completed_metrics_over_an_empty_metric_table():
    callback = RichTrainingStatusCallback(phase="SSL", epochs=1, display="off")
    callback._batch_metrics = {"loss": 1.2, "accuracy": 0.7}
    callback._latest_validation = {"val_loss": 1.3, "val_accuracy": 0.6}
    callback._metrics = {"loss": 1.1, "val_loss": 1.3}

    board = callback._board()
    metrics = board.renderable.renderables[2]

    assert len(metrics.rows) == 3
    assert metrics.row_styles == ["", "on grey15"]


def test_status_exposes_post_epoch_analysis_progress():
    callback = RichTrainingStatusCallback(phase="Classification", epochs=1, display="off")
    callback.begin_post_epoch_analysis("train", total_batches=20)
    callback.update_post_epoch_analysis(5)

    assert "Post-epoch analysis" in callback._external_summary()
    assert "5/20 batches" in callback._external_summary()

    callback.end_post_epoch_analysis()
    assert callback._external_summary() is None


def test_status_writes_versioned_atomic_web_snapshot_with_transient_batch_metrics(tmp_path):
    status_path = tmp_path / "training-status.json"
    callback = RichTrainingStatusCallback(
        phase="Classification",
        epochs=2,
        display="off",
        status_path=status_path,
        status_interval_seconds=0,
    )
    callback.set_params({"steps": 4, "epochs": 2})
    callback.on_train_begin()
    callback.on_epoch_begin(0)
    callback.on_train_batch_end(0, {"loss": 1.2, "accuracy": 0.6})
    callback.begin_post_epoch_analysis("validation", total_batches=3)

    snapshot = json.loads(status_path.read_text())
    assert snapshot["schema_version"] == 1
    assert snapshot["phase"] == "Classification"
    assert snapshot["progress"] == {
        "epoch": 1,
        "total_epochs": 2,
        "completed_batches": 1,
        "total_batches": 4,
    }
    assert snapshot["metrics"]["current_batch"] == {"loss": 1.2, "accuracy": 0.6}
    assert snapshot["metrics"]["current_epoch_history"] == [
        {"batch": 1, "elapsed_seconds": snapshot["metrics"]["current_epoch_history"][0]["elapsed_seconds"], "loss": 1.2, "accuracy": 0.6}
    ]
    assert snapshot["metrics"]["validation"] == {}
    assert snapshot["external_analysis"]["phase"] == "rich metrics: validation"

    callback.on_epoch_end(0, {"loss": 1.0, "val_loss": 1.1})
    completed = json.loads(status_path.read_text())
    assert completed["metrics"]["last_completed_epoch"] == {"loss": 1.0, "val_loss": 1.1}
    assert completed["metrics"]["last_completed_epoch"]["val_loss"] == 1.1
    assert completed["metrics"]["history"]["loss"] == [1.0]


def test_status_publishes_the_sealed_dashboard_watchlist(tmp_path):
    status_path = tmp_path / "training-status.json"
    callback = RichTrainingStatusCallback(
        phase="Classification",
        display="off",
        status_path=status_path,
        monitoring={"target_enabled": True, "target_metric": "val_macro_f1", "target_value": 0.9},
    )
    callback.on_train_begin()

    assert json.loads(status_path.read_text())["watchlist"] == {
        "target_enabled": True,
        "target_metric": "val_macro_f1",
        "target_value": 0.9,
    }


def test_status_keeps_a_bounded_current_epoch_metric_trace():
    callback = RichTrainingStatusCallback(phase="Classification", display="off")
    callback.set_params({"steps": 500})
    callback.on_train_begin()
    callback.on_epoch_begin(0)
    for batch in range(361):
        callback.on_train_batch_end(batch, {"loss": float(batch), "macro_f1": 0.4})

    trace = callback._status_snapshot()["metrics"]["current_epoch_history"]
    assert len(trace) <= 181
    assert trace[-1]["batch"] == 361
    assert trace[-1]["macro_f1"] == 0.4
