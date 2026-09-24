import io

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


def test_rich_board_shows_every_metric_with_alternating_metric_rows():
    callback = RichTrainingStatusCallback(phase="SSL", epochs=1, display="off")
    callback._metrics = {f"metric_{index}": float(index) for index in range(20)}

    board = callback._board()
    metrics = board.renderable.renderables[1]

    assert len(metrics.rows) == 21  # Header plus every supplied metric.
    assert metrics.row_styles == ["", "on grey15"]
