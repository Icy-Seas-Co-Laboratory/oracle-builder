import io

from oracle_builder.training.callbacks import build_callbacks
from oracle_builder.training.status import RichTrainingStatusCallback


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
