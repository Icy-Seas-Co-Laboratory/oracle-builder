import json

from tensorflow import keras

from oracle_builder.classification.stratified_training import _advance_early_stopping
from oracle_builder.progress import PostTrainingProgress
from oracle_builder.training.callbacks import build_callbacks
from oracle_builder.worker.pull import training_progress_payload


class _Model:
    stop_training = False

    def get_weights(self):
        return []

    def set_weights(self, weights):
        pass


def _early_callbacks(tmp_path, *, saved=None):
    callbacks = build_callbacks(
        {"training": {"epochs": 8, "display": "off"}, "recovery": {"enabled": False},
         "callbacks": {"early_stopping": True, "early_stopping_patience": 3,
                       "checkpoint_monitor": "val_loss"}},
        tmp_path, tmp_path / "training.sqlite", "run-1", resume_callback_state=saved,
    )
    early = next(callback for callback in callbacks if isinstance(callback, keras.callbacks.EarlyStopping))
    model = _Model()
    for callback in callbacks:
        if callback is early or callback.__class__.__name__ == "_CallbackStateRestorer":
            callback.set_model(model)
            callback.on_train_begin()
    return early, model, callbacks


def test_early_stopping_ignores_early_low_loss_and_waits_after_warmup(tmp_path):
    early, model, _ = _early_callbacks(tmp_path)
    assert early.start_from_epoch == 2  # User-facing epoch 3.
    for epoch, loss in enumerate([0.01, 0.02, 1.0, 1.1, 1.2, 1.3]):
        early.on_epoch_end(epoch, {"val_loss": loss})
        assert model.stop_training is (epoch == 5)
    assert early.best == 1.0
    assert early.best_epoch == 2
    assert early.stopped_epoch == 5


def test_resume_discards_pre_warmup_legacy_baseline(tmp_path):
    early, model, _ = _early_callbacks(tmp_path, saved={"early_stopping": {
        "best": 0.01, "best_epoch": 0, "wait": 2, "best_weights": [], "stopped_epoch": 0,
    }})
    assert early.wait == 0
    early.on_epoch_end(2, {"val_loss": 1.0})
    assert early.best == 1.0
    assert model.stop_training is False


def test_stratified_warmup_and_patience_count_parent_epochs():
    state = {}
    for epoch, value in [(5, 0.01), (10, 0.02), (18, 1.0), (23, 1.1), (33, 1.2)]:
        stopped = _advance_early_stopping(state, epoch=epoch, value=value,
                                          mode="min", patience=18, eligible=True)
        assert stopped is False
    assert state["early_stopping_best"] == 1.0
    assert state["early_stopping_best_epoch"] == 18
    assert _advance_early_stopping(dict(state), epoch=38, value=1.3,
                                   mode="min", patience=18, eligible=True) is True


def test_post_training_stage_preserves_metrics_and_reports_item_progress(tmp_path):
    path = tmp_path / "training-status.json"
    path.write_text(json.dumps({"progress": {"epoch": 21, "total_epochs": 25},
                                "history": [{"epoch": 21, "val_loss": 2.1}],
                                "metrics": {"last_completed_epoch": {"val_loss": 2.1}}}))
    status = PostTrainingProgress(3, status_path=path, planned_epochs=25)
    with status.stage("Building classification evidence"):
        status.update_items(64, 100)
        raw = json.loads(path.read_text())
        assert raw["post_training"]["stopped_early"] is True
        assert raw["post_training"]["stage_index"] == 1
        assert raw["history"] == [{"epoch": 21, "val_loss": 2.1}]
    projection = training_progress_payload(path)
    assert projection["post_training"]["status"] == "completed"
    assert projection["post_training"]["completed_epochs"] == 21
    assert projection["history"] == [{"epoch": 21, "val_loss": 2.1}]
