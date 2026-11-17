import numpy as np
import pytest
import tensorflow as tf
import uuid

from oracle_builder.inference import InferenceBundle, InferenceItem, ModelReference
from oracle_builder.inference.executor import ClassificationExecutor
from oracle_builder.saving.load_test import SavedModelPredictor


def test_disabled_stratification_ignores_conditioning_default(monkeypatch):
    from types import SimpleNamespace
    from oracle_builder.inference import bundle

    instance = object.__new__(bundle.InferenceBundle)
    instance.config = {
        "data": {"input_shape": [4, 4, 1]},
        "classification": {"stratification": {"enabled": False, "conditioning": {"enabled": True}}},
    }
    monkeypatch.setattr(bundle, "prepare_classification_input", lambda image, *_: image)
    monkeypatch.setattr(bundle, "auxiliary_features_enabled", lambda _: True)
    monkeypatch.setattr(bundle, "auxiliary_feature_vector", lambda *_: np.array([1, 2]))
    item = SimpleNamespace(inputs={"image": SimpleNamespace(values=np.zeros((4, 4, 1)))}, metadata={})

    prepared = instance._prepare_classification(item)
    assert set(prepared) == {"image", "metadata"}
    instance.config["classification"]["stratification"]["enabled"] = True
    with pytest.raises(ValueError, match="active stratum dimension"):
        instance._prepare_classification(item)


def test_bundle_prefers_keras_model(tmp_path, monkeypatch):
    from oracle_builder.inference import bundle

    config = {"run": {"task": "classification", "model": "resnet"},
              "data": {"input_shape": [4, 4, 1]}}
    manifest = {"artifact_id": "artifact", "run_id": "run"}
    model = object()
    calls = []
    monkeypatch.setattr(bundle, "read_run_config", lambda _: config)
    monkeypatch.setattr(bundle, "read_run_manifest", lambda _: manifest)
    def load_model(path, config, *, prefer_savedmodel=False):
        calls.append(prefer_savedmodel)
        return model
    monkeypatch.setattr(bundle, "load_model_for_run", load_model)
    class Executor:
        execution_diagnostics = {}
        def __init__(self, *_):
            pass
    monkeypatch.setattr(bundle, "ClassificationExecutor", Executor)
    assert bundle.InferenceBundle.load(tmp_path).model is model
    assert calls == [False]


def test_savedmodel_warmup_supplies_named_metadata(tmp_path):
    class Classifier(tf.Module):
        @tf.function(input_signature=[
            tf.TensorSpec([None, 4, 4, 1], tf.float32, name="image"),
            tf.TensorSpec([None, 2], tf.float32, name="metadata"),
        ])
        def serve(self, image, metadata):
            values = tf.reduce_mean(image, axis=[1, 2, 3]) + tf.reduce_sum(metadata, axis=1)
            logits = tf.stack([values, -values], axis=1)
            return {"logits": logits, "probabilities": tf.nn.softmax(logits)}

    model = Classifier()
    tf.saved_model.save(model, str(tmp_path / "saved"), signatures={"serving_default": model.serve})
    executor = ClassificationExecutor(SavedModelPredictor(tmp_path / "saved"), (4, 4, 1))

    assert executor.metadata_count == 2
    assert executor.warm(2)["batch_size"] == 2
    result = executor.predict({
        "image": np.zeros((1, 4, 4, 1), dtype="float32"),
        "metadata": np.array([[1.0, 2.0]], dtype="float32"),
    })
    np.testing.assert_allclose(result["logits"], [[3.0, -3.0]])

    # Exercise the serving bundle too: real inputs must use the ROI's source
    # dimensions and fitted feature values, rather than the warmup zeros.
    features = [
        {"name": "area", "source": "roi.bounding_box_area_px", "transform": "none", "mean": 2.0, "scale": 2.0},
        {"name": "aspect", "source": "roi.aspect_ratio", "transform": "none", "mean": 0.0, "scale": 1.0},
    ]
    config = {
        "run": {"task": "classification", "model": "test"},
        "data": {"input_shape": [4, 4, 1]},
        "model": {"auxiliary_features": features, "auxiliary_features_fitted": features},
        "preprocessing": {"resize_mode": "fit_pad", "normalization": "dtype", "rescale": True,
                          "channel_mode": "grayscale", "interpolation": "bilinear"},
    }
    reference = ModelReference(artifact_id=str(uuid.uuid4()), run_id=str(uuid.uuid4()),
                               task="classification", architecture="test", artifact_fingerprint="a" * 64)
    bundle = InferenceBundle(executor.model, config, reference)
    assert bundle.warm_for_serving((1, 2))["resolved_max_batch_size"] == 2
    results = bundle.predict_batch([InferenceItem.from_array(np.zeros((2, 4), dtype="uint8"))])
    assert results.counts["succeeded"] == 1
    np.testing.assert_allclose(results.results[0].output["logits"], [5.0, -5.0])


def test_keras_warmup_supplies_named_metadata():
    from oracle_builder.registry import get_model_builder

    config = {
        "run": {"task": "classification", "model": "simple_cnn"},
        "data": {"input_shape": [8, 8, 1], "num_classes": 2},
        "model": {"base_filters": 2, "dropout": 0.0, "embedding_dim": 4,
                  "auxiliary_features": [{"name": "area"}, {"name": "aspect"}]},
    }
    executor = ClassificationExecutor(get_model_builder("simple_cnn")(config), (8, 8, 1))
    assert executor.metadata_count == 2
    assert executor.warm(2)["batch_size"] == 2

