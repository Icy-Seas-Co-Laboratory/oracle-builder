from __future__ import annotations

import numpy as np

from oracle_builder.posthoc.head import posthoc_output


def test_posthoc_output_keeps_neural_class_index_alignment():
    class Estimator:
        classes_ = np.array([0, 2])

        def predict_proba(self, values):
            assert values.shape == (1, 3)
            return np.array([[0.2, 0.8]])

    output = posthoc_output(Estimator(), np.array([1, 2, 3]), 3)
    assert output["class_index"] == 2
    assert output["probabilities"] == [0.2, 0.0, 0.8]
    assert output["confidence"] == 0.8


def test_posthoc_output_preserves_configured_class_count_when_highest_class_was_unseen():
    class Estimator:
        classes_ = np.array([0, 1])
        oracle_posthoc_class_count_ = 3

        def predict_proba(self, values):
            return np.array([[0.4, 0.6]])

    output = posthoc_output(Estimator(), np.array([1, 2, 3]))
    assert output["probabilities"] == [0.4, 0.6, 0.0]
    assert output["classes"] == [0, 1, 2]
    assert output["trained_classes"] == [0, 1]
