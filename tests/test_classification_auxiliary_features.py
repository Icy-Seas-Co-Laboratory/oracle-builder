from __future__ import annotations

import math

import numpy as np

from oracle_builder.classification.metadata import vector
from oracle_builder.classification.stratification import stratum_for_shape


def test_log_roi_area_is_taken_from_original_shape_and_standardized():
    config = {
        "model": {
            "auxiliary_features_fitted": [
                {
                    "name": "log_roi_area",
                    "source": "roi.bounding_box_area_px",
                    "transform": "log",
                    "mean": 5.0,
                    "scale": 2.0,
                    "standardize": True,
                }
            ]
        }
    }
    values = vector(config, {}, (20, 34))
    assert values.shape == (1,)
    assert values[0] == np.float32((math.log(680) - 5.0) / 2.0)


def test_resolution_stratum_uses_smallest_containing_maximum_dimension():
    dimensions = [32, 64, 128]
    assert stratum_for_shape((20, 34), dimensions) == 64
    assert stratum_for_shape((32, 1), dimensions) == 32
    assert stratum_for_shape((65, 16), dimensions) == 128
    assert stratum_for_shape((500, 20), dimensions) == 128
