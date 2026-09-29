from __future__ import annotations

import numpy as np

from oracle_tools.roi import threshold_mask, validate_mask
from oracle_tools.roi.morphology import connected_components


def test_portable_roi_api_has_no_builder_module_dependency():
    assert threshold_mask.__module__ == "oracle_tools.roi.threshold"
    assert validate_mask.__module__ == "oracle_tools.roi.validation"
    _, sizes = connected_components(np.array([[1, 0], [0, 1]], dtype="uint8"))
    assert sizes == [2]
