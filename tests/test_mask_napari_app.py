from __future__ import annotations

import pytest

from oracle_builder.masking.napari_app import _viewer_theme_for_background


def test_viewer_background_maps_to_napari_theme():
    assert _viewer_theme_for_background("Black") == "dark"
    assert _viewer_theme_for_background("white") == "light"


def test_unknown_viewer_background_is_rejected():
    with pytest.raises(ValueError):
        _viewer_theme_for_background("gray")

