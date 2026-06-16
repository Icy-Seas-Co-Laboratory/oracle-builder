from __future__ import annotations

import pytest

from oracle_builder.masking.napari_app import _replace_mask_layer, _selected_mask_layer, _viewer_theme_for_background


class FakeLayer:
    def __init__(self, name, data=None, visible=False):
        self.name = name
        self.data = data
        self.visible = visible
        self.refreshed = False

    def refresh(self):
        self.refreshed = True


class FakeSelection:
    def __init__(self, active):
        self.active = active


class FakeLayers:
    def __init__(self, active):
        self.selection = FakeSelection(active)


class FakeViewer:
    def __init__(self, active):
        self.layers = FakeLayers(active)


def test_viewer_background_maps_to_napari_theme():
    assert _viewer_theme_for_background("Black") == "dark"
    assert _viewer_theme_for_background("white") == "light"


def test_unknown_viewer_background_is_rejected():
    with pytest.raises(ValueError):
        _viewer_theme_for_background("gray")


def test_selected_candidate_mask_layer_can_be_targeted():
    default = FakeLayer("validated mask")
    candidate = FakeLayer("candidate mask")

    assert _selected_mask_layer(FakeViewer(candidate), default) is candidate


def test_selected_image_layer_falls_back_to_validated_mask():
    default = FakeLayer("validated mask")
    image = FakeLayer("image")

    assert _selected_mask_layer(FakeViewer(image), default) is default


def test_replace_mask_layer_reveals_and_refreshes_layer():
    layer = FakeLayer("candidate mask", visible=False)

    _replace_mask_layer(layer, [[0, 1], [1, 0]], reveal=True)

    assert layer.visible is True
    assert layer.refreshed is True
    assert layer.data.dtype == "uint8"
