from __future__ import annotations

import pytest

from oracle_builder.masking.napari_app import (
    CANDIDATE_MASK_COLOR,
    VALIDATED_MASK_COLOR,
    VALIDATED_MASK_OPACITY,
    _enable_layer_editing,
    _replace_mask_layer,
    _selected_mask_layer,
    _set_label_foreground_color,
    _set_layer_opacity,
    _viewer_theme_for_background,
)


class FakeLayer:
    def __init__(self, name, data=None, visible=False):
        self.name = name
        self.data = data
        self.visible = visible
        self.editable = False
        self.color = None
        self.opacity = 0.7
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


def test_candidate_mask_layer_editing_can_be_enabled():
    layer = FakeLayer("candidate mask")

    _enable_layer_editing(layer)

    assert layer.editable is True


def test_label_foreground_color_can_be_set_for_refined_mask():
    layer = FakeLayer("validated mask")

    _set_label_foreground_color(layer, VALIDATED_MASK_COLOR)

    assert layer.color == {1: VALIDATED_MASK_COLOR}


def test_candidate_mask_color_is_red():
    layer = FakeLayer("candidate mask")

    _set_label_foreground_color(layer, CANDIDATE_MASK_COLOR)

    assert layer.color == {1: "#ff0000"}


def test_validated_mask_opacity_defaults_to_half():
    layer = FakeLayer("validated mask")

    _set_layer_opacity(layer, VALIDATED_MASK_OPACITY)

    assert layer.opacity == 0.5
