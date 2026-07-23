from __future__ import annotations

import numpy as np
import pytest

from oracle_builder.masking.napari_app import (
    CANDIDATE_MASK_COLOR,
    TRANSPARENT_LABEL_COLOR,
    VALIDATED_MASK_COLOR,
    VALIDATED_MASK_OPACITY,
    _enable_layer_editing,
    _replace_mask_layer,
    _selected_mask_layer,
    _set_label_foreground_color,
    _set_layer_opacity,
    _thumbnail_rgb,
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


def test_replace_mask_layer_copies_source_data():
    source = np.array([[0, 1], [1, 0]], dtype="uint8")
    layer = FakeLayer("validated mask")

    _replace_mask_layer(layer, source)
    source[0, 1] = 0

    assert layer.data[0, 1] == 1
    assert not np.shares_memory(layer.data, source)


def test_layers_replaced_from_same_source_do_not_share_data():
    source = np.array([[0, 1], [1, 0]], dtype="uint8")
    candidate = FakeLayer("candidate mask")
    validated = FakeLayer("validated mask")

    _replace_mask_layer(candidate, source)
    _replace_mask_layer(validated, source)
    candidate.data[0, 1] = 0

    assert validated.data[0, 1] == 1
    assert not np.shares_memory(candidate.data, validated.data)


def test_candidate_mask_layer_editing_can_be_enabled():
    layer = FakeLayer("candidate mask")

    _enable_layer_editing(layer)

    assert layer.editable is True


def test_label_foreground_color_can_be_set_for_refined_mask():
    layer = FakeLayer("validated mask")

    _set_label_foreground_color(layer, VALIDATED_MASK_COLOR)

    assert layer.color == {None: TRANSPARENT_LABEL_COLOR, 0: TRANSPARENT_LABEL_COLOR, 1: VALIDATED_MASK_COLOR}


def test_candidate_mask_color_is_red():
    layer = FakeLayer("candidate mask")

    _set_label_foreground_color(layer, CANDIDATE_MASK_COLOR)

    assert layer.color == {None: TRANSPARENT_LABEL_COLOR, 0: TRANSPARENT_LABEL_COLOR, 1: "#ff0000"}


def test_validated_mask_opacity_defaults_to_half():
    layer = FakeLayer("validated mask")

    _set_layer_opacity(layer, VALIDATED_MASK_OPACITY)

    assert layer.opacity == 0.5


def test_thumbnail_preserves_aspect_ratio_and_letterboxes():
    image = np.tile(np.arange(8, dtype="uint8"), (4, 1))

    thumbnail = _thumbnail_rgb(image, size=16)

    assert thumbnail.shape == (16, 16, 3)
    assert np.all(thumbnail[:4] == 0)
    assert np.any(thumbnail[4:12] > 0)
    assert np.all(thumbnail[12:] == 0)


def test_thumbnail_scales_unit_float_images_to_rgb():
    image = np.linspace(0, 1, 16, dtype="float32").reshape(4, 4)

    thumbnail = _thumbnail_rgb(image, size=4)

    assert thumbnail.shape == (4, 4, 3)
    assert thumbnail[0, 0, 0] == 0
    assert thumbnail[-1, -1, 0] == 255
    assert np.array_equal(thumbnail[..., 0], thumbnail[..., 1])
