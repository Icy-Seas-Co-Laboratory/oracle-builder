from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from oracle_builder.masking.morphology import fill_holes, keep_largest_component, remove_small_objects
from oracle_builder.masking.sqlite_io import create_or_update_image_sample, load_sample, open_database, save_mask_annotation
from oracle_builder.masking.threshold import invert_display_image, threshold_mask
from oracle_builder.masking.validation import validate_mask

CANDIDATE_MASK_COLOR = "#ff0000"
VALIDATED_MASK_COLOR = "#20c997"
VALIDATED_MASK_OPACITY = 0.5
TRANSPARENT_LABEL_COLOR = [0.0, 0.0, 0.0, 0.0]


def _empty_mask_for(image: np.ndarray) -> np.ndarray:
    return np.zeros(np.asarray(image).shape[:2], dtype="uint8")


def _viewer_theme_for_background(background: str) -> str:
    normalized = background.strip().lower()
    if normalized == "white":
        return "light"
    if normalized == "black":
        return "dark"
    raise ValueError("Viewer background must be 'black' or 'white'")


def _is_mask_layer(layer: Any) -> bool:
    if layer is None or not hasattr(layer, "data"):
        return False
    name = getattr(layer, "name", "")
    if name in {"candidate mask", "validated mask"}:
        return True
    return layer.__class__.__name__ == "Labels"


def _selected_mask_layer(viewer: Any, default_layer: Any) -> Any:
    selection = getattr(getattr(viewer, "layers", None), "selection", None)
    active_layer = getattr(selection, "active", None)
    return active_layer if _is_mask_layer(active_layer) else default_layer


def _replace_mask_layer(layer: Any, new_mask: np.ndarray, reveal: bool = False) -> None:
    layer.data = np.array(new_mask, dtype="uint8", copy=True)
    if reveal and hasattr(layer, "visible"):
        layer.visible = True
    refresh = getattr(layer, "refresh", None)
    if callable(refresh):
        refresh()


def _enable_layer_editing(layer: Any) -> None:
    if hasattr(layer, "editable"):
        layer.editable = True


def _set_label_foreground_color(layer: Any, color: str) -> None:
    colors = {None: TRANSPARENT_LABEL_COLOR, 0: TRANSPARENT_LABEL_COLOR, 1: color}
    if hasattr(layer, "color"):
        layer.color = colors
        return
    if hasattr(layer, "colormap"):
        layer.colormap = colors


def _set_layer_opacity(layer: Any, opacity: float) -> None:
    if hasattr(layer, "opacity"):
        layer.opacity = float(opacity)


def _provenance(
    threshold: float,
    invert: bool,
    blur_method: str,
    blur_kernel_size: int,
    morphology_state: dict[str, Any],
) -> dict[str, Any]:
    return {
        "tool": "oracle-builder mask_builder",
        "tool_version": "0.1.0",
        "source": "manual_threshold_plus_edit",
        "threshold": threshold,
        "inverted": invert,
        "blur": {"method": blur_method, "kernel_size": blur_kernel_size},
        "morphology": morphology_state,
        "manual_edits": True,
        "saved_from": "napari",
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }


def launch_mask_builder_app(
    image: np.ndarray,
    sample_uuid: str,
    db_path: str | None = None,
    output_db_path: str | None = None,
    initial_mask: np.ndarray | None = None,
    initial_candidate_mask: np.ndarray | None = None,
    initial_metadata: dict[str, Any] | None = None,
    mask_encoding: str = "png",
    read_only: bool = False,
    sample_queue: list[dict] | None = None,
    sample_loader: Callable[[dict], dict[str, Any]] | None = None,
    debug: bool = False,
) -> None:
    try:
        import napari
        from qtpy.QtWidgets import (
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QFormLayout,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QPushButton,
            QSpinBox,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )
    except Exception as exc:  # pragma: no cover - depends on optional GUI stack
        raise RuntimeError(
            "Launching mask_builder requires GUI dependencies. Install them with "
            "`python3 -m pip install -r requirements-gui.txt`."
        ) from exc

    current: dict[str, Any] = {
        "image": np.asarray(image),
        "uuid": sample_uuid,
        "index": 0,
        "morphology": {
            "fill_holes_applied": False,
            "remove_small_objects_min_size": None,
            "keep_largest_component_applied": False,
        },
        "last_validation": None,
        "metadata": initial_metadata or {},
    }
    queue = sample_queue or [{"uuid": sample_uuid}]
    output_path = output_db_path or db_path
    candidate_mask = (
        np.array(initial_candidate_mask, dtype="uint8", copy=True)
        if initial_candidate_mask is not None
        else np.array(initial_mask, dtype="uint8", copy=True)
        if initial_mask is not None
        else _empty_mask_for(current["image"])
    )
    mask = np.array(initial_mask, dtype="uint8", copy=True) if initial_mask is not None else candidate_mask.copy()

    viewer = napari.Viewer(title=f"oracle-builder mask builder: {sample_uuid}")
    viewer.theme = _viewer_theme_for_background("black")
    image_layer = viewer.add_image(current["image"], name="image")
    candidate_layer = viewer.add_labels(candidate_mask.copy(), name="candidate mask", visible=False)
    _enable_layer_editing(candidate_layer)
    _set_label_foreground_color(candidate_layer, CANDIDATE_MASK_COLOR)
    labels_layer = viewer.add_labels(mask.copy(), name="validated mask")
    _enable_layer_editing(labels_layer)
    _set_label_foreground_color(labels_layer, VALIDATED_MASK_COLOR)
    _set_layer_opacity(labels_layer, VALIDATED_MASK_OPACITY)

    panel = QWidget()
    layout = QVBoxLayout()
    panel.setLayout(layout)
    title = QLabel(f"Sample: {sample_uuid}")
    layout.addWidget(title)

    form = QFormLayout()
    threshold_value = QDoubleSpinBox()
    threshold_value.setRange(0.0, 1.0)
    threshold_value.setSingleStep(0.01)
    threshold_value.setValue(0.5)
    invert_value = QCheckBox()
    blur_method = QComboBox()
    blur_method.addItems(["none", "gaussian", "median"])
    blur_kernel = QSpinBox()
    blur_kernel.setRange(0, 99)
    blur_kernel.setValue(0)
    small_object_size = QSpinBox()
    small_object_size.setRange(1, 1000000)
    small_object_size.setValue(25)
    viewer_background = QComboBox()
    viewer_background.addItems(["Black", "White"])
    notes = QLineEdit()
    form.addRow("Threshold", threshold_value)
    form.addRow("Invert image", invert_value)
    form.addRow("Blur", blur_method)
    form.addRow("Blur kernel", blur_kernel)
    form.addRow("Min object size", small_object_size)
    form.addRow("Background", viewer_background)
    form.addRow("Notes", notes)
    layout.addLayout(form)

    validation_text = QTextEdit()
    validation_text.setReadOnly(True)
    validation_text.setMinimumHeight(150)

    def set_status(message: str, report: dict[str, Any] | None = None) -> None:
        if report is not None:
            validation_text.setPlainText(message + "\n" + json.dumps(report, indent=2, default=str))
        else:
            validation_text.setPlainText(message)
        if debug:
            print(message)

    def set_viewer_background(background: str) -> None:
        viewer.theme = _viewer_theme_for_background(background)
        set_status(f"Viewer background set to {background.lower()}.")

    viewer_background.currentTextChanged.connect(set_viewer_background)

    def displayed_image() -> np.ndarray:
        return invert_display_image(current["image"]) if invert_value.isChecked() else current["image"]

    def refresh_image_layer() -> None:
        image_layer.data = displayed_image()
        image_layer.refresh()

    invert_value.stateChanged.connect(lambda _state: refresh_image_layer())

    def current_mask() -> np.ndarray:
        return (np.asarray(labels_layer.data) > 0).astype("uint8")

    def current_candidate_mask() -> np.ndarray:
        return (np.asarray(candidate_layer.data) > 0).astype("uint8")

    def selected_mask_layer():
        return _selected_mask_layer(viewer, labels_layer)

    def current_selected_mask() -> np.ndarray:
        return (np.asarray(selected_mask_layer().data) > 0).astype("uint8")

    def replace_mask(new_mask: np.ndarray) -> None:
        _replace_mask_layer(labels_layer, new_mask)

    def replace_candidate_mask(new_mask: np.ndarray) -> None:
        _replace_mask_layer(candidate_layer, new_mask)

    def replace_selected_mask(new_mask: np.ndarray):
        layer = selected_mask_layer()
        _replace_mask_layer(layer, new_mask, reveal=True)
        return layer

    def apply_threshold() -> None:
        new_mask = threshold_mask(
            current["image"],
            threshold_value.value(),
            invert=invert_value.isChecked(),
            blur_method=blur_method.currentText(),
            kernel_size=blur_kernel.value(),
        )
        layer = replace_selected_mask(new_mask)
        if layer is labels_layer:
            current["morphology"] = {
                "fill_holes_applied": False,
                "remove_small_objects_min_size": None,
                "keep_largest_component_applied": False,
            }
        layer_name = getattr(layer, "name", "selected mask")
        set_status(f"Applied threshold. This replaced {layer_name}.")

    def do_fill_holes() -> None:
        layer = replace_selected_mask(fill_holes(current_selected_mask()))
        if layer is labels_layer:
            current["morphology"]["fill_holes_applied"] = True
        layer_name = getattr(layer, "name", "selected mask")
        set_status(f"Filled holes in {layer_name}.")

    def do_remove_small_objects() -> None:
        min_size = small_object_size.value()
        layer = replace_selected_mask(remove_small_objects(current_selected_mask(), min_size))
        if layer is labels_layer:
            current["morphology"]["remove_small_objects_min_size"] = min_size
        layer_name = getattr(layer, "name", "selected mask")
        set_status(f"Removed foreground components smaller than {min_size} pixels from {layer_name}.")

    def do_keep_largest() -> None:
        layer = replace_selected_mask(keep_largest_component(current_selected_mask()))
        if layer is labels_layer:
            current["morphology"]["keep_largest_component_applied"] = True
        layer_name = getattr(layer, "name", "selected mask")
        set_status(f"Kept largest foreground component in {layer_name}.")

    def do_validate() -> dict[str, Any]:
        report = validate_mask(current_mask(), current["image"])
        current["last_validation"] = report
        status = "Mask is valid." if report["valid"] else "Mask is invalid."
        set_status(status, report)
        return report

    def load_queue_index(index: int) -> bool:
        if index >= len(queue):
            set_status("No more samples in queue.")
            return False
        sample_info = queue[index]
        if sample_loader is not None:
            sample = sample_loader(sample_info)
        elif db_path:
            with open_database(db_path) as conn:
                sample = load_sample(conn, sample_info["uuid"])
        else:
            set_status("No loader is configured for the next sample.")
            return False
        current["image"] = sample["image"]
        current["uuid"] = sample["uuid"]
        current["index"] = index
        current["metadata"] = sample.get("metadata", {})
        current["last_validation"] = None
        current["morphology"] = {
            "fill_holes_applied": False,
            "remove_small_objects_min_size": None,
            "keep_largest_component_applied": False,
        }
        image_layer.data = current["image"]
        refresh_image_layer()
        loaded_candidate = sample.get("candidate_mask")
        if loaded_candidate is None:
            loaded_candidate = sample["mask"] if sample["mask"] is not None else _empty_mask_for(current["image"])
        replace_candidate_mask(loaded_candidate)
        replace_mask(sample["mask"] if sample["mask"] is not None else current_candidate_mask().copy())
        title.setText(f"Sample: {current['uuid']}")
        viewer.title = f"oracle-builder mask builder: {current['uuid']}"
        set_status(f"Loaded sample {index + 1} of {len(queue)}.")
        return True

    def save_current(advance: bool = False) -> None:
        if read_only:
            set_status("Read-only mode: no SQLite writes were performed.")
            return
        if output_path is None:
            set_status("No output database path is configured.")
            return
        report = current["last_validation"] or do_validate()
        if not report["valid"]:
            set_status("Mask is invalid; fix validation errors before saving.", report)
            return
        with open_database(output_path) as conn:
            create_or_update_image_sample(
                conn,
                current["uuid"],
                current["image"],
                "png",
                current.get("metadata") or {"created_by": "oracle-builder mask_builder", "source": "mask_builder_runtime_copy"},
                candidate_mask=current_candidate_mask(),
            )
            annotation_id = save_mask_annotation(
                conn,
                current["uuid"],
                current_mask(),
                mask_encoding,
                method="manual_threshold_plus_edit",
                parameters=_provenance(
                    threshold_value.value(),
                    invert_value.isChecked(),
                    blur_method.currentText(),
                    blur_kernel.value(),
                    current["morphology"],
                ),
                validation=report,
                accepted=True,
                notes=notes.text() or None,
            )
        set_status(f"Saved annotation {annotation_id}.")
        if advance:
            load_queue_index(current["index"] + 1)

    def skip_current() -> None:
        load_queue_index(current["index"] + 1)

    def add_button(label: str, callback) -> QPushButton:
        button = QPushButton(label)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    add_button("Apply threshold", apply_threshold)
    add_button("Fill holes", do_fill_holes)
    add_button("Remove small objects", do_remove_small_objects)
    add_button("Keep largest component", do_keep_largest)
    add_button("Validate", do_validate)

    save_buttons = QHBoxLayout()
    save_button = QPushButton("Save")
    save_button.clicked.connect(lambda: save_current(False))
    save_next_button = QPushButton("Save and next")
    save_next_button.clicked.connect(lambda: save_current(True))
    skip_button = QPushButton("Skip")
    skip_button.clicked.connect(skip_current)
    save_buttons.addWidget(save_button)
    save_buttons.addWidget(save_next_button)
    save_buttons.addWidget(skip_button)
    layout.addLayout(save_buttons)
    if read_only:
        save_button.setEnabled(False)
        save_next_button.setEnabled(False)
    if len(queue) <= 1:
        save_next_button.setEnabled(False)
        skip_button.setEnabled(False)

    layout.addWidget(validation_text)
    viewer.window.add_dock_widget(panel, area="right", name="Mask Builder")
    set_status("Ready. Paint label 1 for foreground and label 0 to erase.")
    napari.run()
