from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from oracle_builder.masking.morphology import fill_holes, keep_largest_component, remove_small_objects
from oracle_builder.masking.sqlite_io import (
    create_or_update_image_sample,
    delete_image_sample,
    duplicate_image_sample,
    load_sample,
    open_database,
    save_mask_annotation,
)
from oracle_builder.masking.threshold import invert_display_image, threshold_mask
from oracle_builder.masking.validation import validate_mask

CANDIDATE_MASK_COLOR = "#ff0000"
VALIDATED_MASK_COLOR = "#20c997"
VALIDATED_MASK_OPACITY = 0.5
TRANSPARENT_LABEL_COLOR = [0.0, 0.0, 0.0, 0.0]


def _thumbnail_rgb(image: np.ndarray, size: int = 112) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[-1] >= 3:
        rgb = array[..., :3].astype("float32")
    elif array.ndim == 3 and array.shape[-1] == 1:
        rgb = np.repeat(array.astype("float32"), 3, axis=-1)
    elif array.ndim == 2:
        rgb = np.repeat(array[..., None].astype("float32"), 3, axis=-1)
    else:
        raise ValueError(f"Unsupported thumbnail image shape: {array.shape}")
    finite = rgb[np.isfinite(rgb)]
    if finite.size == 0:
        rgb = np.zeros_like(rgb)
    elif finite.min() >= 0 and finite.max() <= 1:
        rgb = rgb * 255
    else:
        lo, hi = np.percentile(finite, [1, 99])
        rgb = np.zeros_like(rgb) if hi <= lo else (rgb - lo) * (255 / (hi - lo))
    source = Image.fromarray(np.clip(rgb, 0, 255).astype("uint8"), mode="RGB")
    source.thumbnail((size, size), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), "black")
    canvas.paste(source, ((size - source.width) // 2, (size - source.height) // 2))
    return np.asarray(canvas)


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
        from qtpy.QtCore import QSize, Qt
        from qtpy.QtGui import QIcon, QImage, QKeySequence, QPixmap
        try:
            from qtpy.QtGui import QShortcut
        except ImportError:  # Qt5 exposes QShortcut from QtWidgets.
            from qtpy.QtWidgets import QShortcut
        from qtpy.QtWidgets import (
            QButtonGroup,
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QFormLayout,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMessageBox,
            QPushButton,
            QScrollArea,
            QSpinBox,
            QTextEdit,
            QToolButton,
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
    sample_cache: dict[int, dict[str, Any]] = {
        0: {
            "uuid": sample_uuid,
            "image": current["image"],
            "mask": initial_mask,
            "candidate_mask": initial_candidate_mask,
            "metadata": current["metadata"],
        }
    }
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
        if index < 0 or index >= len(queue):
            set_status("No more samples in queue.")
            return False
        sample_info = queue[index]
        if index in sample_cache:
            sample = sample_cache[index]
        elif sample_loader is not None:
            sample = sample_loader(sample_info)
        elif db_path:
            with open_database(db_path) as conn:
                sample = load_sample(conn, sample_info["uuid"])
        else:
            set_status("No loader is configured for the next sample.")
            return False
        sample_cache[index] = sample
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
        if "thumbnail_buttons" in navigation:
            navigation["thumbnail_buttons"][index].setChecked(True)
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

    def delete_current() -> None:
        if read_only:
            set_status("Read-only mode: no SQLite writes were performed.")
            return
        if output_path is None:
            set_status("No output database path is configured.")
            return
        sample_id = current["uuid"]
        choice = QMessageBox.question(
            panel,
            "Delete image",
            f"Delete image {sample_id} and its mask annotations from the database?\n\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if choice != QMessageBox.Yes:
            return
        try:
            with open_database(output_path, create=False) as conn:
                delete_image_sample(conn, sample_id)
        except Exception as exc:
            set_status(f"Could not delete image {sample_id}: {exc}")
            return
        deleted_index = current["index"]
        del queue[deleted_index]
        sample_cache.clear()
        if not queue:
            title.setText("No images remaining")
            viewer.title = "oracle-builder mask builder: no images remaining"
            rebuild_thumbnail_buttons()
            set_status(f"Deleted image {sample_id}. No images remain in the queue.")
            return
        next_index = min(deleted_index, len(queue) - 1)
        rebuild_thumbnail_buttons()
        load_queue_index(next_index)
        set_status(f"Deleted image {sample_id}.")

    def duplicate_current() -> None:
        if read_only:
            set_status("Read-only mode: no SQLite writes were performed.")
            return
        if output_path is None:
            set_status("No output database path is configured.")
            return
        source_index = current["index"]
        source_id = current["uuid"]
        try:
            with open_database(output_path, create=False) as conn:
                duplicate_id = duplicate_image_sample(conn, source_id)
        except Exception as exc:
            set_status(f"Could not duplicate image {source_id}: {exc}")
            return
        queue.insert(source_index + 1, {"uuid": duplicate_id})
        sample_cache.clear()
        rebuild_thumbnail_buttons()
        load_queue_index(source_index + 1)
        set_status(f"Duplicated image {source_id} as {duplicate_id}.")

    def add_button(label: str, callback) -> QPushButton:
        button = QPushButton(label)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    add_button("Apply threshold (A)", apply_threshold)
    add_button("Fill holes (F)", do_fill_holes)
    add_button("Remove small objects (R)", do_remove_small_objects)
    add_button("Keep largest component (L)", do_keep_largest)
    add_button("Validate", do_validate)

    save_buttons = QHBoxLayout()
    save_button = QPushButton("Save")
    save_button.clicked.connect(lambda: save_current(False))
    save_next_button = QPushButton("Save and next")
    save_next_button.clicked.connect(lambda: save_current(True))
    skip_button = QPushButton("Skip")
    skip_button.clicked.connect(skip_current)
    delete_button = QPushButton("Delete image")
    delete_button.clicked.connect(delete_current)
    duplicate_button = QPushButton("Duplicate image")
    duplicate_button.clicked.connect(duplicate_current)
    save_buttons.addWidget(save_button)
    save_buttons.addWidget(save_next_button)
    save_buttons.addWidget(skip_button)
    save_buttons.addWidget(delete_button)
    save_buttons.addWidget(duplicate_button)
    layout.addLayout(save_buttons)
    if read_only:
        save_button.setEnabled(False)
        save_next_button.setEnabled(False)
        delete_button.setEnabled(False)
        duplicate_button.setEnabled(False)
    if len(queue) <= 1:
        save_next_button.setEnabled(False)
        skip_button.setEnabled(False)

    layout.addWidget(validation_text)
    viewer.window.add_dock_widget(panel, area="right", name="Mask Builder")

    navigation: dict[str, Any] = {}
    thumbnail_panel = QWidget()
    thumbnail_panel_layout = QVBoxLayout()
    thumbnail_panel.setLayout(thumbnail_panel_layout)
    position_label = QLabel(f"ROI 1 of {len(queue)}")
    thumbnail_panel_layout.addWidget(position_label)
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    strip = QWidget()
    strip_layout = QHBoxLayout()
    strip.setLayout(strip_layout)
    button_group = QButtonGroup(strip)
    button_group.setExclusive(True)
    thumbnail_buttons: list[Any] = []

    def load_thumbnail_sample(index: int) -> dict[str, Any]:
        info = queue[index]
        if index == current["index"]:
            return {"image": current["image"]}
        if index in sample_cache:
            return sample_cache[index]
        if sample_loader is not None:
            sample = sample_loader(info)
            sample_cache[index] = sample
            return sample
        if db_path:
            with open_database(db_path) as conn:
                sample = load_sample(conn, info["uuid"])
            sample_cache[index] = sample
            return sample
        raise ValueError("No sample loader is configured")

    def navigate_to(index: int) -> None:
        if load_queue_index(index):
            position_label.setText(f"ROI {index + 1} of {len(queue)}")

    def rebuild_thumbnail_buttons() -> None:
        for button in thumbnail_buttons:
            button_group.removeButton(button)
            strip_layout.removeWidget(button)
            button.deleteLater()
        thumbnail_buttons.clear()
        for index, info in enumerate(queue):
            button = QToolButton()
            button.setCheckable(True)
            button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            button.setIconSize(QSize(112, 112))
            button.setText(str(index + 1))
            button.setToolTip(f"ROI {index + 1}: {info.get('uuid', index)}")
            try:
                thumbnail = np.ascontiguousarray(_thumbnail_rgb(load_thumbnail_sample(index)["image"]))
                height, width, _channels = thumbnail.shape
                qimage = QImage(thumbnail.data, width, height, width * 3, QImage.Format_RGB888).copy()
                button.setIcon(QIcon(QPixmap.fromImage(qimage)))
            except Exception as exc:
                if debug:
                    print(f"Could not load thumbnail for {info.get('uuid', index)}: {exc}")
            button.clicked.connect(lambda _checked=False, selected=index: navigate_to(selected))
            button_group.addButton(button, index)
            strip_layout.addWidget(button)
            thumbnail_buttons.append(button)
        if thumbnail_buttons:
            selected_index = min(current["index"], len(thumbnail_buttons) - 1)
            thumbnail_buttons[selected_index].setChecked(True)

    rebuild_thumbnail_buttons()
    strip_layout.addStretch(1)
    scroll.setWidget(strip)
    thumbnail_panel_layout.addWidget(scroll)
    navigation["thumbnail_buttons"] = thumbnail_buttons
    viewer.window.add_dock_widget(thumbnail_panel, area="bottom", name="ROI Navigator")

    shortcuts: list[Any] = []

    def add_shortcut(key: str, callback: Callable[[], None]) -> None:
        shortcut = QShortcut(QKeySequence(key), panel)
        shortcut.setContext(Qt.ApplicationShortcut)
        shortcut.activated.connect(callback)
        shortcuts.append(shortcut)

    add_shortcut("Left", lambda: navigate_to(current["index"] - 1))
    add_shortcut("Right", lambda: navigate_to(current["index"] + 1))
    add_shortcut("A", apply_threshold)
    add_shortcut("F", do_fill_holes)
    add_shortcut("R", do_remove_small_objects)
    add_shortcut("L", do_keep_largest)
    navigation["shortcuts"] = shortcuts

    set_status("Ready. Paint label 1 for foreground and label 0 to erase.")
    napari.run()
