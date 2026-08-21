from __future__ import annotations

import sqlite3

from oracle_data_contracts.datasets import (
    initialize_database,
    normalize_item_geometry,
    read_dataset_info,
    validate_database,
)
from oracle_data_contracts.datasets.repository import SQLiteDatasetRepository


def test_pelagia_geometry_preserves_distinct_object_and_crop_boxes():
    geometry = normalize_item_geometry(
        {
            "pelagia": {
                "spatial": {
                    "coordinate_space": "source_frame_pixels",
                    "bbox": {"x": 12, "y": 23, "w": 4, "h": 5},
                    "crop_bbox": {"x": 10, "y": 20, "w": 9, "h": 11},
                }
            }
        }
    )

    assert geometry is not None
    assert geometry["coordinate_space"] == "source_frame_pixels"
    assert geometry["bbox"] == {"x": 12, "y": 23, "w": 4, "h": 5}
    assert geometry["crop_bbox"] == {"x": 10, "y": 20, "w": 9, "h": 11}
    assert geometry["metadata"]["fallback"] is None


def test_single_compatible_rectangle_defines_bbox_and_crop():
    geometry = normalize_item_geometry({"roi_bbox": [7, 8, 9, 10]})

    assert geometry is not None
    assert geometry["bbox"] == geometry["crop_bbox"] == {"x": 7, "y": 8, "w": 9, "h": 10}
    assert geometry["metadata"]["fallback"] == "crop_from_bbox"


def test_repository_uses_linked_image_extent_when_roi_geometry_is_absent():
    connection = sqlite3.connect(":memory:")
    initialize_database(connection, "classification")
    repository = SQLiteDatasetRepository(connection)
    asset = repository.add_asset(b"image", encoding="raw", shape=[6, 8, 3])
    item_id = repository.add_item(source_key="roi/1", metadata={"pelagia": {"run_id": "run-1"}})
    label_id = repository.add_classification_label(0, "unlabeled")
    repository.add_classification_item(item_id=item_id, image_asset_id=asset.asset_id, label_id=label_id)

    row = connection.execute(
        """SELECT coordinate_space,bbox_x,bbox_y,bbox_w,bbox_h,
           crop_bbox_x,crop_bbox_y,crop_bbox_w,crop_bbox_h
           FROM item_geometry WHERE item_id=?""",
        (item_id,),
    ).fetchone()
    assert tuple(row) == ("source_frame_pixels", 0, 0, 8, 6, 0, 0, 8, 6)
    assert validate_database(connection)["valid"]


def test_schema_1_4_migration_backfills_item_geometry_from_asset_shape():
    connection = sqlite3.connect(":memory:")
    initialize_database(connection, "classification")
    repository = SQLiteDatasetRepository(connection)
    asset = repository.add_asset(b"image", encoding="raw", shape=[5, 7])
    item_id = repository.add_item(source_key="roi/legacy")
    label_id = repository.add_classification_label(0, "unlabeled")
    repository.add_classification_item(item_id=item_id, image_asset_id=asset.asset_id, label_id=label_id)
    connection.commit()
    connection.execute("DROP TABLE item_geometry")
    connection.execute("UPDATE ob_schema SET schema_version='1.4.0'")
    connection.commit()

    assert read_dataset_info(connection)["schema_version"] == "1.5.0"
    assert tuple(connection.execute(
        "SELECT bbox_w,bbox_h,crop_bbox_w,crop_bbox_h FROM item_geometry WHERE item_id=?",
        (item_id,),
    ).fetchone()) == (7, 5, 7, 5)
    assert validate_database(connection)["valid"]
