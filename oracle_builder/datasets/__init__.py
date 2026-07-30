"""Versioned dataset contracts and storage adapters."""

from oracle_builder.datasets.lifecycle import save_checkpoint, thaw_database
from oracle_builder.datasets.metadata import (
    add_metadata_document,
    discover_metadata_documents,
    parse_metadata_document,
)
from oracle_builder.datasets.repository import SQLiteDatasetRepository
from oracle_builder.datasets.schema import (
    DATASET_TYPES,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    DatasetSchemaError,
    dataset_fingerprint,
    initialize_database,
    read_dataset_info,
    set_dataset_lifecycle,
    validate_database,
)
from oracle_builder.datasets.legacy_roi import (
    ensure_mask_refinement_database,
    inspect_dataset_kind,
    is_legacy_roi_database,
    migrate_legacy_roi_if_needed,
    migrate_legacy_roi_database,
)
from oracle_builder.datasets.transfer import export_dataset, import_dataset_export

__all__ = [
    "DATASET_TYPES",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "DatasetSchemaError",
    "SQLiteDatasetRepository",
    "add_metadata_document",
    "dataset_fingerprint",
    "discover_metadata_documents",
    "export_dataset",
    "import_dataset_export",
    "initialize_database",
    "ensure_mask_refinement_database",
    "inspect_dataset_kind",
    "is_legacy_roi_database",
    "migrate_legacy_roi_if_needed",
    "migrate_legacy_roi_database",
    "parse_metadata_document",
    "read_dataset_info",
    "save_checkpoint",
    "set_dataset_lifecycle",
    "thaw_database",
    "validate_database",
]
