"""Versioned dataset contracts and storage adapters."""

from oracle_data_contracts.datasets.lifecycle import (
    restore_workspace_snapshot,
    release_training_dataset,
    save_checkpoint,
    save_workspace_snapshot,
    thaw_database,
)
from oracle_data_contracts.datasets.metadata import (
    add_metadata_document,
    discover_metadata_documents,
    parse_metadata_document,
)
from oracle_data_contracts.datasets.repository import SQLiteDatasetRepository
from oracle_data_contracts.datasets.schema import (
    DATASET_TYPES,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    DatasetSchemaError,
    dataset_fingerprint,
    initialize_database,
    read_dataset_info,
    set_dataset_lifecycle,
    validate_database,
    workspace_fingerprint,
)
from oracle_builder.datasets.legacy_roi import (
    ensure_mask_refinement_database,
    inspect_dataset_kind,
    is_legacy_roi_database,
    migrate_legacy_roi_if_needed,
    migrate_legacy_roi_database,
)
from oracle_data_contracts.datasets.transfer import export_dataset, import_dataset_export
from oracle_data_contracts.datasets.subset import subset_classification_dataset
from oracle_data_contracts.datasets.workspace import (
    add_annotation_label,
    add_annotation_review,
    add_item_label_annotation,
    add_model_evidence,
    complete_inference_run,
    create_inference_run,
    store_evidence_array,
)

__all__ = [
    "DATASET_TYPES",
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "DatasetSchemaError",
    "SQLiteDatasetRepository",
    "add_metadata_document",
    "add_annotation_label",
    "add_annotation_review",
    "add_item_label_annotation",
    "add_model_evidence",
    "create_inference_run",
    "complete_inference_run",
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
    "save_workspace_snapshot",
    "subset_classification_dataset",
    "set_dataset_lifecycle",
    "store_evidence_array",
    "thaw_database",
    "restore_workspace_snapshot",
    "release_training_dataset",
    "validate_database",
    "workspace_fingerprint",
]
