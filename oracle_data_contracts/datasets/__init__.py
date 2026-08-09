"""SQLite Dataset V1 contract, lifecycle, and annotation-workspace APIs."""

from oracle_data_contracts.datasets.lifecycle import (
    release_training_dataset,
    restore_workspace_snapshot,
    save_checkpoint,
    save_workspace_snapshot,
    thaw_database,
)
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
from oracle_data_contracts.datasets.taxonomy import (
    import_taxonomy_concepts,
    map_classification_label_to_concept,
    taxonomy_concept_id,
)

__all__ = [
    "DATASET_TYPES", "SCHEMA_NAME", "SCHEMA_VERSION", "DatasetSchemaError",
    "add_annotation_label", "add_annotation_review", "add_item_label_annotation",
    "add_model_evidence", "complete_inference_run", "create_inference_run",
    "dataset_fingerprint", "initialize_database", "read_dataset_info",
    "release_training_dataset", "restore_workspace_snapshot", "save_checkpoint",
    "save_workspace_snapshot", "set_dataset_lifecycle", "store_evidence_array",
    "subset_classification_dataset", "thaw_database", "validate_database",
    "workspace_fingerprint", "import_taxonomy_concepts",
    "map_classification_label_to_concept", "taxonomy_concept_id",
]
