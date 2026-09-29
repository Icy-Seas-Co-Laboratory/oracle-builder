"""Portable, storage-neutral Oracle Builder inference contracts.

The package root is intentionally lightweight.  Prediction serialization uses
``inference.contracts`` and is itself used by the batch workflow; importing
the workflow here would turn that ordinary contracts import into a cycle.
"""

from typing import TYPE_CHECKING

from oracle_builder.inference.bundle import InferenceBundle
from oracle_builder.inference.batching import (
    InferenceBatchPlan,
    resolve_inference_batch_size,
)
from oracle_builder.inference.connectors import (
    InMemorySink,
    JSONLinesSink,
    run_connector,
)
from oracle_builder.inference.contracts import (
    ArrayPayload,
    InferenceItem,
    InferenceResult,
    InferenceResultSet,
    ModelReference,
    SourceReference,
)
if TYPE_CHECKING:
    from oracle_builder.inference.workflow import (
        INFERENCE_RESULT_SCHEMA_NAME,
        INFERENCE_RESULT_SCHEMA_VERSION,
        INFERENCE_SHARD_SCHEMA_VERSION,
        InferenceRequest,
        merge_inference_shards,
        run_inference,
    )


def __getattr__(name: str):
    """Lazily expose the optional batch-workflow convenience API.

    Keeping this import local preserves the historical root-level API while
    allowing low-level prediction code to import ``inference.contracts``.
    """
    if name in {
        "INFERENCE_RESULT_SCHEMA_NAME", "INFERENCE_RESULT_SCHEMA_VERSION",
        "INFERENCE_SHARD_SCHEMA_VERSION",
        "InferenceRequest", "merge_inference_shards", "run_inference",
    }:
        from oracle_builder.inference import workflow
        return getattr(workflow, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ArrayPayload",
    "InferenceBundle",
    "InferenceBatchPlan",
    "InferenceItem",
    "InMemorySink",
    "InferenceResult",
    "InferenceResultSet",
    "InferenceRequest",
    "INFERENCE_RESULT_SCHEMA_NAME",
    "INFERENCE_RESULT_SCHEMA_VERSION",
    "INFERENCE_SHARD_SCHEMA_VERSION",
    "ModelReference",
    "JSONLinesSink",
    "SourceReference",
    "run_connector",
    "resolve_inference_batch_size",
    "run_inference",
    "merge_inference_shards",
]
