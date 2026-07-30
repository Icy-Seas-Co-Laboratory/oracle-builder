"""Portable, storage-neutral Oracle Builder inference contracts."""

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

__all__ = [
    "ArrayPayload",
    "InferenceBundle",
    "InferenceBatchPlan",
    "InferenceItem",
    "InMemorySink",
    "InferenceResult",
    "InferenceResultSet",
    "ModelReference",
    "JSONLinesSink",
    "SourceReference",
    "run_connector",
    "resolve_inference_batch_size",
]
