from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Protocol

from oracle_builder.inference.bundle import InferenceBundle
from oracle_builder.inference.contracts import (
    InferenceItem,
    InferenceResult,
    InferenceResultSet,
)


class InferenceSource(Protocol):
    """Edge connector contract; implementations retain ownership of source data."""

    def __iter__(self) -> Iterator[InferenceItem]: ...


class InferenceSink(Protocol):
    """Explicit destination for results. Prediction itself never selects a sink."""

    def start(self, result_set: InferenceResultSet) -> None: ...

    def write(self, result: InferenceResult) -> None: ...

    def complete(self, result_set: InferenceResultSet) -> None: ...


class InMemorySink:
    def __init__(self) -> None:
        self.results: list[InferenceResult] = []

    def start(self, result_set: InferenceResultSet) -> None:
        self.results.clear()

    def write(self, result: InferenceResult) -> None:
        self.results.append(result)

    def complete(self, result_set: InferenceResultSet) -> None:
        return None


class JSONLinesSink:
    """An explicit, append-free portable stream serialization sink."""

    def __init__(
        self, path: str | Path, *, include_array_data: bool = True
    ) -> None:
        self.path = Path(path)
        self.include_array_data = include_array_data
        self._handle = None

    def start(self, result_set: InferenceResultSet) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("x", encoding="utf-8")
        self._write(
            {
                "event": "result_set_start",
                "schema_version": "1.0.0",
                "result_set_id": result_set.result_set_id,
                "model": result_set.model.to_dict(),
                "started_at": result_set.started_at,
            }
        )

    def write(self, result: InferenceResult) -> None:
        self._write(
            {
                "event": "result",
                "value": result.to_dict(
                    include_array_data=self.include_array_data
                ),
            }
        )

    def complete(self, result_set: InferenceResultSet) -> None:
        self._write(
            {
                "event": "result_set_complete",
                "result_set_id": result_set.result_set_id,
                "completed_at": result_set.completed_at,
                "counts": result_set.counts,
            }
        )
        assert self._handle is not None
        self._handle.close()
        self._handle = None

    def _write(self, value: dict) -> None:
        if self._handle is None:
            raise RuntimeError("JSONLinesSink has not been started")
        self._handle.write(json.dumps(value, sort_keys=True) + "\n")


def run_connector(
    bundle: InferenceBundle,
    source: Iterable[InferenceItem],
    *,
    sink: InferenceSink | None = None,
    source_dataset: dict | None = None,
) -> InferenceResultSet:
    """Run a connector without persistence unless the caller supplies a sink."""
    result_set = InferenceResultSet(
        model=bundle.model_reference,
        source_dataset=source_dataset,
    )
    destination = sink or InMemorySink()
    destination.start(result_set)
    for sequence, item in enumerate(source):
        result = bundle.predict(
            item,
            result_set_id=result_set.result_set_id,
            sequence_number=sequence,
        )
        result_set.append(result)
        destination.write(result)
    result_set.complete()
    destination.complete(result_set)
    return result_set
