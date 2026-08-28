from __future__ import annotations

import time
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from oracle_builder.artifacts import read_run_config, read_run_manifest
from oracle_builder.clustering.evidence import ClusterEvidenceIndex
from oracle_builder.classification.evidence import IdentityEvidenceIndex
from oracle_builder.data.decoders import (
    prepare_classification_input,
)
from oracle_builder.data.sqlite_dataset import prepare_segmentation_input
from oracle_builder.data.tiling import extract_tile, plan_tiles, reassemble_tiles
from oracle_builder.evaluation.segmentation_targets import (
    CANDIDATE_DELTA,
    reconstruct_validated_mask,
    reconstruct_validated_probability,
    segmentation_target_mode,
)
from oracle_builder.inference.contracts import (
    ArrayPayload,
    InferenceItem,
    InferenceResult,
    InferenceResultSet,
    ModelReference,
    new_uuid,
    utc_now,
)
from oracle_builder.inference.executor import (
    ClassificationExecutor,
    EmbeddingExecutor,
    execution_device_diagnostics,
)
from oracle_builder.saving.load_test import load_model_for_run


def _labels(config: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(row["class_index"]): dict(row)
        for row in config.get("dataset", {}).get("labels", [])
    }


def _segmentation_outputs(model, batch: np.ndarray, activation: str) -> dict[str, Any]:
    if hasattr(model, "predict_outputs"):
        outputs = model.predict_outputs(batch, verbose=0)
        return {
            "logits": np.asarray(outputs["logits"]),
            "probabilities": np.asarray(outputs["probabilities"]),
            "logits_source": outputs.get("logits_source", "model"),
        }
    try:
        direct = model.predict(batch, verbose=0)
        if isinstance(direct, dict) and {"logits", "probabilities"}.issubset(direct):
            return {
                "logits": np.asarray(direct["logits"]),
                "probabilities": np.asarray(direct["probabilities"]),
                "logits_source": "model",
            }
        from tensorflow import keras

        view = keras.Model(
            model.input,
            {
                "logits": model.get_layer("logits").output,
                "probabilities": model.output,
            },
        )
        outputs = view.predict(batch, verbose=0)
        return {
            "logits": np.asarray(outputs["logits"]),
            "probabilities": np.asarray(outputs["probabilities"]),
            "logits_source": "model",
        }
    except (AttributeError, ValueError):
        probabilities = np.asarray(model.predict(batch, verbose=0))
        clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
        if activation == "sigmoid":
            logits = np.log(clipped) - np.log1p(-clipped)
            source = "derived_inverse_sigmoid"
        else:
            logits = np.log(np.clip(probabilities, 1e-7, 1.0))
            source = "derived_log_probability"
        return {
            "logits": logits.astype("float32"),
            "probabilities": probabilities,
            "logits_source": source,
        }


def _prepare_segmentation_tile(
    raw: np.ndarray, config: dict[str, Any]
) -> np.ndarray:
    prepared = prepare_segmentation_input(
        raw, config["data"]["input_shape"], config
    )
    if config.get("data", {}).get("candidate_sdf", False):
        return np.asarray(prepared, dtype="float32")
    value = np.asarray(prepared)
    if value.ndim == 3 and value.shape[-1] == 2:
        roi = value[..., 0].astype("float32")
        if np.asarray(raw).dtype.kind in {"u", "i"} and roi.max(initial=0) > 1:
            roi /= float(np.iinfo(np.asarray(raw).dtype).max)
        candidate = (value[..., 1] > 0).astype("float32")
        return np.stack([roi, candidate], axis=-1)
    if value.dtype.kind in {"u", "i"} and value.max(initial=0) > 1:
        value = value.astype("float32") / float(np.iinfo(value.dtype).max)
    return value.astype("float32")


class InferenceBundle:
    """A portable preprocessor + neural core + postprocessor inference unit."""

    def __init__(
        self,
        model: Any,
        config: dict[str, Any],
        model_reference: ModelReference,
        *,
        evidence_index: IdentityEvidenceIndex | None = None,
        cluster_index: ClusterEvidenceIndex | None = None,
    ):
        self.model = model
        self.config = config
        self.model_reference = model_reference
        self.evidence_index = evidence_index
        self.cluster_index = cluster_index
        self._classification_executor = (
            ClassificationExecutor(model, tuple(config["data"]["input_shape"]))
            if model_reference.task in {"classification", "clustering"}
            else None
        )
        self._embedding_executor = (
            EmbeddingExecutor(model, tuple(config["data"]["input_shape"]))
            if model_reference.task == "embedding"
            else None
        )
        self.execution_diagnostics = (
            self._classification_executor.execution_diagnostics
            if self._classification_executor is not None
            else self._embedding_executor.execution_diagnostics
            if self._embedding_executor is not None
            else execution_device_diagnostics()
        )
        self.runtime_diagnostics: dict[str, Any] = {
            "execution": self.execution_diagnostics,
        }

    @classmethod
    def load(cls, run_dir: str | Path) -> "InferenceBundle":
        run_dir = Path(run_dir).expanduser().resolve()
        config = read_run_config(run_dir)
        manifest = read_run_manifest(run_dir)
        model = load_model_for_run(run_dir, config, prefer_savedmodel=True)
        evidence_path = run_dir / "model" / "classification_evidence"
        if not evidence_path.exists():
            evidence_path = run_dir / "model" / "classification_evidence.npz"
        evidence = (
            IdentityEvidenceIndex.load(evidence_path)
            if evidence_path.exists()
            else None
        )
        cluster_path = run_dir / "model" / "clustering_evidence"
        cluster_index = (
            ClusterEvidenceIndex.load(cluster_path)
            if cluster_path.exists()
            else None
        )
        return cls(
            model,
            config,
            ModelReference(
                artifact_id=manifest["artifact_id"],
                artifact_fingerprint=manifest.get("fingerprint_sha256"),
                run_id=manifest["run_id"],
                task=config["run"]["task"],
                architecture=config["run"]["model"],
            ),
            evidence_index=evidence,
            cluster_index=cluster_index,
        )

    def predict(
        self,
        item: InferenceItem,
        *,
        result_set_id: str | None = None,
        sequence_number: int | None = None,
    ) -> InferenceResult:
        resolved_result_set_id = result_set_id or new_uuid()
        started = time.perf_counter()
        received_at = utc_now()
        try:
            output = (
                self._predict_classification(item)
                if self.model_reference.task in {"classification", "clustering"}
                else self._predict_embedding(item)
                if self.model_reference.task == "embedding"
                else self._predict_segmentation(item)
            )
            status = "ok"
            error = None
        except (TypeError, ValueError, KeyError) as exc:
            output = None
            status = "rejected"
            error = {"type": type(exc).__name__, "message": str(exc)}
        except Exception as exc:  # inference streams must return correlated failures
            output = None
            status = "failed"
            error = {"type": type(exc).__name__, "message": str(exc)}
        return InferenceResult(
            request_id=item.request_id,
            item_id=item.item_id,
            source=item.source,
            model=self.model_reference,
            output=output,
            input_sha256=item.input_sha256,
            result_set_id=resolved_result_set_id,
            sequence_number=sequence_number,
            status=status,
            received_at=received_at,
            completed_at=utc_now(),
            duration_ms=(time.perf_counter() - started) * 1000.0,
            error=error,
        )

    def predict_batch(
        self,
        items: Iterable[InferenceItem],
        *,
        source_dataset: dict[str, Any] | None = None,
    ) -> InferenceResultSet:
        """Predict a caller-supplied batch with one classifier forward pass.

        Inputs are independently prepared and outputs remain correlated to the
        original request/item IDs.  Segmentation remains item-oriented because
        each ROI can expand into a variable number of tiles; its batching is a
        separate tile scheduler concern.
        """
        result_set = InferenceResultSet(
            model=self.model_reference,
            source_dataset=source_dataset,
            execution=self.execution_diagnostics,
        )
        materialized_items = list(items)
        if self.model_reference.task == "embedding":
            self._predict_embedding_batch(materialized_items, result_set)
            return result_set.complete()
        if self.model_reference.task not in {"classification", "clustering"}:
            for index, item in enumerate(materialized_items):
                result_set.append(
                    self.predict(
                        item,
                        result_set_id=result_set.result_set_id,
                        sequence_number=index,
                    )
                )
            return result_set.complete()

        results: list[InferenceResult | None] = [None] * len(materialized_items)
        prepared: list[np.ndarray] = []
        prepared_indices: list[int] = []
        started: dict[int, tuple[float, str]] = {}
        for index, item in enumerate(materialized_items):
            started[index] = (time.perf_counter(), utc_now())
            try:
                prepared.append(self._prepare_classification(item))
                prepared_indices.append(index)
            except Exception as exc:
                results[index] = self._error_result(
                    item,
                    exc,
                    result_set_id=result_set.result_set_id,
                    sequence_number=index,
                    started=started[index],
                )
        if prepared:
            try:
                values = self._classification_executor.predict(np.stack(prepared, axis=0))
                cluster_packets = None
                if self.cluster_index is not None and values["features"] is not None:
                    embeddings = np.asarray(values["features"], dtype="float32")
                    query_uuids = [
                        materialized_items[item_index].source.resource_id
                        if materialized_items[item_index].source is not None
                        else None
                        for item_index in prepared_indices
                    ]
                    cluster_packets = self.cluster_index.packet_many(
                        embeddings,
                        query_uuids=query_uuids,
                        top_k=int(self.config.get("clustering", {}).get("top_k", 5)),
                    )
                for batch_index, item_index in enumerate(prepared_indices):
                    item = materialized_items[item_index]
                    results[item_index] = self._success_result(
                        item,
                        self._classification_output_from_values(
                            item,
                            values,
                            batch_index,
                            cluster_packet=(
                                cluster_packets[batch_index]
                                if cluster_packets is not None
                                else None
                            ),
                        ),
                        result_set_id=result_set.result_set_id,
                        sequence_number=item_index,
                        started=started[item_index],
                    )
            except Exception as exc:
                for item_index in prepared_indices:
                    results[item_index] = self._error_result(
                        materialized_items[item_index],
                        exc,
                        result_set_id=result_set.result_set_id,
                        sequence_number=item_index,
                        started=started[item_index],
                    )
        for result in results:
            assert result is not None
            result_set.append(result)
        return result_set.complete()

    def predict_stream(
        self, items: Iterable[InferenceItem]
    ) -> Iterator[InferenceResult]:
        result_set = InferenceResultSet(model=self.model_reference)
        for index, item in enumerate(items):
            yield self.predict(
                item,
                result_set_id=result_set.result_set_id,
                sequence_number=index,
            )

    async def predict_stream_async(
        self, items: AsyncIterable[InferenceItem]
    ) -> AsyncIterator[InferenceResult]:
        result_set = InferenceResultSet(model=self.model_reference)
        index = 0
        async for item in items:
            yield self.predict(
                item,
                result_set_id=result_set.result_set_id,
                sequence_number=index,
            )
            index += 1

    def _predict_classification(self, item: InferenceItem) -> dict[str, Any]:
        prepared = self._prepare_classification(item)
        values = self._classification_executor.predict(prepared[None, ...])
        return self._classification_output_from_values(item, values, 0)

    def _predict_embedding(self, item: InferenceItem) -> dict[str, Any]:
        prepared = self._prepare_classification(item)
        values = self._embedding_executor.predict(prepared[None, ...])[0]
        return {
            "type": "embedding",
            "embedding": ArrayPayload(np.asarray(values, dtype="float32")),
            "embedding_normalized": bool(
                self.config.get("model", {}).get("normalize_embeddings", True)
            ),
        }

    def _predict_embedding_batch(
        self, items: list[InferenceItem], result_set: InferenceResultSet
    ) -> None:
        """Prepare and execute representation requests in one model call."""
        results: list[InferenceResult | None] = [None] * len(items)
        prepared: list[np.ndarray] = []
        indices: list[int] = []
        started: dict[int, tuple[float, str]] = {}
        for index, item in enumerate(items):
            started[index] = (time.perf_counter(), utc_now())
            try:
                prepared.append(self._prepare_classification(item))
                indices.append(index)
            except Exception as exc:
                results[index] = self._error_result(
                    item,
                    exc,
                    result_set_id=result_set.result_set_id,
                    sequence_number=index,
                    started=started[index],
                )
        if prepared:
            try:
                values = self._embedding_executor.predict(np.stack(prepared, axis=0))
                for batch_index, item_index in enumerate(indices):
                    item = items[item_index]
                    output = {
                        "type": "embedding",
                        "embedding": ArrayPayload(
                            np.asarray(values[batch_index], dtype="float32")
                        ),
                        "embedding_normalized": bool(
                            self.config.get("model", {}).get(
                                "normalize_embeddings", True
                            )
                        ),
                    }
                    results[item_index] = self._success_result(
                        item,
                        output,
                        result_set_id=result_set.result_set_id,
                        sequence_number=item_index,
                        started=started[item_index],
                    )
            except Exception as exc:
                for item_index in indices:
                    results[item_index] = self._error_result(
                        items[item_index],
                        exc,
                        result_set_id=result_set.result_set_id,
                        sequence_number=item_index,
                        started=started[item_index],
                    )
        for result in results:
            if result is not None:
                result_set.append(result)

    def warm_for_serving(self, batch_sizes: tuple[int, ...] = (1,)) -> dict[str, Any]:
        """Compile and exercise the resident callable without retaining input data."""
        if self.model_reference.task == "embedding" and self._embedding_executor is not None:
            executor = self._embedding_executor
        elif self.model_reference.task in {"classification", "clustering"} and self._classification_executor is not None:
            executor = self._classification_executor
        else:
            self.runtime_diagnostics = {
                "runtime": "segmentation",
                "execution": self.execution_diagnostics,
                "warmup": [],
            }
            return self.runtime_diagnostics
        warmup: list[dict[str, Any]] = []
        for requested_size in dict.fromkeys(max(1, int(value)) for value in batch_sizes):
            candidate = requested_size
            while True:
                try:
                    warmup.append(executor.warm(candidate))
                    break
                except Exception as exc:
                    try:
                        import tensorflow as tf
                        is_oom = isinstance(exc, tf.errors.ResourceExhaustedError)
                    except ImportError:  # pragma: no cover
                        is_oom = False
                    if not is_oom or candidate <= 1:
                        raise
                    candidate = max(1, candidate // 2)
        self.runtime_diagnostics = {
            "runtime": executor.runtime,
            "execution": self.execution_diagnostics,
            "warmup": warmup,
            "resolved_max_batch_size": warmup[-1]["batch_size"] if warmup else 1,
        }
        return self.runtime_diagnostics

    def _prepare_classification(self, item: InferenceItem) -> np.ndarray:
        raw = item.inputs["image"].values
        promotion = self.config.get("promotion", {})
        embedded_preprocessing = bool(
            promotion.get("preprocessing", {}).get("embedded", False)
        )
        if embedded_preprocessing:
            # The promoted Keras model owns resize/channel/intensity handling.
            # Do not silently apply the dataset pipeline a second time.
            prepared = np.asarray(raw)
            if prepared.ndim == 2:
                prepared = prepared[..., None]
        else:
            prepared = prepare_classification_input(
                raw, self.config["data"]["input_shape"], self.config
            )
        return np.asarray(prepared)

    def _classification_output_from_values(
        self,
        item: InferenceItem,
        values: dict[str, Any],
        batch_index: int,
        cluster_packet: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        logits = np.asarray(values["logits"][batch_index], dtype="float32")
        probabilities = np.asarray(
            values["probabilities"][batch_index], dtype="float32"
        )
        features = values["features"]
        embedding = (
            np.asarray(features[batch_index], dtype="float32")
            if features is not None
            else None
        )
        if self.model_reference.task == "clustering":
            if embedding is None:
                raise ValueError("Clustering model does not expose an embedding")
            if self.cluster_index is None:
                raise ValueError("Clustering model is missing its clustering evidence index")
            return {
                "type": "clustering",
                "embedding": ArrayPayload(embedding),
                "embedding_normalized": bool(
                    self.config.get("model", {}).get("normalize_embeddings", True)
                ),
                "evidence": cluster_packet or self.cluster_index.packet(
                    embedding,
                    query_uuid=(item.source.resource_id if item.source is not None else None),
                    top_k=int(self.config.get("clustering", {}).get("top_k", 5)),
                ),
            }
        class_index = int(np.argmax(probabilities))
        label = _labels(self.config).get(class_index, {})
        output: dict[str, Any] = {
            "type": "classification",
            "decision": {
                "class_index": class_index,
                "label_id": label.get("label_id"),
                "label_name": label.get("name"),
                "concept_id": label.get("concept_id"),
                "concept_node_id": label.get("concept_node_id"),
                "concept_relationship": label.get("concept_relationship"),
                "abstained": False,
            },
            "logits": [float(value) for value in logits],
            "logits_source": values["logits_source"],
            "probabilities": [
                {
                    "class_index": index,
                    "label_id": _labels(self.config).get(index, {}).get("label_id"),
                    "label_name": _labels(self.config).get(index, {}).get("name"),
                    "concept_id": _labels(self.config).get(index, {}).get("concept_id"),
                    "concept_node_id": _labels(self.config).get(index, {}).get("concept_node_id"),
                    "concept_relationship": _labels(self.config).get(index, {}).get("concept_relationship"),
                    "probability": float(probability),
                }
                for index, probability in enumerate(probabilities)
            ],
        }
        if embedding is not None:
            output["embedding"] = ArrayPayload(embedding)
            output["embedding_normalized"] = bool(
                self.config.get("model", {}).get("normalize_embeddings", True)
            )
            if self.evidence_index is not None:
                output["evidence"] = self.evidence_index.packet(
                    embedding,
                    probabilities,
                    query_uuid=(
                        item.source.resource_id if item.source is not None else None
                    ),
                    k=int(self.config.get("evidence", {}).get("knn_k", 5)),
                )
            if self.cluster_index is not None:
                output["clustering_evidence"] = cluster_packet or self.cluster_index.packet(
                    embedding,
                    query_uuid=(item.source.resource_id if item.source is not None else None),
                    top_k=int(self.config.get("clustering", {}).get("top_k", 5)),
                )
        return output

    def _success_result(
        self,
        item: InferenceItem,
        output: dict[str, Any],
        *,
        result_set_id: str,
        sequence_number: int,
        started: tuple[float, str],
    ) -> InferenceResult:
        return InferenceResult(
            request_id=item.request_id,
            item_id=item.item_id,
            source=item.source,
            model=self.model_reference,
            output=output,
            input_sha256=item.input_sha256,
            result_set_id=result_set_id,
            sequence_number=sequence_number,
            status="ok",
            received_at=started[1],
            completed_at=utc_now(),
            duration_ms=(time.perf_counter() - started[0]) * 1000.0,
        )

    def _error_result(
        self,
        item: InferenceItem,
        exc: Exception,
        *,
        result_set_id: str,
        sequence_number: int,
        started: tuple[float, str],
    ) -> InferenceResult:
        status = "rejected" if isinstance(exc, (TypeError, ValueError, KeyError)) else "failed"
        return InferenceResult(
            request_id=item.request_id,
            item_id=item.item_id,
            source=item.source,
            model=self.model_reference,
            output=None,
            input_sha256=item.input_sha256,
            result_set_id=result_set_id,
            sequence_number=sequence_number,
            status=status,
            received_at=started[1],
            completed_at=utc_now(),
            duration_ms=(time.perf_counter() - started[0]) * 1000.0,
            error={"type": type(exc).__name__, "message": str(exc)},
        )

    def _predict_segmentation(self, item: InferenceItem) -> dict[str, Any]:
        raw = np.asarray(item.inputs["image"].values)
        candidate_payload = item.inputs.get("candidate_mask")
        if candidate_payload is not None:
            candidate = np.asarray(candidate_payload.values)
            if candidate.ndim == 3 and candidate.shape[-1] == 1:
                candidate = candidate[..., 0]
            if raw.shape[:2] != candidate.shape[:2]:
                raise ValueError("image and candidate_mask spatial shapes must match")
            raw = np.stack([raw.squeeze(), candidate], axis=-1)
        tile_h, tile_w = (
            int(value) for value in self.config["data"]["input_shape"][:2]
        )
        tiling = self.config.get("tiling", {})
        should_tile = bool(tiling.get("enabled", False)) and (
            raw.shape[0] > tile_h
            or raw.shape[1] > tile_w
            or not bool(tiling.get("tile_large_rois_only", True))
        )
        if should_tile:
            plans = plan_tiles(
                raw.shape[:2],
                (tile_h, tile_w),
                float(tiling.get("overlap_fraction", 0.5)),
            )
            prepared = np.stack(
                [
                    _prepare_segmentation_tile(
                        extract_tile(raw, plan), self.config
                    )
                    for plan in plans
                ]
            )
        else:
            plans = None
            prepared = _prepare_segmentation_tile(raw, self.config)[None, ...]
        values = _segmentation_outputs(
            self.model,
            prepared,
            self.config.get("model", {}).get("final_activation", "sigmoid"),
        )
        logits = list(np.asarray(values["logits"]))
        probabilities = list(np.asarray(values["probabilities"]))
        if plans is not None:
            blend = tiling.get("blend_mode", "hann")
            logits_value = reassemble_tiles(logits, plans, blend_mode=blend)
            probability = reassemble_tiles(probabilities, plans, blend_mode=blend)
        else:
            logits_value = logits[0]
            probability = probabilities[0]
        threshold = float(
            self.config.get("evaluation", {}).get(
                "segmentation_threshold", 0.5
            )
        )
        binary = probability >= threshold
        target_mode = segmentation_target_mode(self.config)
        output: dict[str, Any] = {
            "type": "mask_refinement",
            "target_mode": target_mode,
            "threshold": {
                "value": threshold,
                "source": "artifact_validation_optimization",
            },
            "logits": ArrayPayload(np.asarray(logits_value, dtype="float32")),
            "logits_source": values["logits_source"],
            "probability_map": ArrayPayload(
                np.asarray(probability, dtype="float32")
            ),
            "mask": ArrayPayload(binary.astype("uint8")),
            "transform": {
                "original_shape": list(item.inputs["image"].values.shape[:2]),
                "tile_shape": [tile_h, tile_w],
                "tile_count": len(plans) if plans is not None else 1,
                "tile_overlap": (
                    float(tiling.get("overlap_fraction", 0.5))
                    if plans is not None
                    else 0.0
                ),
                "reassembly": (
                    tiling.get("blend_mode", "hann")
                    if plans is not None
                    else "none"
                ),
            },
        }
        if target_mode == CANDIDATE_DELTA:
            if candidate_payload is None:
                if raw.ndim != 3 or raw.shape[-1] < 2:
                    raise ValueError(
                        "candidate_delta inference requires candidate_mask"
                    )
                candidate = raw[..., 1]
            output["reconstructed_probability_map"] = ArrayPayload(
                reconstruct_validated_probability(candidate, probability)
            )
            output["reconstructed_mask"] = ArrayPayload(
                reconstruct_validated_mask(candidate, binary).astype("uint8")
            )
        return output
