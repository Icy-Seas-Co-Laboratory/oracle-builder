from __future__ import annotations

import time
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from oracle_builder.artifacts import read_run_config, read_run_manifest
from oracle_builder.classification.evidence import IdentityEvidenceIndex
from oracle_builder.classification.features import predict_classification_outputs
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
    ):
        self.model = model
        self.config = config
        self.model_reference = model_reference
        self.evidence_index = evidence_index

    @classmethod
    def load(cls, run_dir: str | Path) -> "InferenceBundle":
        run_dir = Path(run_dir).expanduser().resolve()
        config = read_run_config(run_dir)
        manifest = read_run_manifest(run_dir)
        model = load_model_for_run(run_dir, config)
        evidence_path = run_dir / "model" / "classification_evidence"
        if not evidence_path.exists():
            evidence_path = run_dir / "model" / "classification_evidence.npz"
        evidence = (
            IdentityEvidenceIndex.load(evidence_path)
            if evidence_path.exists()
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
                if self.model_reference.task == "classification"
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
        result_set = InferenceResultSet(
            model=self.model_reference,
            source_dataset=source_dataset,
        )
        for index, item in enumerate(items):
            result_set.append(
                self.predict(
                    item,
                    result_set_id=result_set.result_set_id,
                    sequence_number=index,
                )
            )
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
        values = predict_classification_outputs(self.model, prepared[None, ...])
        logits = np.asarray(values["logits"][0], dtype="float32")
        probabilities = np.asarray(values["probabilities"][0], dtype="float32")
        features = values["features"]
        embedding = (
            np.asarray(features[0], dtype="float32") if features is not None else None
        )
        class_index = int(np.argmax(probabilities))
        label = _labels(self.config).get(class_index, {})
        output: dict[str, Any] = {
            "type": "classification",
            "decision": {
                "class_index": class_index,
                "label_id": label.get("label_id"),
                "label_name": label.get("name"),
                "abstained": False,
            },
            "logits": [float(value) for value in logits],
            "logits_source": values["logits_source"],
            "probabilities": [
                {
                    "class_index": index,
                    "label_id": _labels(self.config).get(index, {}).get("label_id"),
                    "label_name": _labels(self.config).get(index, {}).get("name"),
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
        return output

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
