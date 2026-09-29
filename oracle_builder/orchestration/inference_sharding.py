"""Pure planning helpers for ordered, durable inference shard chains.

The service owns transactions and leases; this module only creates portable V1
infer descriptors and validates the state transitions it must persist.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any, Iterable, Mapping
from pathlib import Path
import sqlite3
import uuid

from oracle_data_contracts.artifacts import ArtifactRef
from oracle_data_contracts.work_units import WorkUnit


INFERENCE_SHARD_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS inference_shards (
 inference_run_id TEXT NOT NULL, shard_id TEXT NOT NULL, ordinal INTEGER NOT NULL,
 item_ids_json TEXT NOT NULL, item_ids_sha256 TEXT NOT NULL, status TEXT NOT NULL,
 output_ref_json TEXT, job_id TEXT, PRIMARY KEY(inference_run_id, shard_id),
 UNIQUE(inference_run_id, ordinal)
)
"""


def stable_item_ids(item_ids: Iterable[str]) -> tuple[str, ...]:
    ids = tuple(sorted(str(value) for value in item_ids))
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("inference shard population must be non-empty and unique")
    return ids


def item_ids_digest(item_ids: Iterable[str]) -> str:
    return hashlib.sha256(json.dumps(list(stable_item_ids(item_ids)), separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class InferenceShardPlan:
    item_ids: tuple[str, ...]
    shard_size: int = 1024
    unsharded_threshold: int = 1024

    def __post_init__(self) -> None:
        object.__setattr__(self, "item_ids", stable_item_ids(self.item_ids))
        if self.shard_size < 1 or self.unsharded_threshold < 1: raise ValueError("shard sizes must be positive")

    @property
    def sharded(self) -> bool: return len(self.item_ids) > self.unsharded_threshold

    def shards(self) -> tuple[tuple[str, ...], ...]:
        if not self.sharded: return (self.item_ids,)
        return tuple(self.item_ids[index:index + self.shard_size] for index in range(0, len(self.item_ids), self.shard_size))


def infer_shard_unit(base: WorkUnit, *, shard_id: str, item_ids: Iterable[str]) -> WorkUnit:
    ids = stable_item_ids(item_ids); parameters = dict(base.parameters)
    parameters.pop('shard_size', None)
    parameters.update({"shard_id": shard_id, "shard_item_ids": list(ids)})
    return WorkUnit(work_unit_id=base.work_unit_id, attempt_id=base.attempt_id, specification_id=base.specification_id,
                    action="infer", inputs=base.inputs, configuration=base.configuration, staging=base.staging,
                    resources=base.resources, parameters=parameters)


def merge_unit(base: WorkUnit, *, expected_item_ids: Iterable[str], shard_outputs: Mapping[str, ArtifactRef]) -> WorkUnit:
    ids = stable_item_ids(expected_item_ids)
    if not shard_outputs: raise ValueError("merge requires at least one committed shard output")
    inputs = dict(base.inputs)
    inputs.update({f"shard_{shard_id}": ref for shard_id, ref in sorted(shard_outputs.items())})
    parameters = dict(base.parameters); parameters.update({"merge_item_ids": list(ids), "shard_ids": sorted(shard_outputs)})
    parameters.pop('shard_size', None)
    return WorkUnit(work_unit_id=base.work_unit_id, attempt_id=base.attempt_id, specification_id=base.specification_id,
                    action="infer", inputs=inputs, configuration=base.configuration, staging=base.staging,
                    resources=base.resources, parameters=parameters)


def next_chain_step(plan: InferenceShardPlan, committed: Mapping[str, ArtifactRef]) -> tuple[str, tuple[str, ...]] | None:
    """Return next shard or ``merge`` only after all shard outputs commit."""
    shards = plan.shards()
    for ordinal, ids in enumerate(shards):
        shard_id = f"shard-{ordinal:05d}"
        if shard_id not in committed: return shard_id, ids
    return ("merge", plan.item_ids)


class InferenceShardingMixin:
    """Transactional helpers for the Orchestrator service to compose.

    The host supplies its normal job insertion method; this keeps scheduler
    policy independent of HTTP and worker transport.
    """
    inference_shard_threshold = 1024

    def _prepare_inference_shards(self, db: sqlite3.Connection, row: Mapping[str, Any], now: str):
        work = WorkUnit.from_dict(json.loads(row["work_unit_json"]))
        size = int(work.parameters.get("shard_size", self.inference_shard_threshold))
        input_ref = work.inputs["input"]
        dataset_path = self.artifact_store.resolve(input_ref) / 'payload'
        with sqlite3.connect(str(dataset_path)) as source:
            split = str(work.parameters.get("split", "all"))
            # Split membership is immutable model protocol; unsharded external
            # inference has no model split and therefore uses all item ids.
            ids = [str(value[0]) for value in source.execute("SELECT item_id FROM dataset_items ORDER BY item_id")]
        if split != 'all':
            from oracle_data_contracts.artifacts.splits import read_split_manifest
            manifest = read_split_manifest(self.artifact_store.resolve(work.inputs['model']))
            recorded = manifest.get('dataset', {})
            dataset = db.execute('SELECT dataset_id,revision_id FROM datasets WHERE dataset_id=?', (input_ref.artifact_id,)).fetchone()
            if not dataset or recorded.get('dataset_id') != dataset['dataset_id'] or recorded.get('revision_id') != dataset['revision_id']:
                raise ValueError('Named inference splits require the exact model dataset revision')
            selected = {entry['item_id'] for entry in manifest['assignments'] if entry['split'] == split}
            ids = [item for item in ids if item in selected]
        plan = InferenceShardPlan(tuple(ids), shard_size=size, unsharded_threshold=size)
        if not plan.sharded: return None
        for ordinal, item_ids in enumerate(plan.shards()):
            shard_id = f"shard-{ordinal:05d}"
            db.execute("INSERT INTO inference_shards(inference_run_id,shard_id,ordinal,item_ids_json,item_ids_sha256,status) VALUES(?,?,?,?,?,?)", (row["inference_run_id"], shard_id, ordinal, json.dumps(item_ids), item_ids_digest(item_ids), "pending"))
        return infer_shard_unit(work, shard_id="shard-00000", item_ids=plan.shards()[0])

    def _validate_inference_shard_output(self, path: str | Path, unit: WorkUnit) -> None:
        root = Path(path); artifact = json.loads((root / "artifact.json").read_text()); shard = json.loads((root / "inference_shard.json").read_text())
        expected = tuple(unit.parameters.get("shard_item_ids", ()))
        if shard.get("schema") != {"name": "oracle_builder_inference_shard", "version": 1}:
            raise ValueError("unsupported inference shard manifest schema")
        if shard.get("shard_id") != unit.parameters.get("shard_id") or tuple(shard.get("item_ids", ())) != expected or shard.get("item_ids_sha256") != item_ids_digest(expected):
            raise ValueError("inference shard output does not match its sealed work-unit population")
        if artifact.get("model", {}).get("artifact_id") != unit.inputs["model"].artifact_id:
            raise ValueError("inference shard model identity does not match its work unit")
        if artifact.get("input", {}).get("reference", {}).get("artifact_id") != unit.inputs["input"].artifact_id:
            raise ValueError("inference shard input identity does not match its work unit")
        prediction_set = artifact.get("parameters", {}).get("prediction_set")
        with sqlite3.connect(root / "predictions.sqlite") as db:
            actual = tuple(str(row[0]) for row in db.execute("SELECT uuid FROM predictions WHERE prediction_set=? ORDER BY uuid", (prediction_set,)))
        if actual != expected:
            raise ValueError("inference shard prediction rows do not match its sealed item-id population")

    def _validate_inference_merge_output(self, path: str | Path, unit: WorkUnit) -> None:
        root = Path(path); artifact = json.loads((root / "artifact.json").read_text())
        expected = tuple(sorted(str(value) for value in unit.parameters.get("merge_item_ids", ())))
        if not expected: raise ValueError("merge work unit lacks expected item ids")
        if artifact.get("model", {}).get("artifact_id") != unit.inputs["model"].artifact_id:
            raise ValueError("merged inference model identity does not match its work unit")
        prediction_set = artifact.get("parameters", {}).get("prediction_set")
        with sqlite3.connect(root / "predictions.sqlite") as db:
            actual = tuple(str(row[0]) for row in db.execute("SELECT uuid FROM predictions WHERE prediction_set=? ORDER BY uuid", (prediction_set,)))
        if actual != expected: raise ValueError("merged inference prediction rows do not match expected population")

    def _commit_inference_shard(self, db: sqlite3.Connection, job: Mapping[str, Any], lease_id: str, output_ref: ArtifactRef, now: str):
        """Persist one shard output and enqueue exactly one successor atomically."""
        run = db.execute("SELECT * FROM inference_runs WHERE job_id=?", (job["job_id"],)).fetchone()
        if run is None: return None
        unit = WorkUnit.from_dict(json.loads(job["work_unit_json"])); shard_id = unit.parameters.get("shard_id")
        if not shard_id: return None
        db.execute("UPDATE inference_shards SET status='completed',output_ref_json=?,job_id=? WHERE inference_run_id=? AND shard_id=?", (json.dumps(output_ref.to_dict(), sort_keys=True), job["job_id"], run["inference_run_id"], shard_id))
        rows = db.execute("SELECT shard_id,item_ids_json,output_ref_json FROM inference_shards WHERE inference_run_id=? ORDER BY ordinal", (run["inference_run_id"],)).fetchall()
        pending = next((row for row in rows if row[2] is None), None)
        base = WorkUnit.from_dict(json.loads(run["work_unit_json"])); next_job_id = str(uuid.uuid4())
        attempt_id = str(uuid.uuid4())
        base = replace(base, work_unit_id=next_job_id, attempt_id=attempt_id, staging=ArtifactRef('staging', next_job_id, revision=attempt_id))
        if pending is not None:
            successor = infer_shard_unit(base, shard_id=str(pending[0]), item_ids=json.loads(pending[1]))
        else:
            refs = {str(row[0]): ArtifactRef.from_dict(json.loads(row[2])) for row in rows}
            expected = [item for row in rows for item in json.loads(row[1])]
            successor = merge_unit(base, expected_item_ids=expected, shard_outputs=refs)
        encoded = successor.to_dict(); digest = successor.sha256
        db.execute("INSERT INTO jobs(job_id,specification_id,oracle_serve_url,action,parameters_json,resources_json,work_unit_json,work_unit_sha256,worker_pool_id,status,submitted_at,updated_at) VALUES(?,NULL,?,'infer',?,?,?,?,?,'queued',?,?)", (next_job_id, f"pull://{run['worker_pool_id']}", json.dumps({}), run["resources_json"], json.dumps(encoded, sort_keys=True), digest, run["worker_pool_id"], now, now))
        db.execute("UPDATE inference_runs SET job_id=?,status='queued',updated_at=? WHERE inference_run_id=?", (next_job_id, now, run["inference_run_id"]))
        return successor
