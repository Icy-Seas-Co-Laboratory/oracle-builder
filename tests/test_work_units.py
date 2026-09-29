from __future__ import annotations

import uuid

import pytest

from oracle_data_contracts.artifacts import ArtifactRef
from oracle_data_contracts.work_units import (
    ExecutionEnvelope,
    WorkUnit,
    WorkUnitError,
    WorkUnitV2,
    parse_work_unit,
)
from oracle_builder.orchestration.work_units import LocalPathWorkUnitAdapter, build_work_unit
from oracle_builder.orchestration.storage import LocalArtifactStore, artifact_ref_for_file


def test_work_unit_is_path_free_canonical_and_round_trips():
    work_unit = WorkUnit(
        work_unit_id=str(uuid.uuid4()),
        attempt_id=str(uuid.uuid4()),
        specification_id=str(uuid.uuid4()),
        action="train",
        inputs={"input": ArtifactRef("dataset", "dataset-1", "revision-1", "a" * 64)},
        configuration=ArtifactRef("configuration", "config-1", fingerprint_sha256="b" * 64),
        staging=ArtifactRef("staging", "stage-1", "attempt-1"),
        resources={"gpu_count": 1},
    )

    encoded = work_unit.to_dict()
    assert WorkUnit.from_dict(encoded) == work_unit
    assert WorkUnit.from_dict(encoded).canonical_json() == work_unit.canonical_json()
    assert len(work_unit.sha256) == 64
    assert "/" not in work_unit.canonical_json()


def test_work_unit_rejects_paths_and_wrong_staging_kind():
    base = {
        "schema": {"name": "oracle_work_unit", "version": 1},
        "work_unit_id": str(uuid.uuid4()),
        "attempt_id": str(uuid.uuid4()),
        "specification_id": str(uuid.uuid4()),
        "action": "run_validate",
        "inputs": {"run": {"kind": "model_run", "artifact_id": "run-1"}},
        "configuration": None,
        "staging": {"kind": "staging", "artifact_id": "job-1", "revision": "attempt-1"},
        "resources": {},
    }
    broken_path = {**base, "inputs": {"run": {"kind": "model_run", "artifact_id": "/host/path"}}}
    with pytest.raises(WorkUnitError, match="Invalid artifact reference"):
        WorkUnit.from_dict(broken_path)
    broken_staging = {**base, "staging": {"kind": "model_run", "artifact_id": "run-1"}}
    with pytest.raises(WorkUnitError, match="staging"):
        WorkUnit.from_dict(broken_staging)


def test_local_path_adapter_keeps_legacy_paths_out_of_durable_work_unit(tmp_path):
    dataset_id = str(uuid.uuid4())
    work_unit = build_work_unit(
        work_unit_id=str(uuid.uuid4()),
        attempt_id=str(uuid.uuid4()),
        specification_id=str(uuid.uuid4()),
        action="train",
        parameters={
            "config": str(tmp_path / "config.toml"),
            "input": str(tmp_path / "dataset.sqlite"),
            "runs_dir": str(tmp_path / "runs"),
            "output": "run-1",
        },
        resources={"gpu_count": 0},
        dataset={"dataset_id": dataset_id, "revision_id": "v1", "fingerprint_sha256": "c" * 64},
    )
    legacy = {"config": str(tmp_path / "config.toml"), "input": str(tmp_path / "dataset.sqlite"), "output": "run-1"}
    envelope = LocalPathWorkUnitAdapter.request_envelope(work_unit, legacy)

    assert str(tmp_path) not in work_unit.canonical_json()
    assert envelope["parameters"] == legacy
    assert envelope["work_unit"]["inputs"]["input"]["kind"] == "dataset"
    assert envelope["work_unit"]["staging"]["kind"] == "staging"


def test_portable_work_unit_uses_explicit_artifacts_and_rejects_path_fallback(tmp_path):
    dataset = tmp_path / "frozen.sqlite"
    config = tmp_path / "train.toml"
    dataset.write_bytes(b"SQLite format 3\x00portable dataset")
    config.write_text("[training]\nepochs = 1\n", encoding="utf-8")
    store = LocalArtifactStore(tmp_path / "store")
    dataset_ref = artifact_ref_for_file("dataset", "dataset-1", dataset, revision="1")
    config_ref = artifact_ref_for_file("configuration", "config-1", config)
    store.ingest_file(dataset_ref, dataset)
    store.ingest_file(config_ref, config)

    unit = build_work_unit(
        work_unit_id=str(uuid.uuid4()), attempt_id=str(uuid.uuid4()), specification_id=str(uuid.uuid4()),
        action="train", parameters={
            "input": str(dataset), "config": str(config),
            "artifact_inputs": {"input": dataset_ref.to_dict()},
            "configuration_artifact": config_ref.to_dict(),
        }, resources={}, require_portable=True,
    )
    assert unit.inputs == {"input": dataset_ref}
    assert unit.configuration == config_ref
    assert str(tmp_path) not in unit.canonical_json()
    assert (store.resolve(dataset_ref) / "payload").read_bytes() == dataset.read_bytes()
    assert (store.resolve(config_ref) / "manifest.json").is_file()

    with pytest.raises(ValueError, match="artifact_inputs.input"):
        build_work_unit(
            work_unit_id=str(uuid.uuid4()), attempt_id=str(uuid.uuid4()), specification_id=str(uuid.uuid4()),
            action="train", parameters={"input": str(dataset)}, resources={}, require_portable=True,
        )


def test_portable_file_ingestion_rejects_symlink_and_digest_mismatch(tmp_path):
    source = tmp_path / "dataset.sqlite"
    source.write_bytes(b"immutable")
    store = LocalArtifactStore(tmp_path / "store")
    ref = artifact_ref_for_file("dataset", "dataset-1", source)
    source.write_bytes(b"changed after reference")
    with pytest.raises(ValueError, match="fingerprint"):
        store.ingest_file(ref, source)
    link = tmp_path / "dataset-link.sqlite"
    link.symlink_to(source)
    with pytest.raises(ValueError, match="non-symlink"):
        artifact_ref_for_file("dataset", "dataset-2", link)


def _v2_unit(**overrides):
    values = {
        "run_id": str(uuid.uuid4()),
        "work_unit_id": str(uuid.uuid4()),
        "sequence": 2,
        "phase": "train",
        "predecessor_work_unit_id": str(uuid.uuid4()),
        "inputs": {
            "dataset": ArtifactRef("dataset", "dataset-1", "r1", "a" * 64),
            "split_manifest": ArtifactRef("split_manifest", "split-1", "r1", "b" * 64),
            "checkpoint": ArtifactRef("checkpoint", "checkpoint-1", "r1", "c" * 64),
        },
        "configuration": ArtifactRef("configuration", "config-1", "r1", "d" * 64),
        "resources": {"gpu_count": 1},
        "parameters": {"epoch": 3},
        "start_cursor": {"epoch": 2, "global_step": 200},
        "stop_boundary": {"kind": "epoch", "epoch": 3},
        "output_contract": {"checkpoint": {"required": True}},
        "compatibility": {"framework": "tensorflow", "minimum_worker_protocol": 2},
    }
    values.update(overrides)
    return WorkUnitV2(**values)


def test_v2_work_unit_is_attempt_free_pinned_and_canonical():
    unit = _v2_unit()
    encoded = unit.to_dict()
    assert "attempt_id" not in encoded
    assert "staging" not in encoded
    assert WorkUnitV2.from_dict(encoded) == unit
    assert parse_work_unit(encoded) == unit
    assert WorkUnitV2.from_dict(encoded).canonical_json() == unit.canonical_json()
    assert len(unit.sha256) == 64
    assert encoded["schema"] == {"name": "oracle_work_unit", "version": 2}
    assert encoded["action"] == "train"
    assert unit.action == "train"
    assert _v2_unit(phase="infer").action == "infer"


@pytest.mark.parametrize("field,value,match", [
    ("sequence", -1, "non-negative"),
    ("sequence", True, "non-negative"),
    ("phase", "package", "Unsupported V2"),
    ("resources", {"limit": float("nan")}, "NaN or Infinity"),
    ("parameters", {"limit": float("inf")}, "NaN or Infinity"),
    ("inputs", {"dataset": ArtifactRef("dataset", "unsealed")}, "fingerprint"),
    ("configuration", ArtifactRef("configuration", "unsealed"), "fingerprint"),
])
def test_v2_work_unit_rejects_unpinned_and_non_strict_contract_values(field, value, match):
    with pytest.raises(WorkUnitError, match=match):
        _v2_unit(**{field: value})


def test_v2_parser_rejects_unknown_and_missing_typed_fields_without_relaxing_v1():
    encoded = _v2_unit().to_dict()
    with_extra = {**encoded, "attempt_id": str(uuid.uuid4())}
    with pytest.raises(WorkUnitError, match="unknown fields"):
        WorkUnitV2.from_dict(with_extra)
    missing = dict(encoded)
    missing.pop("output_contract")
    with pytest.raises(WorkUnitError, match="missing fields"):
        WorkUnitV2.from_dict(missing)
    with pytest.raises(WorkUnitError, match="action must be"):
        WorkUnitV2.from_dict({**encoded, "action": "infer"})

    v1 = WorkUnit(
        work_unit_id=str(uuid.uuid4()), attempt_id=str(uuid.uuid4()), specification_id=str(uuid.uuid4()),
        action="train", inputs={"dataset": ArtifactRef("dataset", "legacy")}, configuration=None,
        staging=ArtifactRef("staging", "stage"), resources={},
    )
    assert parse_work_unit(v1.to_dict()) == v1


def test_execution_envelope_is_fenced_and_excludes_bearer_or_staging_fields():
    unit = _v2_unit()
    envelope = ExecutionEnvelope(
        work_unit_id=unit.work_unit_id, work_unit_sha256=unit.sha256, attempt_id=str(uuid.uuid4()),
        generation=4, worker_id="worker-1", worker_boot_id=str(uuid.uuid4()), lease_id=str(uuid.uuid4()),
        lease_expires_at="2026-09-29T12:00:00+00:00",
    )
    encoded = envelope.to_dict()
    assert ExecutionEnvelope.from_dict(encoded) == envelope
    assert "lease_token" not in encoded
    assert "staging" not in encoded
    with pytest.raises(WorkUnitError, match="unknown fields"):
        ExecutionEnvelope.from_dict({**encoded, "lease_token": "secret"})
    with pytest.raises(WorkUnitError, match="SHA-256"):
        ExecutionEnvelope(
            work_unit_id=unit.work_unit_id, work_unit_sha256="not-a-digest", attempt_id=str(uuid.uuid4()),
            generation=0, worker_id="worker-1", worker_boot_id=str(uuid.uuid4()), lease_id=str(uuid.uuid4()),
            lease_expires_at="2026-09-29T12:00:00+00:00",
        )
