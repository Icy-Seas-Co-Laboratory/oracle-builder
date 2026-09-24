import json

import numpy as np
import pytest

from oracle_builder.posthoc.cache import export_embedding_cache, open_embedding_cache


def test_cache_round_trip_is_memmapped_and_preserves_provenance(tmp_path):
    values = np.arange(24, dtype=np.float32).reshape(6, 4)
    cache = export_embedding_cache(
        tmp_path / "cache", values,
        records=[{"sample_id": f"s{i}", "split": "test"} for i in range(6)],
        representation="image_embedding", provenance={"run_id": "run-42", "model_hash": "abc"},
        prefer_parquet=False,
    )
    assert isinstance(cache.embeddings, np.memmap)
    assert np.array_equal(cache.embeddings, values)
    assert cache.manifest["provenance"]["run_id"] == "run-42"
    assert cache.records()[3]["sample_id"] == "s3"
    assert [chunk.shape for _, chunk in cache.iter_batches(4)] == [(4, 4), (2, 4)]
    reopened = open_embedding_cache(tmp_path / "cache")
    assert reopened.shape == (6, 4)


def test_cache_rejects_mismatched_records_and_existing_target(tmp_path):
    values = np.ones((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="records"):
        export_embedding_cache(tmp_path / "cache", values, records=[{}])
    export_embedding_cache(tmp_path / "cache", values)
    with pytest.raises(FileExistsError):
        export_embedding_cache(tmp_path / "cache", values)


def test_cache_manifest_detects_tampered_embedding(tmp_path):
    cache = export_embedding_cache(tmp_path / "cache", np.ones((2, 2), dtype=np.float32))
    manifest_path = cache.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["shape"] = [2, 3]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="does not match"):
        open_embedding_cache(cache.path)
