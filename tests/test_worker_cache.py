from __future__ import annotations

from oracle_builder.worker.cache import WorkerArtifactCache, ArtifactCacheError
from oracle_data_contracts.artifacts import ArtifactRef, directory_content_digest
import threading
import pytest


def test_cache_validates_every_reuse_and_quarantines_corruption(tmp_path):
    source = tmp_path / "source"; (source / "payload").mkdir(parents=True); (source / "payload" / "data").write_text("ok")
    digest = directory_content_digest(source / "payload")
    cache = WorkerArtifactCache(tmp_path / "cache")
    calls = 0
    def download(destination):
        nonlocal calls; calls += 1
        root = destination / "artifact"; root.mkdir(); (root / "payload").mkdir(); (root / "payload" / "data").write_text("ok"); return root
    ref = ArtifactRef("dataset", "data", fingerprint_sha256="a" * 64)
    first = cache.materialize(ref, {"content_manifest_sha256": digest}, download)
    assert first.joinpath("data").read_text() == "ok"
    assert cache.materialize(ref, {"content_manifest_sha256": digest}, download) == first
    assert calls == 1
    first.joinpath("data").write_text("tampered")
    import pytest
    with pytest.raises(ArtifactCacheError, match="quarantined"):
        cache.materialize(ref, {"content_manifest_sha256": digest}, download)
    cache.materialize(ref, {"content_manifest_sha256": digest}, download)
    assert calls == 2 and any(cache.quarantine.iterdir())


def test_cache_quarantines_a_symlinked_payload_before_retrying(tmp_path):
    source = tmp_path / "source"; (source / "payload").mkdir(parents=True); (source / "payload" / "data").write_text("ok")
    digest = directory_content_digest(source / "payload")
    cache = WorkerArtifactCache(tmp_path / "cache")
    ref = ArtifactRef("dataset", "data")

    def download(destination):
        root = destination / "artifact"; root.mkdir(); (root / "payload").mkdir(); (root / "payload" / "data").write_text("ok")
        return root

    payload = cache.materialize(ref, {"content_manifest_sha256": digest}, download)
    payload.joinpath("data").unlink()
    payload.joinpath("data").symlink_to(source / "payload" / "data")
    with pytest.raises(ArtifactCacheError, match="quarantined"):
        cache.materialize(ref, {"content_manifest_sha256": digest}, download)
    assert any(cache.quarantine.iterdir())
    assert cache.materialize(ref, {"content_manifest_sha256": digest}, download).joinpath("data").read_text() == "ok"


def test_cache_rejects_missing_or_bad_grant_digest(tmp_path):
    cache = WorkerArtifactCache(tmp_path / "cache")
    ref = ArtifactRef("dataset", "data")
    try:
        cache.materialize(ref, {}, lambda path: path)
    except ArtifactCacheError as exc:
        assert "pinned" in str(exc)
    else:
        raise AssertionError("missing digest accepted")


def test_cache_pins_entry_during_copy(tmp_path):
    source = tmp_path / "source"; (source / "payload").mkdir(parents=True); (source / "payload" / "data").write_text("ok")
    digest = directory_content_digest(source / "payload"); cache = WorkerArtifactCache(tmp_path / "cache", max_entries=1)
    ref = ArtifactRef("dataset", "data")
    def download(destination):
        root = destination / "artifact"; root.mkdir(); (root / "payload").mkdir(); (root / "payload" / "data").write_text("ok"); return root
    destination = tmp_path / "scratch"
    assert cache.materialize_to(ref, {"content_manifest_sha256": digest}, download, destination) == destination
    assert destination.joinpath("data").read_text() == "ok"


def test_eviction_skips_an_entry_held_by_another_thread(tmp_path):
    cache = WorkerArtifactCache(tmp_path / "cache", max_entries=1)
    def make(name, text):
        root = tmp_path / name; (root / "payload").mkdir(parents=True); (root / "payload" / "data").write_text(text)
        digest = directory_content_digest(root / "payload"); ref = ArtifactRef("dataset", name)
        def download(destination):
            artifact = destination / "artifact"; artifact.mkdir(); (artifact / "payload").mkdir(); (artifact / "payload" / "data").write_text(text); return artifact
        return ref, digest, download
    first, first_digest, first_download = make("one", "one"); second, second_digest, second_download = make("two", "two")
    cache.materialize(first, {"content_manifest_sha256": first_digest}, first_download)
    entered = threading.Event(); release = threading.Event()
    def hold():
        with cache.acquire(first, {"content_manifest_sha256": first_digest}, first_download):
            entered.set(); release.wait(2)
    thread = threading.Thread(target=hold); thread.start(); assert entered.wait(1)
    cache.materialize(second, {"content_manifest_sha256": second_digest}, second_download)
    assert cache.materialize(first, {"content_manifest_sha256": first_digest}, first_download).is_dir()
    release.set(); thread.join(2)
