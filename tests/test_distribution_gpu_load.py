from oracle_builder.training.distribution import _GPU_LEASES, _select_auto_gpu, gpu_lease_scope


def test_auto_gpu_selector_allows_light_sharing_and_prefers_less_loaded(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "oracle_builder.training.distribution._gpu_loads",
        lambda: {0: (400, 4), 1: (2500, 3)},
    )
    with gpu_lease_scope():
        device, lease = _select_auto_gpu(
            ["/GPU:0", "/GPU:1"],
            {
                "gpu_lease_directory": str(tmp_path),
                "gpu_light_share_memory_mb": 1024,
                "gpu_light_share_utilization_percent": 15,
                "allow_busy_fallback": False,
            },
        )
        assert device == "/GPU:0"
        assert lease is not None
        assert lease in _GPU_LEASES
    assert lease not in _GPU_LEASES


def test_gpu_scope_releases_after_success_and_failure_and_preserves_outer(tmp_path):
    import pytest
    from oracle_builder.training.distribution import _reserve_gpu, _GPU_LEASES, gpu_lease_scope
    settings = {'gpu_lease_directory': str(tmp_path)}
    with gpu_lease_scope():
        outer = _reserve_gpu(0, settings)
        assert outer
        with pytest.raises(RuntimeError, match='probe failed'):
            with gpu_lease_scope():
                assert _reserve_gpu(0, settings) is None
                inner = _reserve_gpu(1, settings)
                assert inner
                raise RuntimeError('probe failed')
        assert inner not in _GPU_LEASES
        assert outer in _GPU_LEASES
    assert outer not in _GPU_LEASES
    for _ in range(2):
        with gpu_lease_scope():
            assert _reserve_gpu(0, settings)
