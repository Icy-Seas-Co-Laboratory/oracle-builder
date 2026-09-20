from oracle_builder.training.distribution import _select_auto_gpu


def test_auto_gpu_selector_allows_light_sharing_and_prefers_less_loaded(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "oracle_builder.training.distribution._gpu_loads",
        lambda: {0: (400, 4), 1: (2500, 3)},
    )
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
