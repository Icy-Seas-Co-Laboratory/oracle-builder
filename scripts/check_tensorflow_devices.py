#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import tempfile
import traceback
from pathlib import Path
from importlib import metadata


PACKAGES = ("tensorflow", "tensorflow-metal", "keras", "numpy", "protobuf")


def main() -> int:
    ensure_writable_caches()
    payload = {
        "platform": platform.platform(),
        "packages": installed_versions(),
    }
    try:
        import tensorflow as tf
    except Exception as exc:
        payload["tensorflow_import_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-8:],
        }
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 2

    payload.update({
        "tensorflow_version": tf.__version__,
        "physical_gpus": [str(device) for device in tf.config.list_physical_devices("GPU")],
        "physical_cpus": [str(device) for device in tf.config.list_physical_devices("CPU")],
        "built_with_cuda": bool(getattr(tf.test, "is_built_with_cuda", lambda: False)()),
    })
    try:
        payload["build_info"] = tf.sysconfig.get_build_info()
    except Exception as exc:
        payload["build_info_error"] = str(exc)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


def ensure_writable_caches() -> None:
    cache_root = Path(tempfile.gettempdir()) / "oracle-builder-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)


def installed_versions() -> dict[str, str | None]:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


if __name__ == "__main__":
    raise SystemExit(main())
