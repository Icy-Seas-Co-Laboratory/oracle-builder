from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", None)
    except Exception:
        return None


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def collect_environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "tensorflow_version": _version("tensorflow"),
        "keras_version": _version("keras"),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "os": {"system": platform.system(), "release": platform.release(), "version": platform.version()},
        "hostname": socket.gethostname(),
        "git_commit": _git_commit(),
        "cuda_available": None,
        "gpus": [],
    }
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        info["cuda_available"] = bool(gpus)
        info["gpus"] = [str(gpu) for gpu in gpus]
    except Exception as exc:
        info["cuda_error"] = str(exc)
    return info


def write_environment(run_dir: str | Path) -> dict[str, Any]:
    env = collect_environment()
    Path(run_dir, "environment.json").write_text(json.dumps(env, indent=2, sort_keys=True) + "\n")
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    except Exception as exc:
        freeze = f"# pip freeze failed: {exc}\n"
    Path(run_dir, "requirements_freeze.txt").write_text(freeze)
    return env

