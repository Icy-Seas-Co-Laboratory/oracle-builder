"""Clock that advances only while the host can run worker heartbeats."""
from __future__ import annotations

import time


def awake_monotonic() -> float:
    # macOS exposes an uptime clock that excludes system sleep. On Linux,
    # CLOCK_MONOTONIC also excludes suspend; other platforms use Python's
    # monotonic clock as the best available approximation.
    clock = getattr(time, "CLOCK_UPTIME_RAW", None)
    if clock is not None:
        return time.clock_gettime(clock)
    return time.monotonic()
