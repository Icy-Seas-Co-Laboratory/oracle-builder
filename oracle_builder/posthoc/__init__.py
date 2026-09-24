"""Out-of-core embedding artifacts and sklearn representation baselines.

This package deliberately has no TensorFlow dependency.  It is suitable for
analysing a completed run on a smaller machine than the one that trained it.
"""

from .cache import EmbeddingCache, export_embedding_cache, export_embedding_cache_stream, open_embedding_cache
from .pipelines import PosthocConfig, build_posthoc_pipeline
from .probes import ProbeResult, representation_diagnostics, run_probe_suite

__all__ = [
    "EmbeddingCache", "PosthocConfig", "ProbeResult", "build_posthoc_pipeline",
    "export_embedding_cache", "export_embedding_cache_stream", "open_embedding_cache", "representation_diagnostics",
    "run_probe_suite",
]
