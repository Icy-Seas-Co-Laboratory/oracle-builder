"""Bounded diagnostics for the representation saved by an SSL-only run."""
from __future__ import annotations

from typing import Any
import copy
import numpy as np

from oracle_builder.clustering.training import _serving_embedding_health
from oracle_builder.data.sqlite_stream import SQLiteSplitIndex


def inspect_embedding_health(model: Any, source: Any, index: SQLiteSplitIndex, config: dict[str, Any]) -> dict[str, Any]:
    settings = config.get('self_supervised', config.get('pretraining', {})).get('embedding_health', {})
    if not settings.get('enabled', True):
        return {'status': 'disabled'}
    count = min(len(index.refs), int(settings.get('sample_count', 4096)))
    if count < 2:
        return {'status': 'unavailable', 'reason': 'At least two examples are required', 'sample_count': count}
    positions = np.arange(count) * len(index.refs) // count
    sample = SQLiteSplitIndex(index.sqlite_path, index.split, [index.refs[int(position)] for position in positions])
    values = [np.asarray(model(images, training=False)) for images in source.image_dataset(
        sample, batch_size=int(config.get('data', {}).get('batch_size', 16)), shuffle=False)]
    try:
        diagnostic_config = copy.deepcopy(config)
        diagnostic_ssl = diagnostic_config.setdefault('self_supervised', dict(config.get('pretraining', {})))
        diagnostic_ssl.setdefault('collapse_embedding_spread_threshold', 0.005)
        diagnostics = _serving_embedding_health(np.concatenate(values), diagnostic_config)
    except ValueError as exc:
        return {'status': 'warning', 'reason': str(exc), 'sample_count': count}
    thresholds = diagnostics['thresholds']
    failures = [metric for metric, threshold, minimum in (
        ('effective_rank', 'minimum_effective_rank', True),
        ('centered_rms', 'minimum_centered_rms', True),
        ('mean_pairwise_cosine', 'maximum_mean_pairwise_cosine', False),
        ('p95_pairwise_cosine', 'maximum_p95_pairwise_cosine', False),
        ('mean_direction_norm', 'maximum_mean_direction_norm', False),
    ) if (diagnostics[metric] < thresholds[threshold] if minimum else diagnostics[metric] > thresholds[threshold])]
    return {**diagnostics, 'status': 'warning' if failures else 'passed', 'threshold_failures': failures}
