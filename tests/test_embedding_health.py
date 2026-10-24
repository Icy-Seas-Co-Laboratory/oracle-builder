from pathlib import Path
from types import SimpleNamespace

import numpy as np

from oracle_builder.embedding.health import inspect_embedding_health
from oracle_builder.data.sqlite_stream import SQLiteSplitIndex


def test_health_sampling_is_bounded_and_deterministic():
    seen = []
    class Source:
        def image_dataset(self, index, **kwargs):
            seen.extend(index.refs)
            yield np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    result = inspect_embedding_health(lambda images, **kwargs: images, Source(),
        SQLiteSplitIndex(Path('unused'), 'all', list(range(100))),
        {'run': {'seed': 1}, 'self_supervised': {'collapse_embedding_spread_threshold': .005,
            'embedding_health': {'enabled': True, 'sample_count': 4, 'pair_count': 20, 'minimum_effective_rank': 1}}})
    assert seen == [0, 25, 50, 75]
    assert result['sample_count'] == 4
    assert result['effective_rank'] == 2
    assert result['status'] in {'passed', 'warning'}


def test_disabled_health_does_not_run_inference():
    result = inspect_embedding_health(None, None, None,
        {'self_supervised': {'embedding_health': {'enabled': False}}})
    assert result == {'status': 'disabled'}
