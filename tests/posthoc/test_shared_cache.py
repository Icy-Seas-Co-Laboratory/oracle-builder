import json
import numpy as np

from oracle_builder.posthoc.cache import export_embedding_cache
from oracle_builder.posthoc.shared_cache import reusable_representation_cache


def _write(root):
    for split in ('train', 'validation', 'test'):
        export_embedding_cache(root / 'intermediates' / 'representations' / split / 'fused_embedding',
                               np.array([[1., 2.]]), records=[{'label': 0, 'sample_id': 'a'}], prefer_parquet=False)


def test_shared_embeddings_reused_only_with_matching_identity(tmp_path):
    shared = tmp_path / 'shared'
    with reusable_representation_cache(tmp_path / 'one', shared, {'model': 'm', 'fingerprint': 'first', 'split': 's'}):
        _write(tmp_path / 'one')
    (tmp_path / 'two').mkdir()
    (tmp_path / 'two/artifact.json').write_text(json.dumps({'artifact_id': 'second', 'run_id': 'run2'}))
    with reusable_representation_cache(tmp_path / 'two', shared, {'model': 'm', 'fingerprint': 'first', 'split': 's'}):
        assert (tmp_path / 'two/intermediates/representations/test/fused_embedding/embeddings.npy').is_file()
        manifest = json.loads((tmp_path / 'two/intermediates/representations/test/fused_embedding/manifest.json').read_text())
        assert manifest['provenance']['cache_reuse']['consumer_artifact_id'] == 'second'
        assert manifest['provenance']['cache_reuse']['identity'] == {'model': 'm', 'fingerprint': 'first', 'split': 's'}
    with reusable_representation_cache(tmp_path / 'three', shared, {'model': 'm', 'fingerprint': 'different-content', 'split': 's'}):
        assert not (tmp_path / 'three/intermediates').exists()


def test_failed_fit_does_not_publish_cache(tmp_path):
    import pytest
    shared = tmp_path / 'shared'
    with pytest.raises(RuntimeError), reusable_representation_cache(tmp_path / 'one', shared, {'model': 'm'}):
        _write(tmp_path / 'one')
        raise RuntimeError('fit failed')
    assert not list(shared.glob('*/train'))
