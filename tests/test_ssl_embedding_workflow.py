import json
import sqlite3

from oracle_builder.data.sqlite_dataset import create_synthetic_classification
from oracle_builder.datasets.schema import set_dataset_lifecycle
from oracle_builder.embedding.training import train_embedding_run
from oracle_builder.orchestration.ssl import ssl_artifact_summary


def test_ssl_only_embedding_seals_metrics_weights_and_health(tmp_path):
    database = tmp_path / 'embedding.sqlite'
    create_synthetic_classification(database, n=4, shape=(16, 16, 1), classes=2)
    with sqlite3.connect(database) as db:
        set_dataset_lifecycle(db, 'frozen')
    recipe = tmp_path / 'embedding.toml'
    recipe.write_text('''
[run]
task = "embedding"
model = "simple_cnn"
[architecture]
version = 1
[data]
input_shape = [16, 16, 1]
batch_size = 2
[model]
base_filters = 2
embedding_dim = 4
dropout = 0.0
[training]
epochs = 7
loss = "sparse_categorical_crossentropy"
display = "off"
[self_supervised]
enabled = true
method = "byol"
epochs = 1
projection_dim = 4
projection_hidden_dim = 8
[self_supervised.embedding_health]
enabled = true
sample_count = 4
pair_count = 16
minimum_effective_rank = 1
[distribution]
strategy = "cpu"
[output]
export_savedmodel = false
''')
    output = tmp_path / 'embedding'
    report = train_embedding_run(recipe, database, output)
    config = json.loads((output / 'config/resolved.json').read_text())
    summary = ssl_artifact_summary(output, config)
    assert summary['planned_epochs'] == 1
    assert summary['completed_epochs'] == 1
    assert summary['outputs']['weights'] is True
    assert summary['health']['sample_count'] == 4
    assert summary['health']['status'] in {'passed', 'warning'}
    assert json.loads((output / 'artifact.json').read_text())['status'] == 'complete'
    assert json.loads((output / 'model/model_manifest.json').read_text())['task'] == 'embedding'
    assert report['self_supervised_health'] == summary['health']
