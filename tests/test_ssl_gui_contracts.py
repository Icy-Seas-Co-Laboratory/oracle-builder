import json
import sqlite3
import time

import pytest
from fastapi.testclient import TestClient

from test_pull_queue import queue_setup, finish_verification
from oracle_builder.config import load_toml
from oracle_builder.data.sqlite_dataset import create_synthetic_segmentation
from oracle_builder.datasets.schema import set_dataset_lifecycle
from oracle_builder.orchestration.api import create_app
from oracle_builder.orchestration.ssl import phase_budgets, seal_phase_budgets, ssl_artifact_summary
from oracle_data_contracts.artifacts import ArtifactRef


def _queue_ssl(setup, *, task='classification', epochs=7, ssl_epochs=3):
    orch, pool, worker, _ = setup
    config = {'run': {'task': task}, 'training': {'loss': 'sparse_categorical_crossentropy'}, 'self_supervised': {'enabled': True, 'method': 'simclr'}}
    definition = orch.create_model_definition(name=f'ssl-{task}', template_id='resnet18', config=config)
    dataset = next(item for item in orch.datasets() if item['lifecycle'] == 'frozen')
    queued = orch.queue_model_definition_for_pool(
        definition['definition_id'], name='independent phase budgets', dataset_id=dataset['dataset_id'],
        worker_pool_id=pool['pool_id'], resources={'gpu_count': 0}, batch_size=4,
        epochs=epochs, ssl_epochs=ssl_epochs,
    )
    return definition, dataset, queued


def _sealed(orch, queued):
    spec = orch.specification(queued['specification_id'])
    return load_toml(orch.artifact_store.resolve(ArtifactRef.from_dict(spec['parameters']['configuration_artifact'])) / 'payload')


def test_independent_ssl_budgets_survive_verification_and_replacement(queue_setup):
    orch, pool, worker, _ = queue_setup
    definition, dataset, queued = _queue_ssl(queue_setup)
    assert _sealed(orch, queued)['self_supervised']['epochs'] == 3
    assert _sealed(orch, queued)['training']['epochs'] == 7
    assert queued['total_planned_epochs'] == 10
    assert queued['self_supervised']['method'] == 'simclr'
    orch.verify_queued_runs_for_pool(queued_run_ids=[queued['queued_run_id']])
    finish_verification(orch, worker, report_overrides={'self_supervised_whole_phase': True})
    ready = orch.queued_run(queued['queued_run_id'])
    assert ready['batch_execution']['ssl_epochs'] == 3
    assert ready['batch_execution']['supervised_epochs'] == 7
    replacement = orch.replace_queued_run(queued['queued_run_id'], name='new budgets', dataset_id=dataset['dataset_id'],
        worker_pool_id=pool['pool_id'], resources={'gpu_count': 0}, batch_size_mode='manual', batch_size=4,
        epochs=9, ssl_epochs=2)
    assert replacement['total_planned_epochs'] == 11
    assert _sealed(orch, replacement)['self_supervised']['epochs'] == 2
    orch.verify_queued_runs_for_pool(queued_run_ids=[replacement['queued_run_id']])
    finish_verification(orch, worker, report_overrides={'self_supervised_whole_phase': True})
    orch.authorize_queued_runs_for_pool(queued_run_ids=[replacement['queued_run_id']], worker_pool_id=pool['pool_id'])
    lease = orch.acquire_next_worker_lease(worker_id=worker['worker_id'], worker_token=worker['worker_token'])
    segment = lease['work_unit']['parameters']['execution_segment']
    assert segment['phase'] == 'initialize'
    assert segment['total_epochs'] == 2
    assert segment['supervised_epochs'] == 9
    assert orch.execution_run(lease['work_unit']['run_id'])['total_epochs'] == 9


def test_embedding_queue_runs_complete_ssl_workflow_without_supervised_segments(queue_setup):
    orch, pool, worker, _ = queue_setup
    _, _, queued = _queue_ssl(queue_setup, task='embedding')
    assert queued['supervised_epochs'] == 0
    assert queued['total_planned_epochs'] == 3
    orch.verify_queued_runs_for_pool(queued_run_ids=[queued['queued_run_id']])
    finish_verification(orch, worker)
    assert orch.queued_run(queued['queued_run_id'])['batch_execution']['whole_run'] is True
    orch.authorize_queued_runs_for_pool(queued_run_ids=[queued['queued_run_id']], worker_pool_id=pool['pool_id'])
    lease = orch.acquire_next_worker_lease(worker_id=worker['worker_id'], worker_token=worker['worker_token'])
    assert not lease['work_unit']['parameters'].get('execution_segment')
    assert lease['work_unit']['parameters']['queue_execution']['whole_run'] is True


def test_segmentation_ssl_queue_verifies_then_dispatches_initialization(queue_setup):
    orch, pool, worker, _ = queue_setup
    database = orch.workspace_root / 'segmentation.sqlite'
    create_synthetic_segmentation(database, n=4, shape=(16, 16, 1))
    with sqlite3.connect(database) as connection:
        set_dataset_lifecycle(connection, 'frozen')
        connection.commit()
    dataset = orch.ingest_dataset(database)
    definition = orch.create_model_definition(
        name='segmentation with ssl',
        template_id='mask_refinement_unet',
        config={
            'input': {'channels': ['intensity']},
            'data': {
                'input_shape': [16, 16, 1],
                'output_shape': [16, 16, 1],
                'batch_size': 2,
                'validation_split': 0.0,
                'test_split': 0.0,
                'candidate_distance': 'none',
            },
            'training': {'segmentation_target': 'validated_mask'},
            'self_supervised': {
                'enabled': True,
                'method': 'grayscale_reconstruction',
            },
            'augmentation': {'enabled': False},
            'tiling': {'enabled': False},
        },
    )
    queued = orch.queue_model_definition_for_pool(
        definition['definition_id'],
        name='segmentation phase boundary',
        dataset_id=dataset['dataset_id'],
        worker_pool_id=pool['pool_id'],
        resources={'gpu_count': 0},
        batch_size=2,
        epochs=2,
        ssl_epochs=1,
    )
    orch.verify_queued_runs_for_pool(queued_run_ids=[queued['queued_run_id']])
    finish_verification(
        orch,
        worker,
        batch_size=2,
        report_overrides={'self_supervised_whole_phase': True},
    )
    ready = orch.queued_run(queued['queued_run_id'])
    assert ready['status'] == 'ready'
    assert ready['batch_execution']['ssl_epochs'] == 1
    assert ready['batch_execution']['supervised_epochs'] == 2

    orch.authorize_queued_runs_for_pool(
        queued_run_ids=[queued['queued_run_id']], worker_pool_id=pool['pool_id']
    )
    lease = orch.acquire_next_worker_lease(
        worker_id=worker['worker_id'], worker_token=worker['worker_token']
    )
    segment = lease['work_unit']['parameters']['execution_segment']
    assert segment == {'phase': 'initialize', 'total_epochs': 1, 'supervised_epochs': 2}


def test_queue_api_accepts_ssl_budget_and_rejects_disabled_ssl(queue_setup):
    orch, pool, _, _ = queue_setup
    definition, dataset, _ = _queue_ssl(queue_setup)
    body = {'name': 'API phase budgets', 'dataset_id': dataset['dataset_id'], 'worker_pool_id': pool['pool_id'],
            'resources': {'gpu_count': 0}, 'batch_size': 4, 'epochs': 11, 'ssl_epochs': 2}
    with TestClient(create_app(orch)) as client:
        response = client.post(f"/v1/model-definitions/{definition['definition_id']}:queue", json=body)
        assert response.status_code == 201, response.text
        assert response.json()['ssl_epochs'] == 2
        assert response.json()['supervised_epochs'] == 11
        disabled = orch.create_model_definition(name='no ssl', template_id='resnet18')
        response = client.post(f"/v1/model-definitions/{disabled['definition_id']}:queue", json=body)
        assert response.status_code == 422


def test_scheduled_queue_api_preserves_independent_ssl_and_supervised_budgets(queue_setup):
    orch, pool, _, _ = queue_setup
    definition, dataset, _ = _queue_ssl(queue_setup)
    body = {
        'name': 'Scheduled phase budgets',
        'dataset_id': dataset['dataset_id'],
        'worker_pool_id': pool['pool_id'],
        'resources': {'gpu_count': 0},
        'batch_size_mode': 'manual',
        'batch_size': 4,
        'epochs': 11,
        'ssl_epochs': 2,
    }
    with TestClient(create_app(orch)) as client:
        response = client.post(
            f"/v1/model-definitions/{definition['definition_id']}:validate-and-queue/schedule",
            json=body,
        )
        assert response.status_code == 202, response.text
        operation_id = response.json()['operation']['operation_id']
        for _ in range(100):
            operation = client.get(f'/v1/operations/{operation_id}').json()
            if operation['status'] in {'completed', 'failed'}:
                break
            time.sleep(.01)
        assert operation['status'] == 'completed', operation
        queued = operation['result']['queued_run']
        assert queued['ssl_epochs'] == 2
        assert queued['supervised_epochs'] == 11
        assert queued['total_planned_epochs'] == 13
        sealed = _sealed(orch, queued)
        assert sealed['self_supervised']['epochs'] == 2
        assert sealed['training']['epochs'] == 11


def test_ssl_summary_counts_all_epochs_reads_last_metrics_and_lists_actual_files(tmp_path):
    metrics = tmp_path / 'metrics/pretraining'
    metrics.mkdir(parents=True)
    (metrics / 'metrics.csv').write_text('epoch,loss,encoder_representation_std\n0,3,0.01\n1,2,0.02\n2,1,0.03\n')
    (metrics / 'metrics.json').write_text('{}')
    model = tmp_path / 'model/pretraining'
    model.mkdir(parents=True)
    (model / 'student_pretrained.weights.h5').write_bytes(b'weights')
    summary = ssl_artifact_summary(tmp_path, {'self_supervised': {'enabled': True, 'method': 'simclr', 'epochs': 5}})
    assert summary['planned_epochs'] == 5
    assert summary['completed_epochs'] == 3
    assert summary['latest_metrics']['loss'] == 1
    assert summary['latest_metrics']['encoder_representation_std'] == .03
    assert summary['outputs']['weights'] is True
    assert {item['path'] for item in summary['files']} == {
        'metrics/pretraining/metrics.csv', 'metrics/pretraining/metrics.json', 'model/pretraining/student_pretrained.weights.h5'}


def test_ssl_phase_history_api_is_separate_bounded_and_validated(queue_setup, tmp_path, monkeypatch):
    orch, *_ = queue_setup
    root = tmp_path / 'history-artifact'
    (root / 'metrics/pretraining').mkdir(parents=True)
    (root / 'metrics/pretraining/metrics.csv').write_text('epoch,loss\n0,3\n1,2\n2,1\n')
    (root / 'metrics/history.csv').write_text('epoch,loss\n0,0.5\n')
    monkeypatch.setattr(orch, 'artifact', lambda artifact_id: {'path': str(root)} if artifact_id == 'known' else None)
    with TestClient(create_app(orch)) as client:
        ssl = client.get('/v1/artifacts/known/history?phase=self_supervised&limit=2')
        assert ssl.status_code == 200, ssl.text
        assert ssl.json()['rows'] == [{'epoch': 1, 'loss': 3}, {'epoch': 2, 'loss': 2}]
        assert ssl.json()['truncated'] is True
        assert ssl.json()['self_supervised']['completed_epochs'] == 3
        assert client.get('/v1/artifacts/known/history').json()['rows'][0]['loss'] == .5
        assert client.get('/v1/artifacts/known/history?phase=unknown').status_code == 422
        assert client.get('/v1/artifacts/missing/history?phase=self_supervised').status_code == 404


@pytest.mark.parametrize('task,expected', [('classification', 10), ('segmentation', 10), ('embedding', 3)])
def test_phase_budget_totals_and_legacy_fallback(task, expected):
    config = {'run': {'task': task}, 'training': {'loss': 'sparse_categorical_crossentropy'}, 'self_supervised': {'enabled': True}}
    assert seal_phase_budgets(config, 7, 3)['total_planned_epochs'] == expected
    assert phase_budgets(config)['ssl_epochs'] == 3
    assert seal_phase_budgets(config, 7, None)['ssl_epochs'] == 7
