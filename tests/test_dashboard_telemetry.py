import json

import pytest
from fastapi.testclient import TestClient
from oracle_builder.orchestration.api import create_app
from oracle_builder.orchestration.service import Orchestrator


def test_compact_telemetry_retains_latest_per_type_without_changing_ledger(tmp_path):
    orchestrator = Orchestrator(tmp_path / 'state.sqlite', workspace_root=tmp_path)
    now = '2026-11-18T00:00:00+00:00'
    with orchestrator._connection() as db:
        db.execute('INSERT INTO jobs(job_id,oracle_serve_url,action,parameters_json,resources_json,status,submitted_at,updated_at) VALUES(?,?,?,?,?,?,?,?)',
                   ('run', 'pull://worker', 'train', '{}', '{}', 'running', now, now))
        for sequence, kind, data in [(1, 'started', {}), (2, 'worker.training_progress', {'metrics': {'loss': 1}}), (3, 'worker.training_progress', {'metrics': {'loss': .5}, 'history': [{'epoch': 1, 'loss': .5}], 'trace': [{'batch': 2, 'loss': .4}]})]:
            db.execute('INSERT INTO job_events VALUES(?,?,?,?,?,?)', ('run', sequence, now, kind, kind, json.dumps(data)))
    compact = orchestrator.job_telemetry('run')
    assert [event['sequence'] for event in compact['events']] == [1, 3]
    assert all('data_json' not in event for event in compact['events'])
    assert compact['events'][-1]['data']['history'][0]['epoch'] == 1
    assert len(orchestrator.job_events('run')) == 3
    assert compact['summary'] == {'queue_seconds': None, 'runtime_seconds': None}
    with pytest.raises(KeyError):
        orchestrator.job_telemetry('missing')
    with TestClient(create_app(orchestrator)) as client:
        assert client.get('/v1/jobs/run/telemetry').json() == compact
        assert client.get('/v1/jobs/missing/telemetry').status_code == 404
