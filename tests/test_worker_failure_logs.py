import gzip
import json
from types import SimpleNamespace

import pytest

from oracle_builder.worker.supervisor import execute_supervised
from oracle_builder.worker import supervisor


def test_failed_child_retains_bounded_diagnostics_outside_attempt_scratch(tmp_path):
    scratch = tmp_path / 'attempt'
    staging = scratch / 'staging'
    staging.mkdir(parents=True)
    control = SimpleNamespace(control_file=scratch / 'control.json', check=lambda: None)

    with pytest.raises(RuntimeError, match=r'Execution child exited.*diagnostic log:'):
        execute_supervised(
            work_unit={'action': 'train'}, inputs={}, configuration=None,
            staging=staging, emit_event=lambda *_args: None, control=control,
        )

    failure = json.loads((scratch / 'executor-result.json').read_text())['execution_error']
    assert failure['type']
    saved = list((tmp_path / 'failure-logs').glob('*.log.gz'))
    assert len(saved) == 1
    assert 'Traceback' in gzip.decompress(saved[0].read_bytes()).decode()
    assert saved[0].stat().st_mode & 0o777 == 0o600


def test_retained_failure_logs_have_byte_and_count_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, '_FAILURE_LOG_TAIL_BYTES', 16)
    monkeypatch.setattr(supervisor, '_FAILURE_LOG_LIMIT', 2)
    for number in range(3):
        scratch = tmp_path / f'attempt-{number}'
        scratch.mkdir()
        (scratch / 'executor.log').write_bytes(b'ignored prefix' + bytes([65 + number]) * 20)
        saved = supervisor._retain_failure_log(scratch)
        assert saved is not None
        assert gzip.decompress(saved.read_bytes()) == bytes([65 + number]) * 16
    assert len(list((tmp_path / 'failure-logs').glob('*.log.gz'))) == 2
