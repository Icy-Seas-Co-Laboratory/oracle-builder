"""Fixed subprocess entrypoint; accepts only parent-created local requests."""
from __future__ import annotations
import json
from pathlib import Path
import sys

EVENT_PREFIX = 'ORACLE_WORKER_EVENT='


def main():
    from oracle_builder.worker.pull import TrainingExecutor, InferenceExecutor, DeterministicPackagingExecutor
    request_path = Path(sys.argv[1])
    request = json.loads(request_path.read_text())
    def emit(event, message, data=None):
        print(EVENT_PREFIX + json.dumps({'event': event, 'message': message, 'data': data or {}}, allow_nan=False), flush=True)
    executors = {'train': TrainingExecutor, 'infer': InferenceExecutor, 'package': DeterministicPackagingExecutor}
    action = request['work_unit']['action']
    executor = executors[action]()
    # This control path is generated locally and never comes from a WorkUnit.
    executor.control_file = request.get('control_file')
    result = executor.execute(work_unit=request['work_unit'], inputs={key: Path(value) for key,value in request['inputs'].items()},
        configuration=Path(request['configuration']) if request.get('configuration') else None,
        staging=Path(request['staging']), emit_event=emit)
    Path(request['result_path']).write_text(json.dumps(result or {}, allow_nan=False))


if __name__ == '__main__':
    main()
