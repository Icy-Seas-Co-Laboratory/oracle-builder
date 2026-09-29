"""Ordered, immutable work units with transactionally committed continuations."""
from __future__ import annotations
import json
import hashlib
import uuid
from pathlib import Path

from oracle_data_contracts.work_units import WorkUnitV2
from oracle_data_contracts.artifacts import ArtifactRef


class SequentialExecutionMixin:
    def _begin_sequential_in_connection(self, db, *, unit, queued_run_id, pool_id, now):
        execution = unit.parameters['queue_execution']
        total = execution['epochs']
        size = execution.get('unit_epochs', 1)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (total, size)):
            raise ValueError('Training segment epochs must be positive integers')
        run_id = unit.specification_id
        steps = execution.get('unit_steps', 0)
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError('Work unit steps must be a non-negative integer')
        step_fields = {'max_steps': steps, 'start_step': 0, 'start_global_step': 0} if steps else {}
        work = WorkUnitV2(run_id=run_id, work_unit_id=unit.work_unit_id, sequence=0, phase='train', predecessor_work_unit_id=None,
            inputs=unit.inputs, configuration=unit.configuration, resources=unit.resources,
            parameters={**unit.parameters, 'queue_execution': {**execution, 'revalidate_fixed_batch': True}, 'execution_segment': {'phase': 'train', 'start_epoch': 0, 'stop_epoch': min(size, total), 'total_epochs': total, **step_fields}},
            start_cursor={'completed_epoch': 0, **({'batch_offset': 0, 'global_step': 0} if steps else {})}, stop_boundary={'completed_epoch': min(size, total), **({'max_steps': steps} if steps else {})},
            output_contract={'kind': 'training_checkpoint', 'schema_version': 1},
            compatibility={'work_unit_v2': True, 'worker_control_v1': True, 'training_segments_v1': True})
        db.execute('INSERT INTO execution_runs VALUES(?,?,?,?,?,?,0,?,?,NULL,?,?)',
            (run_id, unit.specification_id, queued_run_id, pool_id, 'queued', unit.work_unit_id, total, size, now, now))
        return work

    def execution_run(self, run_id):
        with self._connection() as db:
            row = db.execute('SELECT * FROM execution_runs WHERE run_id=?', (run_id,)).fetchone()
            if not row:
                return None
            result = dict(row)
            result['checkpoint_ref'] = json.loads(result.pop('checkpoint_ref_json')) if result.get('checkpoint_ref_json') else None
            result['units'] = [dict(job) for job in db.execute('SELECT job_id,status,work_unit_json FROM jobs WHERE specification_id=? ORDER BY submitted_at', (row['specification_id'],))]
            for unit in result['units']:
                unit['work_unit'] = json.loads(unit.pop('work_unit_json'))
            return result

    def _validate_training_output_inputs(self, path, unit):
        root = Path(path)
        manifest = json.loads((root / 'artifact.json').read_text())
        inputs = unit.get('inputs', {})
        dataset = inputs.get('input')
        if dataset:
            recorded = manifest.get('dataset', {})
            if recorded.get('dataset_id') != dataset['artifact_id'] or (dataset.get('revision') and recorded.get('revision_id') != dataset['revision']):
                raise ValueError('Training output belongs to a different dataset revision')
        if inputs.get('checkpoint'):
            previous = self.artifact_store.resolve(ArtifactRef.from_dict(inputs['checkpoint']))
            identity = json.loads((previous / 'artifact.json').read_text())
            if any(manifest.get(key) != identity.get(key) for key in ('artifact_id', 'run_id')):
                raise ValueError('Training continuation changed its checkpoint run identity')
        if inputs.get('split_manifest'):
            from oracle_data_contracts.artifacts.splits import read_split_manifest
            pinned = self.artifact_store.resolve(ArtifactRef.from_dict(inputs['split_manifest']))
            if read_split_manifest(root)['fingerprint_sha256'] != read_split_manifest(pinned)['fingerprint_sha256']:
                raise ValueError('Training output changed its verified split manifest')
        execution = unit.get('parameters', {}).get('queue_execution', {})
        if execution.get('mode') == 'verified':
            from oracle_builder.artifacts import read_run_config
            config = read_run_config(root)
            if config.get('data', {}).get('batch_size') != execution['batch_size']:
                raise ValueError('Training output changed its verified batch size')

    @staticmethod
    def _read_segment_result(path, unit):
        candidate = Path(path) / 'segment-result.json'
        if not candidate.is_file() or candidate.stat().st_size > 65536:
            raise ValueError('Training segment must provide a bounded result manifest')
        result = json.loads(candidate.read_text())
        if result.get('schema') != 'oracle_builder_training_segment_result/v1' or result.get('phase') != 'train':
            raise ValueError('Unsupported training segment result schema or phase')
        cursor = result.get('cursor', {}).get('completed_epoch')
        if isinstance(cursor, bool) or not isinstance(cursor, int):
            raise ValueError('Training checkpoint cursor is invalid')
        root = Path(path)
        ordinary = root / 'model/recovery/state.json'
        shared = root / 'model/recovery/stratified_state.json'
        state_path = ordinary if ordinary.is_file() else shared
        if not state_path.is_file() or state_path.stat().st_size > 1024 * 1024:
            raise ValueError('Training segment has no bounded recovery state')
        state = json.loads(state_path.read_text())
        manifest = json.loads((root / 'artifact.json').read_text())
        if any(state.get(key) != manifest.get(key) for key in ('artifact_id', 'run_id')):
            raise ValueError('Checkpoint state belongs to another run artifact')
        model = state if ordinary.is_file() else state.get('shared', {})
        model_path = model.get('model_path', '')
        target = root / model_path
        if not model_path or not target.resolve().is_relative_to(root.resolve()) or not target.is_file():
            raise ValueError('Checkpoint model path is invalid')
        digest = hashlib.sha256()
        with target.open('rb') as data:
            for chunk in iter(lambda: data.read(1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != model.get('model_sha256'):
            raise ValueError('Checkpoint model digest mismatch')
        recorded_epoch = state.get('completed_epoch') if ordinary.is_file() else min((entry.get('completed_epochs', 0) for entry in state.get('children', {}).values()), default=-1)
        if recorded_epoch != cursor:
            raise ValueError('Result cursor does not match committed recovery state')
        early_stop = (state.get('callback_state', {}).get('values', {}).get('early_stopping', {}).get('stopped_epoch', 0) > 0
            if ordinary.is_file() else state.get('cycle_scheduler', {}).get('stopped_early') is True)
        if result.get('stopped_early') and not early_stop:
            raise ValueError('Early stop is not supported by checkpoint state')
        segment = unit['parameters']['execution_segment']
        if unit['phase'] == 'train':
            if segment.get('max_steps'):
                global_step = result['cursor'].get('global_step')
                start_step = segment.get('start_global_step', 0)
                batch_offset = result['cursor'].get('batch_offset')
                if (isinstance(global_step, bool) or not isinstance(global_step, int)
                        or not start_step < global_step <= start_step + segment['max_steps']
                        or isinstance(batch_offset, bool) or not isinstance(batch_offset, int) or batch_offset < 0
                        or not segment['start_epoch'] <= cursor <= segment['stop_epoch']):
                    raise ValueError('Step checkpoint cursor lies outside the sealed work budget')
                if any(state.get('cursor', {}).get(key) != result['cursor'].get(key) for key in ('batch_offset', 'global_step', 'epoch')):
                    raise ValueError('Step result disagrees with committed recovery cursor')
            elif not segment['start_epoch'] < cursor <= segment['stop_epoch']:
                raise ValueError('Training result cursor lies outside its sealed unit boundary')
            if not segment.get('max_steps') and cursor != segment['stop_epoch'] and not (result.get('interrupted') or result.get('stopped_early')):
                raise ValueError('Training segment did not reach its sealed stop boundary')
        if result.get('next_phase') not in {'train', 'finalize', 'complete'}:
            raise ValueError('Training segment next phase is invalid')
        expected_phase = 'finalize' if cursor >= segment['total_epochs'] or early_stop else 'train'
        if result.get('next_phase') != expected_phase:
            raise ValueError('Worker next phase disagrees with authoritative training progress')
        return result

    def _commit_segment_in_connection(self, db, *, job, lease_id, output_ref, result, now):
        unit = json.loads(job['work_unit_json'])
        run = db.execute('SELECT * FROM execution_runs WHERE run_id=?', (unit['run_id'],)).fetchone()
        if not run or run['current_job_id'] != job['job_id']:
            raise ValueError('Stale training unit cannot advance the run')
        cursor = result['cursor']['completed_epoch']
        pending = db.execute("SELECT * FROM worker_commands WHERE lease_id=? AND status IN ('pending','received','accepted') ORDER BY sequence DESC LIMIT 1", (lease_id,)).fetchone()
        if pending and pending['action'] in {'stop_now', 'restart'}:
            raise ValueError('Stopped attempt cannot advance training')
        phase = 'finalize' if cursor >= run['total_epochs'] or result.get('stopped_early') else 'train'
        paused = bool(pending and pending['action'] == 'pause')
        next_id = str(uuid.uuid4())
        inputs = {key: ArtifactRef.from_dict(ref) for key, ref in unit['inputs'].items()}
        inputs['checkpoint'] = output_ref
        end = min(cursor + run['unit_epochs'], run['total_epochs'])
        step_fields = {}
        prior_segment = unit['parameters']['execution_segment']
        if prior_segment.get('max_steps') and phase == 'train':
            step_fields = {'max_steps': prior_segment['max_steps'], 'start_step': result['cursor']['batch_offset'], 'start_global_step': result['cursor']['global_step']}
        successor = WorkUnitV2(run_id=unit['run_id'], work_unit_id=next_id, sequence=unit['sequence']+1,
            phase=phase, predecessor_work_unit_id=job['job_id'], inputs=inputs,
            configuration=ArtifactRef.from_dict(unit['configuration']), resources=unit['resources'],
            parameters={**unit['parameters'], 'execution_segment': {'phase': phase, 'start_epoch': cursor, 'stop_epoch': end, 'total_epochs': run['total_epochs'], **step_fields}},
            start_cursor=result['cursor'], stop_boundary={'completed_epoch': end, **({'max_steps': step_fields['max_steps']} if step_fields else {})},
            output_contract={'kind': 'model_run' if phase == 'finalize' else 'training_checkpoint', 'schema_version': 1}, compatibility=unit['compatibility'])
        status = 'paused' if paused else 'queued'
        db.execute('INSERT INTO jobs(job_id,specification_id,oracle_serve_url,action,parameters_json,resources_json,work_unit_json,work_unit_sha256,worker_pool_id,queued_run_id,status,remote_status,submitted_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
            (next_id, job['specification_id'], job['oracle_serve_url'], 'train', '{}', job['resources_json'], successor.canonical_json(), successor.sha256,
             job['worker_pool_id'], job['queued_run_id'], status, status, now, now))
        db.execute('UPDATE execution_runs SET current_job_id=?,completed_epoch=?,checkpoint_ref_json=?,status=?,updated_at=? WHERE run_id=?',
            (next_id, cursor, json.dumps(output_ref.to_dict()), status, now, unit['run_id']))
        if job['queued_run_id']:
            db.execute('UPDATE queued_runs SET status=?,updated_at=? WHERE queued_run_id=?', (status, now, job['queued_run_id']))
        if pending:
            db.execute("UPDATE worker_commands SET status='applied',result_json=?,updated_at=? WHERE command_id=?", (json.dumps({'checkpoint_ref': output_ref.to_dict(), 'cursor': result['cursor'], 'successor_job_id': next_id}), now, pending['command_id']))
        self._record_job_event_in_connection(db, job['job_id'], 'checkpoint_committed', 'Durable checkpoint committed and continuation scheduled', {'next_job_id': next_id, 'cursor': result['cursor'], 'status': status})
        self._record_job_event_in_connection(db, next_id, 'queued', 'Training continuation created from committed checkpoint', {'predecessor_job_id': job['job_id']})
