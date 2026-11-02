"""Ordered, immutable work units with transactionally committed continuations."""
from __future__ import annotations
import json
import hashlib
import uuid
import re
from datetime import datetime, timezone
from pathlib import Path

from oracle_data_contracts.work_units import WorkUnitV2
from oracle_data_contracts.artifacts import ArtifactRef


class SequentialExecutionMixin:
    def _validate_finalization_restart_checkpoint(self, *, checkpoint: dict, unit: dict, completed_epoch: int) -> None:
        """Verify that a failed finalizer can reuse its committed training output."""
        from oracle_builder.artifacts import validate_run_artifact

        path = self.artifact_store.resolve(ArtifactRef.from_dict(checkpoint))
        report = validate_run_artifact(path)
        if not report['valid'] or report['lifecycle'] != 'sealed':
            raise ValueError('Committed training checkpoint is missing or invalid')
        predecessor_id = unit.get('predecessor_work_unit_id')
        with self._connection() as db:
            predecessor = db.execute('SELECT work_unit_json FROM jobs WHERE job_id=?', (predecessor_id,)).fetchone()
        if predecessor is None:
            raise ValueError('Finalization checkpoint has no recorded training predecessor')
        prior_unit = json.loads(predecessor['work_unit_json'])
        result = self._read_segment_result(path, prior_unit)
        if (result['next_phase'] != 'finalize'
                or result['cursor']['completed_epoch'] != completed_epoch
                or report['run_id'] != unit.get('run_id')):
            raise ValueError('Committed checkpoint does not match finalization progress')

    def restart_failed_training_unit(self, job_id: str) -> dict:
        """Manually requeue a failed unit from its last committed checkpoint."""
        with self._connection() as db:
            initial = db.execute('SELECT status FROM jobs WHERE job_id=?', (job_id,)).fetchone()
        if initial is None:
            raise KeyError(job_id)
        if initial['status'] != 'failed':
            raise ValueError('Only a failed training unit can be restarted')
        # A completed filesystem publication wins over a failure receipt.
        # Reconcile before considering another execution of the same unit.
        self.reconcile_worker_publications()
        with self._connection() as db:
            job = db.execute('SELECT * FROM jobs WHERE job_id=?', (job_id,)).fetchone()
            if job is None:
                raise KeyError(job_id)
            if job['status'] in {'completed', 'indexed'}:
                return self.job(job_id) or {}
            unit = json.loads(job['work_unit_json'] or '{}')
            run = db.execute('SELECT * FROM execution_runs WHERE current_job_id=?', (job_id,)).fetchone()
            if run is None or unit.get('schema', {}).get('version') != 2 or unit.get('phase') not in {'initialize', 'train', 'finalize'}:
                raise ValueError('Only the current sequential training unit can be restarted')
            checkpoint = json.loads(run['checkpoint_ref_json']) if run['checkpoint_ref_json'] else None
            # Older execution-run rows may not have retained the checkpoint
            # reference even though the immutable finalizer unit did.
            if checkpoint is None and unit.get('phase') == 'finalize':
                checkpoint = unit.get('inputs', {}).get('checkpoint')
            attempt = db.execute('SELECT classification FROM execution_attempts WHERE job_id=? ORDER BY ordinal DESC LIMIT 1', (job_id,)).fetchone()
            finalization_retry = bool(unit.get('phase') == 'finalize' and attempt is not None and attempt['classification'] == 'scientific')
        if checkpoint is not None:
            self.artifact_store.resolve(ArtifactRef.from_dict(checkpoint))
        if finalization_retry:
            if checkpoint is None:
                raise ValueError('Finalization retry requires a committed training checkpoint')
            self._validate_finalization_restart_checkpoint(
                checkpoint=checkpoint, unit=unit, completed_epoch=int(run['completed_epoch']),
            )
        with self._connection() as db:
            db.execute('BEGIN IMMEDIATE')
            job = db.execute('SELECT * FROM jobs WHERE job_id=?', (job_id,)).fetchone()
            run = db.execute('SELECT * FROM execution_runs WHERE current_job_id=?', (job_id,)).fetchone()
            if job is None or run is None or job['status'] != 'failed':
                raise ValueError('Only a failed current training unit can be restarted')
            unit = json.loads(job['work_unit_json'] or '{}')
            expected = json.loads(run['checkpoint_ref_json']) if run['checkpoint_ref_json'] else None
            if ((expected is not None and expected != checkpoint)
                    or unit.get('inputs', {}).get('checkpoint') != checkpoint
                    or int(unit.get('start_cursor', {}).get('completed_epoch', -1)) != int(run['completed_epoch'])):
                raise ValueError('The failed unit no longer matches the committed checkpoint')
            attempt = db.execute('SELECT classification FROM execution_attempts WHERE job_id=? ORDER BY ordinal DESC LIMIT 1', (job_id,)).fetchone()
            if attempt is None or (attempt['classification'] != 'infrastructure'
                    and not (unit.get('phase') == 'finalize' and attempt['classification'] == 'scientific' and finalization_retry)):
                raise ValueError('Only infrastructure failures or failed finalization can be manually restarted')
            active = db.execute("SELECT 1 FROM worker_leases WHERE job_id=? AND status IN ('active','publication_pending')", (job_id,)).fetchone()
            if active:
                raise ValueError('This unit still has an active or pending publication lease')
            now = datetime.now(timezone.utc).isoformat()
            db.execute("UPDATE jobs SET status='queued',remote_status='queued',worker_id=NULL,error=NULL,completed_at=NULL,retry_count=0,cancel_requested_at=NULL,cancelled_at=NULL,updated_at=? WHERE job_id=?", (now, job_id))
            db.execute("UPDATE execution_runs SET status='queued',updated_at=? WHERE run_id=?", (now, run['run_id']))
            if job['queued_run_id']:
                db.execute("UPDATE queued_runs SET status='queued',start_authorized=1,failure_reason=NULL,updated_at=? WHERE queued_run_id=?", (now, job['queued_run_id']))
            self._record_job_event_in_connection(db, job_id, 'manual_restart',
                'Failed finalization retried from committed training checkpoint' if finalization_retry else 'Failed training unit manually restarted from committed checkpoint',
                {'completed_epoch': run['completed_epoch'], 'checkpoint_ref': checkpoint, 'phase': unit['phase']})
        return self.job(job_id) or {}

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
        whole_phase_ssl = execution.get('self_supervised_whole_phase') is True
        initial_phase = 'initialize' if whole_phase_ssl else 'train'
        initial_segment = (
            {'phase': 'initialize', 'total_epochs': execution.get('ssl_epochs') or total, 'supervised_epochs': total}
            if whole_phase_ssl else
            {'phase': 'train', 'start_epoch': 0, 'stop_epoch': min(size, total), 'total_epochs': total, **step_fields}
        )
        work = WorkUnitV2(run_id=run_id, work_unit_id=unit.work_unit_id, sequence=0, phase=initial_phase, predecessor_work_unit_id=None,
            inputs=unit.inputs, configuration=unit.configuration, resources=unit.resources,
            parameters={**unit.parameters, 'queue_execution': {**execution, 'revalidate_fixed_batch': True}, 'execution_segment': initial_segment},
            start_cursor={'completed_epoch': 0, **({} if whole_phase_ssl else ({'batch_offset': 0, 'global_step': 0} if steps else {}))}, stop_boundary=({'completed_epoch': 0} if whole_phase_ssl else {'completed_epoch': min(size, total), **({'max_steps': steps} if steps else {})}),
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

    @staticmethod
    def _pretraining_initialization_view(row):
        if row is None:
            return None
        result = dict(row)
        for source, target in (("artifact_ref_json", "artifact_ref"),
                               ("dataset_ref_json", "dataset_ref"),
                               ("configuration_ref_json", "configuration_ref")):
            raw = result.pop(source)
            result[target] = json.loads(raw) if raw else None
        raw_config = result.pop("_pinned_config_json", None)
        config = None
        if raw_config:
            try:
                config = json.loads(raw_config)
            except (TypeError, json.JSONDecodeError):
                config = None
        result["task"] = (config.get("run", {}).get("task")
                          if isinstance(config, dict) and isinstance(config.get("run"), dict) else None)
        encoder = config.get("encoder") if isinstance(config, dict) else None
        model = config.get("model") if isinstance(config, dict) else None
        run = config.get("run") if isinstance(config, dict) else None
        result["architecture"] = (
            (encoder.get("family") or encoder.get("architecture")) if isinstance(encoder, dict) else None
        ) or ((model.get("architecture")) if isinstance(model, dict) else None) or (
            run.get("model") if isinstance(run, dict) else None
        )
        settings = config.get("self_supervised") or config.get("pretraining") if isinstance(config, dict) else None
        result["method"] = (
            settings.get("method", "byol")
            if isinstance(settings, dict) and settings.get("enabled") is True else None
        )
        result["status"] = "recorded"
        return result

    @staticmethod
    def _pretraining_initialization_query(where=""):
        return """SELECT p.*,queued_runs.name AS name,datasets.name AS dataset_name,
                   model_definitions.name AS definition_name,
                   model_definition_revisions.config_json AS _pinned_config_json
            FROM pretraining_initializations p
            LEFT JOIN queued_runs ON queued_runs.queued_run_id=p.queued_run_id
            LEFT JOIN datasets ON datasets.dataset_id=queued_runs.dataset_id
            LEFT JOIN model_definitions ON model_definitions.definition_id=queued_runs.definition_id
            LEFT JOIN model_definition_revisions
              ON model_definition_revisions.definition_id=queued_runs.definition_id
             AND model_definition_revisions.revision=queued_runs.definition_revision
            """ + where

    def pretraining_initializations(self):
        """Return published immutable initialization checkpoints for selection."""
        self._backfill_pretraining_initializations()
        with self._connection() as db:
            rows = db.execute(self._pretraining_initialization_query(
                "ORDER BY p.published_at DESC,p.artifact_id"
            )).fetchall()
        return [self._pretraining_initialization_view(row) for row in rows]

    def pretraining_initialization(self, artifact_id):
        self._backfill_pretraining_initializations()
        with self._connection() as db:
            row = db.execute(self._pretraining_initialization_query(
                "WHERE p.artifact_id=?"
            ), (artifact_id,)).fetchone()
        return self._pretraining_initialization_view(row)

    def _backfill_pretraining_initializations(self):
        """Index older accepted initialization outputs without rehashing them.

        The stored output ref only comes from a completed lease, whose output
        passed the publication validator before it was accepted. This
        backfill checks the reachable artifact's bounded manifest identity;
        byte integrity is checked by materialization on selection.
        """
        with self._connection() as db:
            rows = db.execute("""SELECT jobs.*,worker_leases.output_ref_json,
                       worker_leases.released_at
                    FROM jobs JOIN worker_leases ON worker_leases.job_id=jobs.job_id
                    WHERE worker_leases.outcome='completed'
                      AND worker_leases.output_ref_json IS NOT NULL
                      AND jobs.status IN ('completed','indexed')
                      AND json_valid(jobs.work_unit_json)
                      AND json_extract(jobs.work_unit_json,'$.schema.version')=2
                      AND json_extract(jobs.work_unit_json,'$.phase')='initialize'
                      AND NOT EXISTS (SELECT 1 FROM pretraining_initializations p
                                      WHERE p.source_job_id=jobs.job_id)
                    ORDER BY worker_leases.released_at""").fetchall()
        for row in rows:
            try:
                unit = json.loads(row['work_unit_json'] or '{}')
                if unit.get('schema', {}).get('version') != 2 or unit.get('phase') != 'initialize':
                    continue
                ref = ArtifactRef.from_dict(json.loads(row['output_ref_json']))
                root = self.artifact_store.resolve(ref)
                manifest_path = root / 'artifact.json'
                if not manifest_path.is_file() or manifest_path.stat().st_size > 1024 * 1024:
                    continue
                manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                artifact_id, run_id = str(manifest['artifact_id']), str(manifest['run_id'])
                dataset_ref = unit.get('inputs', {}).get('input') or unit.get('inputs', {}).get('dataset')
                published_at = row['released_at'] or row['completed_at'] or row['updated_at']
                with self._connection() as db:
                    db.execute('BEGIN IMMEDIATE')
                    db.execute("""INSERT INTO pretraining_initializations
                        (artifact_id,run_id,execution_run_id,artifact_ref_json,source_job_id,
                         specification_id,queued_run_id,dataset_ref_json,configuration_ref_json,published_at)
                        VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(artifact_id) DO NOTHING""",
                        (artifact_id, run_id, str(unit['run_id']), json.dumps(ref.to_dict()), row['job_id'],
                         row['specification_id'], row['queued_run_id'],
                         json.dumps(dataset_ref) if dataset_ref else None,
                         json.dumps(unit.get('configuration')), published_at))
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                # Missing historical payloads remain absent rather than
                # becoming selectable as if their bytes were available.
                continue

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

    def _read_segment_result(self, path, unit):
        candidate = Path(path) / 'segment-result.json'
        if not candidate.is_file() or candidate.stat().st_size > 65536:
            raise ValueError('Training segment must provide a bounded result manifest')
        result = json.loads(candidate.read_text())
        if result.get('schema') != 'oracle_builder_training_segment_result/v1' or result.get('phase') != unit.get('phase'):
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
        if unit['phase'] == 'initialize':
            if cursor != 0 or state.get('pretraining_completed') is not True or result.get('next_phase') != 'train':
                raise ValueError('Self-supervised initialization did not produce a valid supervised handoff')
            source_ref = unit.get('inputs', {}).get('pretraining_source')
            initialization = unit.get('parameters', {}).get('initialization', {})
            source_id = initialization.get('source_artifact_id') if isinstance(initialization, dict) else None
            handoff = state.get('pretraining_handoff')
            if (not isinstance(handoff, dict)
                    or handoff.get('boundary') not in {'whole_phase', 'portable_pretraining'}
                    or type(handoff.get('supervised_epoch')) is not int
                    or handoff.get('supervised_epoch') != 0
                    or handoff.get('ssl_continuation') != 'not_portable'):
                raise ValueError('Self-supervised initialization has invalid handoff provenance')
            if handoff['boundary'] == 'portable_pretraining' and (
                    any(not isinstance(handoff.get(key), str) or not handoff[key].strip()
                        for key in ('source_artifact_id', 'source_run_id'))
                    or any(not isinstance(handoff.get(key), str)
                           or not re.fullmatch(r'[0-9a-f]{64}', handoff[key])
                           for key in ('source_fingerprint_sha256', 'source_model_sha256'))):
                raise ValueError('Self-supervised initialization has invalid source provenance')
            if source_ref is not None:
                if (not isinstance(initialization, dict) or initialization.get('mode') != 'pretraining'
                        or not isinstance(source_id, str) or not source_id
                        or handoff['boundary'] != 'portable_pretraining'):
                    raise ValueError('Initialization output does not preserve its selected pretraining source identity')
                self._validate_selected_pretraining_handoff(source_ref, handoff, source_id)
            elif isinstance(initialization, dict) and initialization.get('mode') == 'pretraining':
                raise ValueError('Selected pretraining initialization is missing its sealed source reference')
        elif unit['phase'] == 'train':
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
        expected_phase = 'train' if unit['phase'] == 'initialize' else ('finalize' if cursor >= segment['total_epochs'] or early_stop else 'train')
        if result.get('next_phase') != expected_phase:
            raise ValueError('Worker next phase disagrees with authoritative training progress')
        return result

    def _validate_selected_pretraining_handoff(self, source_ref, handoff, expected_artifact_id):
        """Verify worker-reported lineage against the immutable selected source."""
        from oracle_builder.artifacts import validate_run_artifact

        ref = ArtifactRef.from_dict(source_ref)
        source = self.artifact_store.resolve(ref)
        report = validate_run_artifact(source)
        if not report.get('valid') or report.get('lifecycle') != 'sealed':
            raise ValueError('Selected pretraining source is missing, invalid, or unsealed')
        manifest = json.loads((source / 'artifact.json').read_text(encoding='utf-8'))
        if manifest.get('artifact_id') != expected_artifact_id:
            raise ValueError('Selected pretraining source artifact identity mismatch')
        if ref.fingerprint_sha256:
            from oracle_data_contracts.artifacts import directory_content_digest
            if directory_content_digest(source) != ref.fingerprint_sha256:
                raise ValueError('Selected pretraining source reference fingerprint mismatch')
        state_path = source / 'model/recovery/state.json'
        shared_state_path = source / 'model/recovery/stratified_state.json'
        source_state_path = state_path if state_path.is_file() else shared_state_path
        if not source_state_path.is_file() or source_state_path.stat().st_size > 1024 * 1024:
            raise ValueError('Selected pretraining source has no bounded recovery state')
        source_state = json.loads(source_state_path.read_text(encoding='utf-8'))
        source_handoff = source_state.get('pretraining_handoff')
        if (source_state.get('schema') != {'name': 'oracle_builder_training_recovery', 'version': '2.0.0'}
                or source_state.get('artifact_id') != manifest.get('artifact_id')
                or source_state.get('run_id') != manifest.get('run_id')
                or source_state.get('pretraining_completed') is not True
                or type(source_state.get('completed_epoch')) is not int
                or source_state.get('completed_epoch') != 0 or source_state.get('cursor')
                or not isinstance(source_handoff, dict)
                or source_handoff.get('boundary') not in {'whole_phase', 'portable_pretraining'}
                or type(source_handoff.get('supervised_epoch')) is not int
                or source_handoff.get('supervised_epoch') != 0
                or source_handoff.get('ssl_continuation') != 'not_portable'):
            raise ValueError('Selected pretraining source is not a completed epoch-zero handoff')
        if source_handoff['boundary'] == 'portable_pretraining' and (
                any(not isinstance(source_handoff.get(key), str) or not source_handoff[key].strip()
                    for key in ('source_artifact_id', 'source_run_id'))
                or any(not isinstance(source_handoff.get(key), str)
                       or not re.fullmatch(r'[0-9a-f]{64}', source_handoff[key])
                       for key in ('source_fingerprint_sha256', 'source_model_sha256'))):
            raise ValueError('Selected pretraining source has invalid lineage metadata')
        model_state = source_state if source_state_path == state_path else source_state.get('shared', {})
        model_path = model_state.get('model_path', '')
        model_file = source / model_path
        if (not model_path or not model_file.is_file()
                or not model_file.resolve().is_relative_to(source.resolve())):
            raise ValueError('Selected pretraining source model path is invalid')
        model_digest = hashlib.sha256()
        with model_file.open('rb') as model_bytes:
            for chunk in iter(lambda: model_bytes.read(1024 * 1024), b''):
                model_digest.update(chunk)
        if model_digest.hexdigest() != model_state.get('model_sha256'):
            raise ValueError('Selected pretraining source model checksum mismatch')
        expected = {
            'source_artifact_id': manifest['artifact_id'],
            'source_run_id': manifest['run_id'],
            'source_fingerprint_sha256': report.get('fingerprint_sha256'),
            'source_model_sha256': model_state.get('model_sha256'),
        }
        if any(not expected[key] or handoff.get(key) != expected[key] for key in expected):
            raise ValueError('Initialization output lineage does not match the selected pretraining source')

    def _commit_segment_in_connection(self, db, *, job, lease_id, output_ref, result, now):
        unit = json.loads(job['work_unit_json'])
        run = db.execute('SELECT * FROM execution_runs WHERE run_id=?', (unit['run_id'],)).fetchone()
        if not run or run['current_job_id'] != job['job_id']:
            raise ValueError('Stale training unit cannot advance the run')
        cursor = result['cursor']['completed_epoch']
        pending = db.execute("SELECT * FROM worker_commands WHERE lease_id=? AND status IN ('pending','received','accepted') ORDER BY sequence DESC LIMIT 1", (lease_id,)).fetchone()
        if pending and pending['action'] in {'stop_now', 'restart'}:
            raise ValueError('Stopped attempt cannot advance training')
        if unit.get('phase') == 'initialize':
            # Publication is committed alongside the accepted checkpoint and
            # continuation, so subsequent supervised failure cannot hide it.
            checkpoint_path = self.artifact_store.resolve(output_ref)
            manifest = json.loads((checkpoint_path / 'artifact.json').read_text(encoding='utf-8'))
            artifact_id = str(manifest['artifact_id'])
            dataset_ref = unit.get('inputs', {}).get('input') or unit.get('inputs', {}).get('dataset')
            configuration_ref = unit.get('configuration')
            db.execute(
                """INSERT INTO pretraining_initializations
                   (artifact_id,run_id,execution_run_id,artifact_ref_json,source_job_id,specification_id,queued_run_id,
                    dataset_ref_json,configuration_ref_json,published_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(artifact_id) DO NOTHING""",
                (artifact_id, str(manifest['run_id']), str(unit['run_id']), json.dumps(output_ref.to_dict()), job['job_id'],
                 job['specification_id'], job['queued_run_id'],
                 json.dumps(dataset_ref) if dataset_ref else None,
                 json.dumps(configuration_ref), now),
            )
        phase = ('train' if unit['phase'] == 'initialize'
                 else ('finalize' if cursor >= run['total_epochs'] or result.get('stopped_early') else 'train'))
        paused = bool(pending and pending['action'] == 'pause')
        next_id = str(uuid.uuid4())
        inputs = {key: ArtifactRef.from_dict(ref) for key, ref in unit['inputs'].items()}
        # The committed checkpoint owns the copied weights and provenance.
        # Later segments must not depend on retaining the original source.
        inputs.pop('pretraining_source', None)
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
