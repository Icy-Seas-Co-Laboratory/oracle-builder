"""Durable, attempt-scoped worker directives independent of execution."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import hmac
import threading
import time
import uuid


COMMAND_ACTIONS = frozenset({'pause', 'yield', 'stop_now', 'restart', 'resume'})
TERMINAL_COMMANDS = frozenset({'applied', 'failed', 'expired', 'superseded'})
_CHANGED = threading.Condition()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _public(row):
    value = dict(row)
    value['result'] = json.loads(value.pop('result_json'))
    return value


class WorkerControlMixin:
    def job_commands(self, job_id):
        with self._connection() as db:
            return [_public(row) for row in db.execute('SELECT * FROM worker_commands WHERE job_id=? ORDER BY sequence', (job_id,))]

    def worker_heartbeat_status(self, worker_id):
        with self._connection() as db:
            row = db.execute('SELECT * FROM worker_heartbeats WHERE worker_id=?', (worker_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result['telemetry'] = json.loads(result.pop('telemetry_json'))
        return result

    def request_job_command(self, job_id, *, action, reason='', command_id=None):
        if action not in COMMAND_ACTIONS:
            raise ValueError('Unsupported worker command')
        if not isinstance(reason, str) or len(reason) > 2000:
            raise ValueError('Command reason must be at most 2000 characters')
        command_id = str(uuid.UUID(command_id)) if command_id else str(uuid.uuid4())
        now = _now()
        with self._connection() as db:
            db.execute('BEGIN IMMEDIATE')
            existing = db.execute('SELECT * FROM worker_commands WHERE command_id=?', (command_id,)).fetchone()
            if existing:
                if existing['job_id'] != job_id or existing['action'] != action or existing['reason'] != reason:
                    raise ValueError('Command idempotency key already describes another request')
                return _public(existing)
            job = db.execute('SELECT * FROM jobs WHERE job_id=?', (job_id,)).fetchone()
            if job is None:
                raise KeyError(job_id)
            if not job['worker_pool_id']:
                raise ValueError('Only pull-worker jobs support directives')
            if job['status'] in {'completed', 'indexed', 'failed', 'cancelled', 'historical'}:
                raise ValueError('A terminal job cannot accept a directive')
            lease = db.execute("SELECT * FROM worker_leases WHERE job_id=? AND status='active' AND expires_at>?", (job_id, now)).fetchone()
            generation = 0
            if lease:
                generation = db.execute('SELECT ordinal FROM execution_attempts WHERE lease_id=?', (lease['lease_id'],)).fetchone()[0]
                worker = db.execute('SELECT capabilities_json FROM registered_workers WHERE worker_id=?', (lease['worker_id'],)).fetchone()
                if json.loads(worker[0]).get('worker_control_v1') is not True:
                    raise ValueError('Restart this worker with directive support before sending commands')
            unit = json.loads(job['work_unit_json'])
            segment = unit.get('parameters', {}).get('execution_segment', {})
            if action in {'pause', 'yield'} and segment.get('phase') != 'train':
                raise ValueError('This work unit has no supported resumable boundary')
            pending = db.execute("SELECT * FROM worker_commands WHERE job_id=? AND status NOT IN ('applied','failed','expired','superseded')", (job_id,)).fetchall()
            if pending and action != 'stop_now':
                raise ValueError('A directive is already pending for this job')
            if action == 'stop_now':
                db.execute("UPDATE worker_commands SET status='superseded',updated_at=? WHERE job_id=? AND status NOT IN ('applied','failed','expired','superseded')", (now, job_id))
            status = 'pending'
            if action == 'resume':
                if job['status'] != 'paused' or lease:
                    raise ValueError('Only a paused job can be resumed')
                db.execute("UPDATE jobs SET status='queued',remote_status='queued',updated_at=? WHERE job_id=?", (now, job_id))
                self._control_queue_status(db, job, 'queued', now)
                status = 'applied'
            elif not lease:
                if job['status'] not in {'queued', 'paused'}:
                    raise ValueError('Job has no current active attempt; reconcile its lease first')
                target = {'pause': 'paused', 'yield': 'queued', 'stop_now': 'cancelled', 'restart': 'queued'}[action]
                db.execute('UPDATE jobs SET status=?,remote_status=?,updated_at=? WHERE job_id=?', (target, target, now, job_id))
                self._control_queue_status(db, job, target, now)
                status = 'applied'
            elif action == 'stop_now':
                db.execute("UPDATE jobs SET status='cancel_requested',remote_status='cancel_requested',cancel_requested_at=?,updated_at=? WHERE job_id=?", (now, now, job_id))
            sequence = db.execute('SELECT COALESCE(MAX(sequence),0)+1 FROM worker_commands WHERE job_id=?', (job_id,)).fetchone()[0]
            db.execute('INSERT INTO worker_commands(command_id,job_id,lease_id,generation,sequence,action,status,reason,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?)',
                       (command_id, job_id, lease['lease_id'] if lease else None, generation, sequence, action, status, reason, now, now))
            self._record_job_event_in_connection(db, job_id, 'command_requested', f'Worker directive: {action}', {'command_id': command_id, 'action': action, 'status': status})
            result = _public(db.execute('SELECT * FROM worker_commands WHERE command_id=?', (command_id,)).fetchone())
        with _CHANGED:
            _CHANGED.notify_all()
        return result

    @staticmethod
    def _control_queue_status(db, job, status, now):
        if job['queued_run_id']:
            db.execute('UPDATE queued_runs SET status=?,updated_at=? WHERE queued_run_id=?', (status, now, job['queued_run_id']))
        db.execute('UPDATE execution_runs SET status=?,updated_at=? WHERE current_job_id=?', (status, now, job['job_id']))

    def heartbeat_worker_lease(self, *, lease_id, worker_id, worker_token, lease_token, ttl_seconds=30, telemetry=None):
        telemetry = telemetry or {}
        if not isinstance(telemetry, dict):
            raise ValueError('Heartbeat telemetry must be an object')
        encoded = json.dumps(telemetry, sort_keys=True, allow_nan=False)
        if len(encoded.encode()) > 8192:
            raise ValueError('Heartbeat telemetry exceeds 8 KiB')
        now, expiry = _now(), self._lease_expiry(ttl_seconds)
        with self._connection() as db:
            db.execute('BEGIN IMMEDIATE')
            lease, worker, job = self._authenticate_active_worker_lease(db, lease_id=lease_id, worker_id=worker_id, worker_token=worker_token, lease_token=lease_token)
            generation = db.execute('SELECT ordinal FROM execution_attempts WHERE lease_id=?', (lease_id,)).fetchone()[0]
            if telemetry.get('generation', generation) != generation:
                raise ValueError('Heartbeat belongs to a stale attempt generation')
            boot = json.loads(worker['capabilities_json']).get('execution_instance_id')
            if telemetry.get('worker_boot_id', boot) != boot:
                raise ValueError('Heartbeat belongs to a different worker boot')
            db.execute('UPDATE worker_leases SET expires_at=?,updated_at=? WHERE lease_id=?', (expiry, now, lease_id))
            db.execute('UPDATE registered_workers SET last_seen_at=?,updated_at=? WHERE worker_id=?', (now, now, worker_id))
            db.execute('INSERT INTO worker_heartbeats VALUES(?,?,?,?,?) ON CONFLICT(worker_id) DO UPDATE SET lease_id=excluded.lease_id,generation=excluded.generation,telemetry_json=excluded.telemetry_json,received_at=excluded.received_at', (worker_id, lease_id, generation, encoded, now))
            return {'expires_at': expiry, 'ttl_seconds': ttl_seconds, 'generation': generation, 'cancel_requested': job['cancel_requested_at'] is not None}

    def poll_worker_commands(self, *, lease_id, worker_id, worker_token, lease_token, wait_seconds=0):
        if isinstance(wait_seconds, bool) or not isinstance(wait_seconds, (int, float)) or not 0 <= wait_seconds <= 30:
            raise ValueError('Command wait must be between 0 and 30 seconds')
        deadline = time.monotonic() + wait_seconds
        while True:
            with self._connection() as db:
                lease, worker, job = self._authenticate_active_worker_lease(db, lease_id=lease_id, worker_id=worker_id, worker_token=worker_token, lease_token=lease_token)
                rows = db.execute("SELECT * FROM worker_commands WHERE lease_id=? AND status IN ('pending','received','accepted') ORDER BY sequence", (lease_id,)).fetchall()
                result = {'commands': [_public(row) for row in rows], 'cancel_requested': job['cancel_requested_at'] is not None}
            if rows or result['cancel_requested'] or time.monotonic() >= deadline:
                return result
            # Notifications wake this process immediately; periodic rechecks
            # also observe writes from another process or a restarted server.
            with _CHANGED:
                _CHANGED.wait(timeout=min(1, max(0, deadline-time.monotonic())))

    def acknowledge_worker_command(self, *, lease_id, worker_id, worker_token, lease_token, command_id, status, result=None):
        if status not in {'received', 'accepted', 'applied', 'failed'}:
            raise ValueError('Invalid command acknowledgment')
        encoded = json.dumps(result or {}, sort_keys=True, allow_nan=False)
        if len(encoded.encode()) > 8192:
            raise ValueError('Command acknowledgment exceeds 8 KiB')
        now = _now()
        with self._connection() as db:
            db.execute('BEGIN IMMEDIATE')
            command = db.execute('SELECT * FROM worker_commands WHERE command_id=? AND lease_id=?', (command_id, lease_id)).fetchone()
            if command is None:
                raise KeyError(command_id)
            if command['status'] in TERMINAL_COMMANDS:
                self._authenticate_registered_worker(db, worker_id, worker_token)
                lease = db.execute('SELECT * FROM worker_leases WHERE lease_id=?', (lease_id,)).fetchone()
                if not lease or lease['worker_id'] != worker_id or not hmac.compare_digest(lease['token_sha256'], self._token_digest(lease_token)):
                    raise PermissionError('Command receipt does not belong to this worker lease')
                return _public(command)
            lease, worker, job = self._authenticate_active_worker_lease(db, lease_id=lease_id, worker_id=worker_id, worker_token=worker_token, lease_token=lease_token)
            ranks = {'pending': 0, 'received': 1, 'accepted': 2, 'applied': 3, 'failed': 3}
            if ranks[status] <= ranks[command['status']]:
                return _public(command)
            if status == 'applied' and command['action'] in {'pause', 'yield'}:
                raise ValueError('Pause/yield applies only when a durable checkpoint is committed')
            if status == 'applied' and command['action'] in {'restart', 'stop_now'}:
                target = 'queued' if command['action'] == 'restart' else 'cancelled'
                db.execute("UPDATE worker_leases SET status='released',outcome=?,released_at=?,updated_at=? WHERE lease_id=?", (command['action'], now, now, lease_id))
                self._finish_execution_attempt_in_connection(db, lease_id, status='cancelled', classification='directive', error=command['action'], now=now)
                db.execute('UPDATE jobs SET status=?,remote_status=?,worker_id=NULL,updated_at=? WHERE job_id=?', (target, target, now, job['job_id']))
                self._control_queue_status(db, job, target, now)
                db.execute("UPDATE registered_workers SET state='idle',last_seen_at=?,updated_at=? WHERE worker_id=?", (now, now, worker_id))
            db.execute('UPDATE worker_commands SET status=?,result_json=?,updated_at=? WHERE command_id=?', (status, encoded, now, command_id))
            self._record_job_event_in_connection(db, job['job_id'], 'command_'+status, f"Directive {command['action']}: {status}", {'command_id': command_id})
            return _public(db.execute('SELECT * FROM worker_commands WHERE command_id=?', (command_id,)).fetchone())
