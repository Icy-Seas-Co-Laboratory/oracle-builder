"""Parent-owned lease watchdog and commands; no TensorFlow dependency."""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time


class ExecutionInterrupted(RuntimeError):
    def __init__(self, action, command=None):
        super().__init__(f'Execution interrupted: {action}')
        self.action, self.command = action, command


class WorkerControl:
    def __init__(self, client, *, credentials, control_file, ttl_seconds, generation=1, worker_boot_id=None, interval=5):
        self.client, self.credentials = client, credentials
        self.control_file = Path(control_file)
        self.ttl_seconds, self.interval = ttl_seconds, min(interval, ttl_seconds / 4)
        self.telemetry = {'generation': generation, 'phase': 'materializing'}
        if worker_boot_id is not None:
            self.telemetry['worker_boot_id'] = worker_boot_id
        self.deadline = time.monotonic() + ttl_seconds - min(5, ttl_seconds / 4)
        self.halt = threading.Event()
        self.done = threading.Event()
        self.command = None
        self._sequence = 0
        self.action = None
        self._lock = threading.Lock()
        self._threads = []

    def start(self):
        for target in (self._heartbeat, self._commands, self._watchdog):
            thread = threading.Thread(target=target, daemon=True, name='oracle-worker-control')
            self._threads.append(thread)
            thread.start()

    def progress(self, event, data=None):
        with self._lock:
            self.telemetry.update({'phase': event, 'last_progress_at': time.time()})
            if data:
                self.telemetry['progress'] = {key: value for key, value in data.items() if key in {'epoch','batch','global_step','completed_epoch','phase'}}

    def _interrupt(self, action, command=None):
        with self._lock:
            if self.done.is_set() or self.action == 'stop_now' or self.action == 'lease_lost':
                return
            self.action, self.command = action, command
            sequence = (command or {}).get('sequence', self._sequence + 1)
            self._sequence = max(self._sequence, sequence)
            value = {'command': 'stop' if action in {'stop_now', 'lease_lost'} else action,
                     'sequence': sequence, 'command_id': (command or {}).get('command_id'), 'reason': (command or {}).get('reason')}
            temporary = self.control_file.with_suffix('.tmp')
            temporary.write_text(json.dumps(value))
            os.replace(temporary, self.control_file)
            if action in {'stop_now', 'restart', 'lease_lost'}:
                self.halt.set()

    def _heartbeat(self):
        while not self.done.is_set():
            started = time.monotonic()
            try:
                with self._lock:
                    telemetry = dict(self.telemetry)
                reply = self.client.heartbeat(**self.credentials, ttl_seconds=self.ttl_seconds, telemetry=telemetry)
                # Start time, not response time, provides a conservative bound
                # even when the successful renewal response was delayed.
                self.deadline = started + self.ttl_seconds - min(5, self.ttl_seconds / 4)
                if reply.get('cancel_requested'):
                    # Poll retrieves the durable command when available; this
                    # also handles cancellation through the legacy API.
                    # Do not reuse a prior pause/yield receipt here. A
                    # cancellation can supersede that directive before the
                    # command poll wakes; reporting the old command as
                    # applied would falsely claim a checkpoint exists.
                    self._interrupt('stop_now')
            except Exception as exc:
                if getattr(exc, 'status_code', None) in {401, 403, 404, 409, 422}:
                    self._interrupt('lease_lost')
                    return
            self.done.wait(self.interval)

    def _watchdog(self):
        while not self.done.wait(0.1):
            if time.monotonic() >= self.deadline:
                self._interrupt('lease_lost')
                return

    def _commands(self):
        while not self.done.is_set() and not self.halt.is_set():
            try:
                reply = self.client.commands(**self.credentials, wait_seconds=min(25, self.ttl_seconds / 2))
                commands = reply.get('commands', [])
                for command in commands:
                    if self.done.is_set():
                        return
                    if command['status'] == 'pending':
                        self.client.acknowledge_command(**self.credentials, command_id=command['command_id'], status='received')
                    self.client.acknowledge_command(**self.credentials, command_id=command['command_id'], status='accepted')
                    self._interrupt(command['action'], command)
                if reply.get('cancel_requested') and not commands:
                    self._interrupt('stop_now')
                if commands:
                    self.done.wait(0.5)
            except Exception as exc:
                if getattr(exc, 'status_code', None) in {401,403,404,409,422}:
                    self._interrupt('lease_lost')
                    return
                self.done.wait(min(1, self.interval))

    def check(self):
        if self.halt.is_set():
            raise ExecutionInterrupted(self.action, self.command)

    def stop(self):
        self.done.set()
        # Long-poll is bounded but must never delay termination/publication.
        for thread in self._threads:
            thread.join(timeout=0.1)
