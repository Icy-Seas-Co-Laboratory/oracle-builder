"""Display and execution metadata for the self-supervised training phase."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping


def phase_budgets(config: Mapping[str, Any]) -> dict[str, Any]:
    task = config.get('run', {}).get('task', 'classification')
    ssl = config.get('self_supervised', config.get('pretraining', {})) or {}
    enabled = bool(ssl.get('enabled', False))
    ssl_epochs = int(ssl.get('epochs', 10)) if enabled else None
    supervised = 0 if task == 'embedding' else int(config.get('training', {}).get('epochs', 10))
    return {'task': task, 'ssl_enabled': enabled, 'ssl_epochs': ssl_epochs,
            'supervised_epochs': supervised, 'total_planned_epochs': supervised + (ssl_epochs or 0),
            'self_supervised': {'enabled': enabled, 'method': ssl.get('method'), 'epochs': ssl_epochs}}


def seal_phase_budgets(config: dict[str, Any], epochs: int, ssl_epochs: int | None) -> dict[str, Any]:
    for name, value in [('epochs', epochs), ('ssl_epochs', ssl_epochs)]:
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f'{name} must be a positive integer')
    ssl = config.setdefault('self_supervised', {})
    if config.get('run', {}).get('task') == 'embedding' and not ssl.get('enabled', False):
        raise ValueError('Embedding runs require self_supervised.enabled')
    if ssl_epochs is not None and not ssl.get('enabled', False):
        raise ValueError('ssl_epochs requires self_supervised.enabled')
    config.setdefault('training', {})['epochs'] = epochs
    if ssl.get('enabled', False):
        ssl['epochs'] = ssl_epochs if ssl_epochs is not None else epochs
    return phase_budgets(config)


def ssl_artifact_summary(root: Path, config: Mapping[str, Any], provenance: Any = None) -> dict[str, Any]:
    settings = dict(config.get('self_supervised', config.get('pretraining', {})) or {})
    history_path = root / 'metrics' / 'pretraining' / 'metrics.csv'
    completed, latest, columns = 0, {}, []
    try:
        with history_path.open(newline='', encoding='utf-8') as handle:
            reader = csv.DictReader(handle)
            columns = [name for name in reader.fieldnames or [] if name and name != 'epoch']
            for row in reader:
                if not any(value not in (None, '') for value in row.values()):
                    continue
                completed += 1
                latest = {}
                for name in columns:
                    try:
                        value = float(row.get(name) or '')
                    except (ValueError, TypeError):
                        continue
                    if math.isfinite(value):
                        latest[name] = value
    except OSError:
        pass
    weights = root / 'model' / 'pretraining' / 'student_pretrained.weights.h5'
    reconstruction = root / 'model' / 'pretraining' / 'grayscale_reconstruction.weights.h5'
    enabled = bool(settings.get('enabled', False))
    source = provenance if isinstance(provenance, Mapping) and provenance.get('kind') == 'source_neural_training' else {}
    phase_provenance = {'phase': 'self_supervised', 'metrics_path': 'metrics/pretraining/metrics.csv',
                        **({'source_model_artifact_id': source.get('source_model_artifact_id'),
                            'source_dataset_id': source.get('source_dataset_id')} if source else {})}
    health_path = history_path.parent / 'embedding_health.json'
    try:
        health = json.loads(health_path.read_text())
    except (OSError, ValueError):
        health = None
    files = [{'path': str(path.relative_to(root)), 'kind': kind} for path, kind in (
        (history_path, 'epoch_metrics'),
        (history_path.with_suffix('.json'), 'metric_series'),
        (history_path.with_suffix('.jsonl'), 'training_log'),
        (weights, 'pretrained_weights'), (reconstruction, 'pretrained_weights'),
        (health_path, 'embedding_health'),
    ) if path.is_file()]
    return {'enabled': enabled, 'method': settings.get('method'),
            'planned_epochs': settings.get('epochs') if enabled else None,
            'completed_epochs': completed, 'metric_keys': columns,
            'history_available': completed > 0, 'latest_metrics': latest,
            'settings': settings, 'provenance': phase_provenance, 'files': files, 'health': health,
            'outputs': {'metrics': history_path.is_file(), 'weights': weights.is_file() or reconstruction.is_file()},
            'output_paths': {'metrics': 'metrics/pretraining/metrics.csv' if history_path.is_file() else None,
                             'weights': str(weights.relative_to(root)) if weights.is_file() else str(reconstruction.relative_to(root)) if reconstruction.is_file() else None}}
