import assert from 'node:assert/strict';
import test from 'node:test';
import { phaseHistory, phaseId, visibleMetricKeys, collectPhaseEpochs } from '../src/lib/training-telemetry.ts';

test('keeps SSL and supervised histories isolated', () => {
	const points = [{ phase_id: 'self_supervised', epoch: 1, loss: 2, simclr_loss: 1, encoder_representation_std: .3, custom_ssl_health: .1 }, { phase_id: 'supervised', epoch: 1, loss: .5, accuracy: .8 }];
	assert.equal(phaseHistory(points, 'self_supervised').length, 1);
	assert.deepEqual(visibleMetricKeys(points, 'self_supervised'), ['loss', 'simclr_loss', 'encoder_representation_std', 'custom_ssl_health']);
	assert.deepEqual(visibleMetricKeys(points, 'supervised'), ['loss', 'accuracy']);
	assert.equal(phaseId({ phase: 'SSL · BYOL' }), 'self_supervised');
	assert.equal(phaseId({}, 'self_supervised'), 'self_supervised');
});


test('phase-specific epoch aggregation prevents SSL/supervised epoch collisions', () => {
    const units = [
        { snapshot: { phase_id: 'self_supervised', history: [{epoch: 1, loss: 2}, {epoch: 2, loss: 1}] } },
        { snapshot: { phase_id: 'supervised', history: [{epoch: 1, loss: .5}] }, startEpoch: 0, stopEpoch: 1 },
        { snapshot: { phase_id: 'supervised', history: [{epoch: 1, loss: .3}] }, startEpoch: 1, stopEpoch: 2 }
    ];
    assert.deepEqual(collectPhaseEpochs(units, 'self_supervised').map(row => row.loss), [2, 1]);
    assert.deepEqual(collectPhaseEpochs(units, 'supervised'), [{epoch: 1, loss: .5}, {epoch: 2, loss: .3}]);
});

test('retained tail and legacy initialize snapshots preserve actual SSL epoch numbers', () => {
    const history = Array.from({length: 50}, (_, index) => ({epoch: index + 51, loss: 1}));
    const points = collectPhaseEpochs([{snapshot: {history}, fallbackPhase: 'self_supervised'}], 'self_supervised');
    assert.equal(points.length, 50);
    assert.equal(points.at(-1).epoch, 100);
});
