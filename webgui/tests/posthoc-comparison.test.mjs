import assert from 'node:assert/strict';
import test from 'node:test';
import { exactPosthocComparison, lowerIsBetter } from '../src/lib/models/posthoc-comparison.ts';
const member = (overrides = {}) => ({ posthoc: { evaluation_signature: 'test-population', training_dataset: { fingerprint_sha256: 'frozen-dataset' }, class_names: { 0: 'a', 1: 'b' }, ...overrides } });
test('comparison requires identical complete evidence for every member', () => {
  assert.equal(exactPosthocComparison([member(), member()]), true);
  assert.equal(exactPosthocComparison([member(), {}]), false);
  assert.equal(exactPosthocComparison([member()]), false);
  assert.equal(exactPosthocComparison([member(), member({ evaluation_signature: 'other-test' })]), false);
  assert.equal(exactPosthocComparison([member(), member({ training_dataset: {} })]), false);
  assert.equal(exactPosthocComparison([member(), member({ class_names: {} })]), false);
  assert.equal(exactPosthocComparison([member(), member({ class_names: { 0: 'b', 1: 'a' } })]), false);
});
test('probability losses and calibration error use lower-is-better deltas', () => {
  for (const key of ['log_loss', 'multiclass_brier_score', 'expected_calibration_error']) assert.equal(lowerIsBetter(key), true);
  for (const key of ['accuracy', 'macro_f1', 'macro_roc_auc']) assert.equal(lowerIsBetter(key), false);
});
