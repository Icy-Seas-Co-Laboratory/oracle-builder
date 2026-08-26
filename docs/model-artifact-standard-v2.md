# Oracle Model Artifact Standard V2

V2 separates the deployable model from the training record while retaining the
V1 directory reader and compatibility fields.

## Profiles

`artifact.json.standard.profile` is one of:

- `deployment_asset`: a lean, sealed inference package;
- `training_record`: a reproducible training and retraining record.

Both profiles use the shared `oracle_model_artifact` standard version `2.0.0`.

## Deployment assets

A deployment asset contains the model contract at `model/contract.json`, the
preferred runtime model, optional alternate formats, load-test results, runtime
provenance, JSONL lifecycle events, and checksums. It does not require training
splits, training history, optimizer checkpoints, or prediction databases.

Publish one from a sealed training record with:

```bash
oracle-run publish-deployment \
  runs/example \
  deployments/example
```

Use `--include-weights` for an optional rebuild representation and
`--no-evidence` to omit packaged classification or clustering evidence.

## Training records

Training records preserve configuration, dataset lineage, split protocol,
recovery state, training outputs, and the model assets produced by the run.
They write canonical JSONL streams to `logs/events.jsonl` and
`metrics/metrics.jsonl`. Existing SQLite and CSV outputs remain compatibility
views for V1 consumers.

## Model contract

`model/contract.json` is the authoritative model interface. Consumers should
use it to discover inputs, outputs, preprocessing, postprocessing, and label
spaces rather than reconstructing the contract from configuration files or
directory names.

## Immutability and lineage

Sealed artifacts are immutable. Continuing training or publishing a deployment
asset creates a new artifact with lineage pointing to its parent. The source
training record is not modified by deployment publishing.

Create a retraining package with optional embedded data and source files:

```bash
oracle-run materialize-training \
  runs/example \
  training/example-portable \
  --dataset datasets/example.sqlite \
  --source-root .
```

The copied record retains its training identity, receives a new package
fingerprint, and records embedded resources in `library/manifest.json`.
