# Oracle Builder inference contract V1

Oracle Builder's deployable model is an inference bundle: deterministic
preprocessing, a neural-network core, deterministic postprocessing, and a
versioned input/output contract. Training augmentation is part of the training
recipe and is not executed by the inference bundle.

## Ownership and persistence

The system providing an operational input remains its authoritative owner.
Oracle Builder accepts an `InferenceItem`, performs inference, and returns an
`InferenceResult` in memory. It does not write the input or result unless the
caller explicitly selects a sink. Importing an operational example into a
curated Oracle Builder dataset is a separate, explicit promotion operation.

`InferenceBundle.predict`, `predict_batch`, `predict_stream`, and
`predict_stream_async` are storage-neutral. `run_connector` uses an in-memory
sink by default. `JSONLinesSink` and dataset prediction databases are explicit
persistence choices.

## Identities and hashes

- Dataset, revision, run, model artifact, input item, array asset, result, and
  result-set identities use UUIDs.
- A SHA-256 digest identifies the encoded bytes of every `ArrayPayload`.
- `InferenceItem.input_sha256` covers the ordered role and digest of every input.
- Saved model formats carry SHA-256 hashes in `model/model_manifest.json`.
- A sealed run artifact continues to hash its complete inventory.

UUIDs identify logical resources. Hashes identify exact content. They are not
interchangeable.

## Outputs

Every successful classification result includes:

- exact pre-softmax logits from new Oracle Builder models;
- probabilities and the selected class;
- the fixed-size identity embedding when supported;
- prototype and KNN evidence when an evidence index is present.

Every successful segmentation result includes:

- pre-activation logits;
- the probability map;
- the thresholded mask;
- threshold and spatial transformation information;
- reconstructed outputs for candidate-delta models.

`logits_source` is `model` when values came from the explicit named logits
layer. For probability-only external artifacts it records the reversible
canonical derivation used by the compatibility adapter, such as
`derived_log_probability` or `derived_inverse_sigmoid`.

Large arrays remain NumPy arrays in memory. The JSON representation uses a
typed NPY payload with shape, dtype, asset UUID, SHA-256, and optional base64
data. Connectors may replace JSON transport with multipart, shared-memory, or
object-reference transports without changing the logical contract.

## Minimal use

```python
import numpy as np

from oracle_builder.inference import InferenceBundle, InferenceItem

bundle = InferenceBundle.load("runs/example")
item = InferenceItem.from_array(np.zeros((512, 512), dtype="uint8"))

result = bundle.predict(item)       # no file writes
logits = result.output["logits"]
```

For a stream:

```python
from oracle_builder.inference import run_connector

result_set = run_connector(bundle, pelagia_items)  # retained in memory
```

Persistence must be requested:

```python
from oracle_builder.inference import JSONLinesSink, run_connector

result_set = run_connector(
    bundle,
    pelagia_items,
    sink=JSONLinesSink("predictions.jsonl"),
)
```

## Inference batching

Training and inference batch sizes are independent. The default inference
policy is `auto`: Oracle Builder estimates a bounded batch from the tensor shape,
model family, and configured memory budget, verifies it with a forward pass,
and halves it after a TensorFlow resource-exhaustion error. The resolved size
and attempted sizes are reported and written to the training event log.

```toml
[inference]
batch_size = "auto"       # or an explicit positive integer
minimum_batch_size = 1
maximum_batch_size = 64
memory_budget_mb = 512
progress = true
```

The estimate is intentionally conservative and does not claim to describe
total GPU memory. The verified forward pass and bounded fallback are the
authoritative safeguards.
