# Interactive inference V2: response examples

This page is a shareable reference for the structures returned by Oracle
Builder's interactive inference API. It shows **readable, abbreviated views** of
responses. The actual prediction response is an NPZ archive with media type
`application/vnd.oracle-builder.inference.v2+npz`, not JSON. Its `manifest`
member contains the envelope shown below; numeric arrays live in other NPZ
members. The [V2 client decoder](../oracle_data_contracts/inference_v2.py)
verifies their hashes and returns NumPy arrays in place of the manifest's
array descriptors. Example IDs, scores, and pixels are illustrative; `…`
means entries have been omitted for display.

## The general envelope

One HTTP prediction or WebSocket response frame contains one result set. Each
input item gets a result, including an item that failed during inference. This
is the shape of the decoded response, with the successful item's `output`
shortened here and expanded below:

```json
{
  "schema_name": "oracle_builder.inference.v2.result_set",
  "schema_version": "2.0.0",
  "result_set_id": "11111111-1111-4111-8111-111111111111",
  "model": {
    "artifact_id": "example-roi-classifier",
    "artifact_fingerprint": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "run_id": "example-training-run",
    "task": "classification",
    "architecture": "example-backbone",
    "contract_version": "1.0.0"
  },
  "source_dataset": null,
  "parameters": {},
  "execution": {},
  "started_at": "2026-09-30T12:00:00+00:00",
  "completed_at": "2026-09-30T12:00:00.025000+00:00",
  "counts": {"requested": 2, "succeeded": 1, "rejected": 0, "failed": 1},
  "results": [
    {
      "schema_name": "oracle_builder.inference_result",
      "schema_version": "1.0.0",
      "result_id": "22222222-2222-4222-8222-222222222222",
      "result_set_id": "11111111-1111-4111-8111-111111111111",
      "request_id": "pelagia-batch-42",
      "item_id": "33333333-3333-4333-8333-333333333333",
      "sequence_number": 0,
      "source": {"system": "pelagia", "resource_type": "roi", "resource_id": "roi-17"},
      "input_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
      "model": {"artifact_id": "example-roi-classifier", "artifact_fingerprint": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "run_id": "example-training-run", "task": "classification", "architecture": "example-backbone", "contract_version": "1.0.0"},
      "status": "ok",
      "output": "See the classification output example below",
      "execution": {"received_at": "2026-09-30T12:00:00+00:00", "completed_at": "2026-09-30T12:00:00.010000+00:00", "duration_ms": 10.0},
      "warnings": [],
      "error": null
    },
    {
      "schema_name": "oracle_builder.inference_result",
      "schema_version": "1.0.0",
      "result_id": "44444444-4444-4444-8444-444444444444",
      "result_set_id": "11111111-1111-4111-8111-111111111111",
      "request_id": "pelagia-batch-42",
      "item_id": "55555555-5555-4555-8555-555555555555",
      "sequence_number": 1,
      "source": null,
      "input_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
      "model": {"artifact_id": "example-roi-classifier", "artifact_fingerprint": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "run_id": "example-training-run", "task": "classification", "architecture": "example-backbone", "contract_version": "1.0.0"},
      "status": "failed",
      "output": null,
      "execution": {"received_at": "2026-09-30T12:00:00+00:00", "completed_at": "2026-09-30T12:00:00.025000+00:00", "duration_ms": 25.0},
      "warnings": [],
      "error": {"type": "ValueError", "message": "Example item could not be processed"}
    }
  ]
}
```

The string in the first `output` slot is a **documentation placeholder**; a
real successful result always contains an output object. Use `request_id` and
`item_id` to correlate results with inputs. `result_set_id` groups a response,
and each result has its own `result_id`. WebSocket frames can finish out of
order. A retried request may run again and create new result IDs: the supplied
IDs are correlation keys, not a server-side deduplication guarantee.

## Image analysis output

Classification, embedding, and clustering models all use the `classification`
**request task** and return the same output schema. `model_task` names the
actual artifact task. The eight scientific fields below are always present on
every successful image result. Unrequested fields are `null`; requesting a
field absent from the catalog's `possible_outputs` fails validation instead.
`diagnostics` is a ninth, separate selectable field.

Here is an example **classification** `output`, shown as it might look after
decoding and rendering the vector as a short list:

```json
{
  "schema_name": "oracle_builder.inference.v2.image_output",
  "schema_version": "1.0.0",
  "type": "image_analysis",
  "model_task": "classification",
  "class_probabilities": {
    "unit": "probability",
    "source_artifact_id": "example-roi-classifier",
    "classes": [
      {"class_index": 0, "label_id": "clear", "label_name": "Clear", "concept_id": null, "concept_node_id": null, "probability": 0.08},
      {"class_index": 1, "label_id": "target", "label_name": "Target", "concept_id": null, "concept_node_id": null, "probability": 0.92}
    ]
  },
  "primary_decision": {"class_index": 1, "label_id": "target", "label_name": "Target", "concept_id": null, "concept_node_id": null, "concept_relationship": null, "abstained": false},
  "embedding": {
    "values": [0.12, -0.34, 0.56, "…"],
    "dtype": "float32",
    "dimension": 128,
    "normalized": true,
    "normalization": "l2",
    "source_artifact_id": "example-roi-classifier"
  },
  "embedding_normalized": true,
  "knn": {"metric": "cosine_similarity", "source_artifact_id": "example-roi-classifier", "source_role": "classification_evidence", "k_requested": 5, "k_used": 5, "nearest_neighbor_similarity": 0.89, "nearest_neighbor_uuid": "reference-roi-7"},
  "prototype_similarity": {"metric": "cosine_similarity", "source_artifact_id": "example-roi-classifier", "source_role": "classification_evidence", "similarities": {"0": 0.21, "1": 0.84}, "predicted_class": 1, "nearest_similarity": 0.84, "similarity_margin": 0.63},
  "secondary_heads": null,
  "cluster_evidence": null,
  "diagnostics": {"class_logits": [-1.1, 1.4], "class_logits_unit": "logit", "logits_source": "model", "source_artifact_id": "example-roi-classifier", "class_evidence_schema_version": 1, "cluster_evidence_schema_version": null}
}
```

An embedding-only model has the same keys: `class_probabilities`,
`primary_decision`, `knn`, `prototype_similarity`, `secondary_heads`, and
`cluster_evidence` may all be `null`; `embedding` and
`embedding_normalized` are populated. A clustering model likewise leaves
class predictions `null`. It puts neighbors in `knn` with
`source_role: "clustering_evidence"` and preserves the full clustering packet
in `cluster_evidence` (including cluster decision, candidate clusters, and
novelty calibration). Cluster centroids are not reported as class prototypes.
The reserved `secondary_heads` field is currently unsupported by shipped
artifacts; a future artifact supporting it can return an empty list when no
secondary result is produced.

The original class logits live in `diagnostics.class_logits`. The legacy
classification `evidence.knn` and `evidence.prototype` packets map to `knn`
and `prototype_similarity`; clustering `evidence.nearest_neighbors` maps to
`knn`, while the complete packet maps to `cluster_evidence`. This preserves
scientific evidence while keeping a predictable top-level shape.

## Mask refinement output

Segmentation artifacts use the `mask_refinement` request task. This example
selects all five possible outputs from a candidate-delta model. The displayed
pixel blocks are abbreviated; the decoder returns NumPy arrays.

```json
{
  "type": "mask_refinement",
  "target_mode": "candidate_delta",
  "threshold": {"value": 0.5, "source": "artifact_validation_optimization"},
  "logits_source": "model",
  "transform": {"original_shape": [512, 512], "tile_shape": [256, 256], "tile_count": 9, "tile_overlap": 0.5, "reassembly": "hann"},
  "mask": [[0, 1, "…"], [0, 1, "…"]],
  "probability_map": [[0.02, 0.91, "…"], [0.04, 0.88, "…"]],
  "logits": [[-2.7, 2.3, "…"], [-2.4, 2.0, "…"]],
  "delta_mask": [[0, 1, "…"], [0, 0, "…"]],
  "delta_probability_map": [[0.01, 0.89, "…"], [0.03, 0.12, "…"]]
}
```

`mask` and `probability_map` are the **reconstructed final result** for
candidate-delta models. `delta_mask` and `delta_probability_map` expose the
raw model-target output. The default selection is only `mask`; dense maps,
logits, and delta arrays require explicit output selection. When the model
requires a `candidate_mask`, the catalog lists it under `required_inputs`.

## Catalog and array transport

Ask `GET /v2/inference/models/{artifact_id}` before constructing a request.
Its `capabilities` describe `model_task`, `request_task`, `required_inputs`,
`input_shape`, `default_outputs`, and `possible_outputs`. Image-family entries
also publish `output_schema_name` and `output_schema_version`. Pin the
catalog's `fingerprint_sha256` in `X-Artifact-Fingerprint` on warm and predict.

On the wire, a populated array is represented in the NPZ manifest by a small
descriptor like this; its `transport_key` names another member of the same
archive. The actual SHA-256 is over the array's NPY encoding.

```json
{
  "asset_id": "66666666-6666-4666-8666-666666666666",
  "media_type": "application/x-npy",
  "shape": [128],
  "dtype": "float32",
  "sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
  "transport_key": "output_0_item_33333333-3333-4333-8333-333333333333_embedding_values"
}
```

The decoder verifies array identity, shape, type, hash, and response limits
before replacing this descriptor with a NumPy array. The result-set transport
version (`2.0.0`), per-result contract version (`1.0.0`), and image-output
version (`1.0.0`) are separate. A client can decode either an HTTP response
body or one binary WebSocket response frame with the same function:

```python
from oracle_data_contracts.inference_v2 import decode_v2_inference_result_set

result_set = decode_v2_inference_result_set(response_bytes)
for result in result_set["results"]:
    print(result["request_id"], result["item_id"], result["status"])
```

For endpoint details, projection rules, and retry behavior, see
[Inference APIs](inference-api.md).
