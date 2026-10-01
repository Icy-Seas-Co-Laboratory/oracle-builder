# Inference APIs

Oracle Builder supports durable V1 batch inference and an interactive V2 API.
The former `/v1/models` resident HTTP API remains retired.
For a shareable, illustrated reference to result envelopes and task-specific
outputs, see [Inference envelope V2](inference-envelope-v2.md).

## Interactive V2

The public model selector is a sealed catalog `artifact_id`, never a filesystem
path. `GET /v2/inference/models` lists eligible sealed, complete models;
`GET /v2/inference/models/{artifact_id}` returns the model and its
`fingerprint_sha256` and a verified capabilities descriptor: model and request
tasks, required input roles, input shape, and default/possible outputs. This
does not load TensorFlow, though first access may copy and verify the artifact.
In role-token deployments, detail requires at least an analyst token because
that first verification may be expensive.
Clients should pin the fingerprint in `X-Artifact-Fingerprint` on later calls.
A changed fingerprint returns `409`.

`POST /v2/inference/models/{artifact_id}:warm` verifies and loads the model
before interactive traffic. The first prediction also loads it if needed.
The warm response repeats that descriptor. Classification, embedding, and
clustering artifacts use the `classification` request task because they share
the image-input family; `model_task` identifies the actual artifact task.
Segmentation artifacts use `mask_refinement`.
`POST /v2/inference/models/{artifact_id}:predict` accepts and returns
`application/vnd.oracle-builder.inference.v2+npz`. The public request frame is
limited to 32 MiB; the resident service also limits item count and decompressed
arrays. The resident cache is bounded and evicts idle least-recently-used models.

`WS /v2/inference/models/{artifact_id}/stream` accepts the same NPZ request as
each binary WebSocket message and returns the same NPZ result as a binary
message. Up to eight requests may be in flight; responses may arrive out of
order. Correlate them by `request_id` and `item_id`. Per-frame errors arrive as
JSON text messages with `event`, `frame_sequence` (zero-based send order),
`status`, and `detail`. Configured deployments
require at least an analyst role token for interactive HTTP and WebSocket calls.

Python clients can use `V2InferenceRequest`, `encode_v2_inference_request`, and
`decode_v2_inference_result_set` from the dependency-light
`oracle_data_contracts.inference_v2` module. The Builder runtime uses the same
wire format in `oracle_builder.inference.transport_v2`.
Each request selects `classification` or `mask_refinement` and contains one or
more `InferenceItem` values with matching `request_id`. Optional
`V2ResponseOptions` select outputs and a per-item output byte budget.
The budget is capped at 32 MiB per item and 256 MiB across result arrays.
Every successful image result uses the independently versioned
`oracle_builder.inference.v2.image_output` schema (`1.0.0`). Its eight
scientific keys are always present: `class_probabilities`, `primary_decision`,
`embedding`, `embedding_normalized`, `knn`, `prototype_similarity`,
`secondary_heads`, and `cluster_evidence`. `diagnostics` is a separate stable
field for class logits, logits provenance, and evidence schema versions.
Absent or unrequested fields are `null`. The reserved `secondary_heads` field
is currently unsupported by shipped artifacts and therefore null; an artifact
that later supports it will return `[]` when it has no head results. A selected field unsupported
by the artifact returns `422` before
inference. Compare the catalog's `possible_outputs` with the request's
selection to distinguish unsupported from unrequested nulls. Omitting
`outputs` populates the artifact's `default_outputs`.

Populated class probabilities carry stable class indices and label identities,
probability units, and the model artifact ID. Embeddings carry a typed vector,
dimension, dtype, declared normalization, and artifact ID. KNN and class
prototype values carry their cosine-similarity metric and evidence source
artifact ID. Clustering neighbors also populate `knn`; the complete clustering
packet remains in `cluster_evidence`. Cluster centroids are not relabeled as
class prototypes. `secondary_heads` is reserved for typed, versioned
auxiliary-head results. The image-output version is independent of the NPZ
transport version.

Mask refinement defaults to the final refined `mask`. Dense `probability_map` and `logits` are
opt-in. For candidate-delta models, `mask` and `probability_map` are the
reconstructed result; raw model-target outputs are available as `delta_mask`
and `delta_probability_map`. A mask-refinement request may include `candidate_mask` where required
by the model. The runtime preserves the model's tile and reassembly behavior.

## Timeouts and retries

A timed-out HTTP or WebSocket request has an unknown outcome: inference may
already have completed. `request_id` and `item_id` are correlation keys, not a
server-side deduplication promise. Retrying may execute inference again and
create new result IDs. Retry the same payload and model fingerprint with the
same IDs after a timeout or transient `503`, using bounded backoff. Persist
results idempotently by `(model fingerprint, request_id, item_id)` and accept
the first successful result. Correct `413`/`422` requests instead of repeating
them unchanged; a `409` fingerprint mismatch requires revisiting catalog
detail and choosing the intended model version.

The orchestrator proxies to a loopback-only resident process. The local stack
launcher starts both services and supplies an internal bearer token. For a
separate deployment, start `oracle-inference-runtime` with database, workspace,
and artifact roots and `ORACLE_INFERENCE_RUNTIME_TOKEN`; start the orchestrator
with `ORACLE_INFERENCE_RUNTIME_URL` and the same token. Run one runtime process
per accelerator so the model cache is actually shared.
Training workers and the resident runtime can contend when assigned the same
GPU. On CUDA hosts, set `ORACLE_INFERENCE_CUDA_VISIBLE_DEVICES` in the local
launcher to give inference a distinct GPU, or `-1` to hide CUDA devices from
the runtime. TensorFlow Metal selection on macOS needs separate process or
machine placement for reliable isolation.

The in-memory loaded-model cache is bounded. Verified materialized model
copies remain in the runtime's `inference-runtime-artifacts` directory after
eviction so a later warm can avoid copying them again. This directory can grow
as new model versions are warmed; operators should monitor it and remove stale
copies only while the runtime process is stopped.

## Durable V1 batch inference

Batch inference remains an Orchestrator-owned `infer` WorkUnit:

- `POST /v1/inference-runs` creates a request from a sealed model artifact,
  frozen dataset, selected split, and admitted worker pool.
- `POST /v1/inference-runs/{id}:start` authorizes the durable request.
- `GET /v1/inference-runs` and `GET /v1/inference-runs/{id}` expose its state.
- `POST /v1/inference-runs/{id}:cancel` cancels queued/running work.
- `GET /v1/inference-runs/{id}:download` retrieves the sealed result archive.

Use the Web GUI or `oracle inference` for common workflows. Workers remain
pull-only internal clients and never expose an inference HTTP service.
