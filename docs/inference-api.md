# Oracle Builder inference API

The inference API keeps validated model bundles resident and exposes the same
versioned contract to Pelagia mask refinement and future classification work.
Callers retain ownership of inputs and outputs; the service does not persist
operational data.

## Install and serve

```bash
uv sync --extra api

uv run oracle-serve \
  --models-root ./models \
  --host 127.0.0.1 \
  --port 8100
```

`--models-root` recursively discovers sealed Oracle Builder model runs and
model products below the supplied directory. It uses the artifact name as a
normalized HTTP-safe selector, adding an artifact-ID suffix only when names
collide. Working or malformed artifacts are skipped and reported on stderr.

When a reverse proxy exposes the service beneath a path prefix, supply that
external prefix so OpenAPI documentation and generated URLs remain correct:

```bash
uv run oracle-serve \
  --models-root ./models \
  --root-path /oracle-builder-api \
  --host 127.0.0.1 \
  --port 8100
```

`ORACLE_BUILDER_ROOT_PATH` provides the same setting for ASGI and managed
service deployments.

Use `--model ALIAS=RUN_DIR` to register a specific artifact or choose an
explicit operational selector; repeat it to add more models. Aliases are
operational selectors; every result reports the resolved immutable artifact ID,
run ID, fingerprint, and contract version. Models are validated and preloaded
before readiness succeeds. Use `--no-preload` only for development.

The catalog retains manifest identity and task metadata when runtime validation
fails. Clients can therefore show registered-but-unavailable models and their
load errors without offering those models for inference.

## Serving performance

When a model is preloaded, Oracle Builder creates one resident classification
or embedding callable. It prefers the bundle's SavedModel concrete signature
and otherwise creates one compiled Keras named-output function. The callable is warmed with
batch sizes 1 and 64 (or the configured maximum) before the model is ready.

Each model has one bounded execution queue. Concurrent requests are combined
for a single GPU call until the item limit is reached or the short queue
deadline expires. Input decoding, response construction, and transport work
remain outside that GPU-facing worker. Every caller still receives a separate,
correlated result set.

The service startup options are the single source of truth for these limits:

```bash
uv run oracle-serve --models-root ./models \
  --max-batch-size 256 --max-wait-ms 8 --queue-capacity 1024
```

The request limit always matches the resolved execution limit. If startup
warmup cannot fit the requested batch in memory, Oracle Builder reduces that
model's active limit and rejects larger requests before execution.

Per-model configuration takes precedence over service defaults. `GET
/v1/models` includes runtime warm-up and micro-batch diagnostics after loading.

The model catalog and every prediction result set expose an `execution` object
with `gpu_accelerated`, `accelerator`, `device_type`, `device_name`, and
`device_count`. This tells Pelagia whether TensorFlow selected a GPU or CPU for
the resident inference callable. The prediction response also repeats the
object at `parameters.execution` for callers that consume service parameters.

Set `ORACLE_BUILDER_API_TOKEN` to require a bearer token. Pelagia supplies the
same value through `PELAGIA_ORACLE_API_TOKEN`. ASGI deployments may use
`ORACLE_BUILDER_MODELS_ROOT`, with multiple roots separated by the platform
path separator, instead of enumerating `ORACLE_BUILDER_MODELS`.
Environment deployments can set the service defaults with
`ORACLE_BUILDER_SERVING_MAX_BATCH_SIZE`,
`ORACLE_BUILDER_SERVING_MAX_WAIT_MS`, and
`ORACLE_BUILDER_SERVING_QUEUE_CAPACITY`.

## Endpoints

- `GET /health/live` reports process liveness.
- `GET /health/ready` reports whether registered models are usable.
- `GET /v1/models?task=segmentation` lists model aliases and immutable identity.
- `POST /v1/models/{selector}:predict` executes a versioned inference request.

Inference uses `application/vnd.oracle-builder.inference+npz`. The archive has
a UTF-8 JSON `manifest` stored as a `uint8` array plus named NPY-compatible
arrays referenced by `transport_key`. Loading always uses `allow_pickle=False`.
Requests and responses are size- and item-bounded.

The logical objects remain `InferenceItem`, `InferenceResult`, and
`InferenceResultSet`; NPZ is only their efficient HTTP transport. Mask models
receive whole ROI images and candidate masks. The bundle owns preprocessing,
tiling, model execution, reassembly, and thresholding.
