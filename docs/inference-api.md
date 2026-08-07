# Oracle Builder inference API

The inference API keeps validated model bundles resident and exposes the same
versioned contract to Pelagia mask refinement and future classification work.
Callers retain ownership of inputs and outputs; the service does not persist
operational data.

## Install and serve

```bash
python -m pip install -e '.[api]'

oracle-serve \
  --models-root ./models \
  --host 127.0.0.1 \
  --port 8100
```

`--models-root` recursively discovers sealed Oracle Builder model runs and
model products below the supplied directory. It uses the artifact name as a
normalized HTTP-safe selector, adding an artifact-ID suffix only when names
collide. Working or malformed artifacts are skipped and reported on stderr.

Use `--model ALIAS=RUN_DIR` to register a specific artifact or choose an
explicit operational selector; repeat it to add more models. Aliases are
operational selectors; every result reports the resolved immutable artifact ID,
run ID, fingerprint, and contract version. Models are validated and preloaded
before readiness succeeds. Use `--no-preload` only for development.

The catalog retains manifest identity and task metadata when runtime validation
fails. Clients can therefore show registered-but-unavailable models and their
load errors without offering those models for inference.

Set `ORACLE_BUILDER_API_TOKEN` to require a bearer token. Pelagia supplies the
same value through `PELAGIA_ORACLE_API_TOKEN`. ASGI deployments may use
`ORACLE_BUILDER_MODELS_ROOT`, with multiple roots separated by the platform
path separator, instead of enumerating `ORACLE_BUILDER_MODELS`.

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
