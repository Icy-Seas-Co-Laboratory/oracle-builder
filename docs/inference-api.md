# Oracle Builder inference API

The inference API keeps validated model bundles resident and exposes the same
versioned contract to Pelagia mask refinement and future classification work.
Callers retain ownership of inputs and outputs; the service does not persist
operational data.

## Install and serve

```bash
python -m pip install -e '.[api]'

oracle-serve \
  --model pelagia-refiner=/absolute/path/to/sealed/run \
  --host 127.0.0.1 \
  --port 8100
```

Register more than one `--model ALIAS=RUN_DIR` when needed. Aliases are
operational selectors; every result reports the resolved immutable artifact ID,
run ID, fingerprint, and contract version. Models are validated and preloaded
before readiness succeeds. Use `--no-preload` only for development.

Set `ORACLE_BUILDER_API_TOKEN` to require a bearer token. Pelagia supplies the
same value through `PELAGIA_ORACLE_API_TOKEN`.

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
