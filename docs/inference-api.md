# Retired resident inference API; supported batch inference

`oracle-serve` and its resident-model HTTP inference API are retired. Oracle
Builder no longer exposes a network inference process. Do not build new
integrations against the former `/v1/models` routes.

Supported batch inference is an Orchestrator-owned `infer` WorkUnit:

- `POST /v1/inference-runs` creates a request from a sealed model artifact,
  frozen dataset, selected split, and admitted worker pool.
- `POST /v1/inference-runs/{id}:start` authorizes the durable request.
- `GET /v1/inference-runs` and `GET /v1/inference-runs/{id}` expose its state.
- `POST /v1/inference-runs/{id}:cancel` cancels queued/running work.
- `GET /v1/inference-runs/{id}:download` retrieves the sealed result archive.

Use the Web GUI or `oracle inference` for common workflows. Workers remain
pull-only internal clients and never expose an inference HTTP service.
