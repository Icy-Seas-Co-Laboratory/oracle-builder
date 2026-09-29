# ADR 0003: Work-unit and ROI-tool boundaries

## Status

Accepted for the first migration slices.

## Work units

`WorkUnit v1` is the dependency-light, versioned execution contract shared by
the orchestrator and workers. It identifies inputs, configuration, resources,
and a job-scoped staging target with artifact references. It has a canonical
JSON representation and digest. The orchestrator persists the exact unit at
dispatch time; retries must reuse it rather than reconstructing mutable input.

## ROI tooling

The standalone distribution is named `oracle-tools` and its Python package is
`oracle_tools`. ROI functionality belongs under `oracle_tools.roi`. It may
depend on `oracle_data_contracts` and optional UI/image libraries, but must not
depend on `oracle_builder.orchestration`, worker code, training code, or the
Web GUI. Its output boundary is a frozen Oracle dataset contract.

The existing `oracle_builder.masking` imports and `mask_builder.py` command
remain compatibility adapters while modules move in small, tested steps.
