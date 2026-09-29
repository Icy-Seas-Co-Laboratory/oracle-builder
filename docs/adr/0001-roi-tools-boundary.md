# ADR 0001: establish a portable ROI-tools boundary

## Status

Accepted — initial extraction.

## Context and decision

ROI refinement currently lives under `oracle_builder.masking`. Array
morphology, thresholding, mask validation, and local image codecs do not need
training, the orchestrator, a dataset database, Pelagia, or Napari.
`oracle_tools.roi` is now their canonical, dependency-light home; it depends
only on NumPy and Pillow.

SQLite annotation workspaces and migrations, dataset materialization, U-Net
configuration, Pelagia/API adapters, and Napari UI remain in Builder because
they are coupled to Builder contracts or optional application dependencies.

## Consequences

New portable ROI code imports `oracle_tools.roi`. Existing Builder modules and
the `oracle-mask-builder` CLI are unchanged in this first slice. A later PR
will switch Builder imports to this package and retain former module paths as
compatibility re-exports. `oracle_tools` must never import Builder; dataset
exchange belongs at an explicit `oracle_data_contracts` boundary.
