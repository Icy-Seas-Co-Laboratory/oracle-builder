# Datasets

Oracle Builder dataset schema V1 uses one SQLite file per dataset. A dataset is
typed as either `classification` or `mask_refinement`; both share stable
identity, metadata, assets, provenance, lifecycle, and validation
contracts.

Create tiny working examples:

```bash
python3 -m oracle_builder.data.sqlite_dataset \
  --classification datasets/example_classification.sqlite

python3 -m oracle_builder.data.sqlite_dataset \
  --segmentation datasets/example_segmentation.sqlite
```

Create a frozen training checkpoint:

```bash
oracle-dataset checkpoint datasets/example_classification.sqlite
```

Inspect, validate, export, restore, or explicitly reopen a dataset:

```bash
oracle-dataset info DATASET.sqlite
oracle-dataset validate DATASET.sqlite
oracle-dataset export DATASET.sqlite EXPORT_DIRECTORY
oracle-dataset import EXPORT_DIRECTORY RESTORED.sqlite
oracle-dataset thaw DATASET.sqlite --reason "resume curation"
```

See [the V1 schema contract](../docs/dataset-schema-v1.md).
