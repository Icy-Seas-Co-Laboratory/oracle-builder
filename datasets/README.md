# Datasets

oracle-builder expects a SQLite database with a `samples` table:

```sql
CREATE TABLE samples (
    uuid TEXT PRIMARY KEY,
    split TEXT,
    input_blob BLOB,
    input_blob_encoding TEXT,
    input_blob_dimensions TEXT,
    output_blob BLOB,
    output_blob_encoding TEXT,
    output_blob_dimensions TEXT,
    label_text TEXT,
    sample_weight REAL,
    metadata_json TEXT
);
```

Use `python3 -m oracle_builder.data.sqlite_dataset --classification datasets/example_classification.sqlite`
or `python3 -m oracle_builder.data.sqlite_dataset --segmentation datasets/example_segmentation.sqlite`
to create tiny synthetic examples.

