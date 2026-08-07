# oracle-data-contracts

`oracle-data-contracts` is the dependency-light shared implementation of the
Oracle Dataset V1 contract. It provides SQLite schema initialization and
migration, validation, fingerprints, lifecycle operations, deterministic folder
transfer, annotation-workspace labels/reviews/evidence, snapshots, training
releases, model-run artifact manifests, and split protocols. It does not depend
on TensorFlow, Keras, napari, or application API clients.

Install it independently from this source tree:

```bash
python3 -m pip install ./oracle_data_contracts
```

The current package supports Dataset Schema `1.2.0`.
