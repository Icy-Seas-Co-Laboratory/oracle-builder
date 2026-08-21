# oracle-data-contracts

Package release `0.2.x` owns Oracle Dataset schema `1.5.x`. Consumers should
pin to `oracle-data-contracts>=0.2,<0.3`; this release is intentionally breaking
for Registry's former private schema extensions.

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

The current package supports Dataset Schema `1.5.0`. Version 1.4 promotes
classification reviews and multi-valued target/image descriptors from the
Registry application into the shared dataset contract. Version 1.5 adds
canonical, source-coordinate object bounding boxes and stored crop bounds for
every dataset item while retaining application-specific metadata such as the
full Pelagia provenance document.
