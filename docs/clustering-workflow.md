# Downstream ROI clustering

Oracle Builder publishes representation models, not cluster products. Train a
self-supervised embedding model with:

```bash
oracle-embed \
  --config configs/example_embedding.toml \
  --input datasets/rois.sqlite \
  --output runs/roi-embedding
```

The sealed training record contains the SSL diagnostics and the model assets.
Publish a lean deployment asset with `oracle-run publish-deployment`; its
contract exposes only the embedding and preprocessing needed by consumers.
The model does not choose a clustering algorithm, number of clusters, labels,
or novelty thresholds.

A downstream application can then generate embeddings and own the analysis:

1. Load the sealed embedding deployment asset.
2. Generate embeddings for an explicitly identified dataset or stream.
3. Fit and validate the chosen clustering method.
4. Define cluster names, representatives, thresholds, and scientific meaning.
5. Store assignments as a separate cluster record.

That cluster record should retain the source model artifact ID and fingerprint,
embedding dimension/dtype/normalization, preprocessing contract, source dataset
and revision fingerprint, fitting method and parameters, implementation version,
and its own checksum. Cluster IDs are local to that record and are not model
labels.

## Legacy clustering packages

Existing sealed `task = "clustering"` runs remain readable by older inference
consumers. Extract their dataset-conditioned evidence into a standalone
downstream record without changing the source:

```bash
oracle-run migrate-clustering \
  runs/legacy-clusters \
  records/legacy-clusters
```

The migrated record contains `cluster_manifest.json` and an `evidence/`
directory. It is not a model product and must be interpreted together with the
referenced legacy model artifact. New runs should use `oracle-embed` and leave
clustering to downstream consumers.
