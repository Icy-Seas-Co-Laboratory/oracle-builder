# Unlabeled ROI clustering workflow

Use `oracle-cluster` to train a self-supervised encoder, calculate a spherical
k-means organization of the ROI embeddings, and package both into a sealed
model artifact.

```bash
oracle-cluster \
  --config configs/example_clustering.toml \
  --input datasets/rois.sqlite \
  --output runs/roi-clusters
```

The input must be a frozen classification Dataset V1 database. Existing
classification annotations are ignored; every classification item is used for
self-supervised training and cluster fitting. BYOL is the default method and
SimCLR is also supported. Self-supervised view augmentation is configured only
under `[self_supervised.augmentation]`; it never inherits supervised training
augmentation settings.

The artifact contains:

- the encoder and SavedModel `embed` signature;
- normalized per-ROI embeddings and stable item UUIDs;
- cluster assignments, centroids, sizes, similarity floors, and representative
  (medoid) UUIDs;
- a structure summary including the cluster count and optional silhouette
score.

For large datasets, `fit_batch_size` switches fitting to deterministic
mini-batch spherical k-means once the ROI count exceeds that batch size,
`silhouette_max_samples` bounds the quadratic diagnostic cost, and
`reference_neighbors_per_cluster` retains only the most representative
references per cluster. Cluster sizes, medoids, and novelty thresholds are
still calculated from every fitted ROI. Use `reference_neighbors_per_cluster =
"all"` only when a full nearest-neighbor reference set is essential.

Serve it like any other sealed model artifact:

```bash
oracle-serve --model roi-clusters=runs/roi-clusters
```

An inference result has `output.type = "clustering"` and includes an
`output.evidence` packet with the selected cluster, top candidate clusters,
representative UUIDs, nearest neighbors, similarity thresholds, and a novelty
/ abstention decision. Cluster IDs are run-local (`cluster-0000`, etc.) and
should not be treated as taxonomy labels.

The artifact can also be discovered beneath a models root:

```bash
oracle-serve --models-root runs
```

## Attach clustering evidence to an existing classifier

When a compatible classifier or prior clustering encoder already exists, avoid
retraining it. Fit the cluster structure against its named embedding layer and
explicitly reopen/reseal the artifact:

```bash
oracle-cluster --mode fit \
  --config configs/example_clustering.toml \
  --input datasets/rois.sqlite \
  --encoder-run runs/classifier \
  --reopen-reseal
```

This is intentionally in-place and refuses to overwrite an existing index.
`--mode fit` is optional when `--encoder-run` is supplied.
The resulting classifier continues to return its normal classification output;
`oracle-serve` adds a separate `output.clustering_evidence` packet.
