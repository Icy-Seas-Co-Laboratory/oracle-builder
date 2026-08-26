# Classification configuration defaults

These examples are dataset-agnostic: the training command supplies the SQLite
dataset, and Oracle Builder infers the class count from its label vocabulary.
Copy the closest example and change only the settings your experiment needs.

| Configuration | Best starting use |
| --- | --- |
| `simple_cnn.toml` | Fast pipeline/dataset validation and low-compute baseline |
| `resnet_like.toml` | Small residual model when canonical ResNet is too large |
| `densenet_like.toml` | Small dense-connectivity experiment |
| `resnet.toml` | General-purpose canonical ResNet-18/34/50/101/152 |
| `densenet.toml` | Canonical DenseNet-121/169/201 with strong feature reuse |
| `efficientnet.toml` | EfficientNet B0-B7 and custom compound scaling |

All examples intentionally use the same preprocessing and augmentation policy so
family comparisons are controlled. They assume grayscale input and disable
per-epoch checkpoints while still saving the final portable model. DenseNet may
need a smaller batch size than the shared default because dense feature
concatenation uses more activation memory.

Weighted sparse categorical cross entropy is the default. Class weights are
calculated from the run-owned training split using:

```toml
[training.class_weights]
mode = "effective_number"
beta = 0.999
normalize = true
```

Choose `inverse_frequency` for stronger balancing. Use `explicit` only when
domain knowledge supports hand-selected weights. To opt out of weighting, use
`loss = "sparse_categorical_crossentropy"`.

To enable self-supervised training, set `self_supervised.enabled = true`. Choose
`byol` for the student-teacher approach without negative examples, or `simclr`
for NT-Xent contrastive training. Both transfer the learned encoder into the
selected classification family before supervised training.
