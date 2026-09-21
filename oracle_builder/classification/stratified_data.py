"""Stratum-aware classification datasets.

This module is deliberately a thin layer over :mod:`sqlite_stream`.  It keeps
the database's native image shape as the routing source, then lets the normal
classification decoder resize the selected rows for one child model.  As a
result auxiliary features still describe the original ROI and all existing
streaming behaviour (augmentation, reader threads, and named model inputs)
is retained.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import tensorflow as tf

from oracle_builder.classification.stratification import (
    batch_plan,
    dimensions,
    assignment_policy,
    stratum_for_shape,
    training_stratum,
)
from oracle_builder.data.sqlite_stream import (
    SQLiteClassificationSource,
    SQLiteSplitIndex,
    build_classification_index,
)


RoutingMode = Literal["canonical", "training_stochastic"]


def add_stratum_dimension_input(
    dataset: tf.data.Dataset, dimension: int, config: dict[str, Any]
) -> tf.data.Dataset:
    """Decorate a classifier dataset with the constant runtime stratum input.

    Kept as a no-op unless conditioned shared stratification is enabled, so the
    normal one-input and image-plus-metadata model contracts remain unchanged.
    """
    settings = config.get("classification", {}).get("stratification", {})
    conditioning = settings.get("conditioning", {}) if isinstance(settings, dict) else {}
    if not (settings.get("enabled", False) and conditioning.get("enabled", False)):
        return dataset

    def decorate(features, target):
        if isinstance(features, dict):
            values = dict(features)
            image = values["image"]
        else:
            values = {"image": features}
            image = features
        values["stratum_dimension"] = tf.fill(
            [tf.shape(image)[0], 1], tf.cast(dimension, tf.int32)
        )
        return values, target

    return dataset.map(decorate, num_parallel_calls=tf.data.AUTOTUNE)


@dataclass
class StratumDataset:
    """One split routed and resized for a single resolution child model."""

    dimension: int
    split: str
    epoch: int | None
    routing_mode: RoutingMode
    config: dict[str, Any]
    index: SQLiteSplitIndex
    source: SQLiteClassificationSource
    dataset: tf.data.Dataset

    @property
    def records(self) -> list[dict[str, Any]]:
        return list(self.index.iter_records())

    @property
    def count(self) -> int:
        return len(self.index)


def child_config(config: dict[str, Any], dimension: int) -> dict[str, Any]:
    """Copy ``config`` for one child without mutating the parent run config."""
    if dimension not in dimensions(config):
        raise ValueError(f"Resolution {dimension} is not a configured stratum")
    child = deepcopy(config)
    input_shape = list(child["data"]["input_shape"])
    if len(input_shape) != 3:
        raise ValueError("Classification data.input_shape must be [height, width, channels]")
    input_shape[:2] = [int(dimension), int(dimension)]
    child["data"]["input_shape"] = input_shape
    child["data"]["batch_size"] = batch_plan(config)[int(dimension)]
    stratification = child.setdefault("classification", {}).setdefault("stratification", {})
    stratification["_active_dimension"] = int(dimension)
    return child


def routed_index(
    sqlite_path: str | Path,
    config: dict[str, Any],
    split: str,
    dimension: int,
    *,
    routing_mode: RoutingMode = "canonical",
    epoch: int | None = None,
    labeled_only: bool = True,
) -> SQLiteSplitIndex:
    """Return the rows assigned to ``dimension`` for one split.

    Stochastic routing is intentionally valid only for the train split and
    requires an epoch.  Evaluation and serving callers cannot accidentally
    inherit training's alternate-resolution presentation policy.
    """
    if dimension not in dimensions(config):
        raise ValueError(f"Resolution {dimension} is not a configured stratum")
    if routing_mode == "training_stochastic" and split != "train":
        raise ValueError("Stochastic stratum routing is only valid for the train split")
    if routing_mode == "training_stochastic" and epoch is None:
        raise ValueError("Stochastic stratum routing requires an epoch")
    if routing_mode not in {"canonical", "training_stochastic"}:
        raise ValueError(f"Unsupported stratum routing mode {routing_mode!r}")

    full = build_classification_index(sqlite_path, config, split, labeled_only=labeled_only)
    configured = dimensions(config)
    refs = []
    for ref in full.refs:
        record = ref.record()
        shape = record["original_shape"]
        if shape is None:
            raise ValueError(
                f"Stratification requires original source dimensions; item {ref.uuid!r} has none"
            )
        canonical = stratum_for_shape(
            shape, configured, policy=assignment_policy(config)
        )
        assigned = (
            training_stratum(canonical, item_id=ref.uuid, epoch=int(epoch), config=config)
            if routing_mode == "training_stochastic"
            else canonical
        )
        if assigned == dimension:
            refs.append(ref)
    return SQLiteSplitIndex(full.sqlite_path, split, refs)


def build_stratum_dataset(
    sqlite_path: str | Path,
    config: dict[str, Any],
    split: str,
    dimension: int,
    *,
    routing_mode: RoutingMode = "canonical",
    epoch: int | None = None,
    labeled_only: bool = True,
    shuffle: bool | None = None,
    augment: bool | None = None,
) -> StratumDataset:
    """Build a dataset for one independently trained resolution child.

    The returned dataset has the usual ``(inputs, class_index)`` structure;
    when auxiliary features are enabled, ``inputs`` is the existing
    ``{"image", "metadata"}`` dictionary.
    """
    child = child_config(config, dimension)
    index = routed_index(
        sqlite_path, config, split, dimension, routing_mode=routing_mode,
        epoch=epoch, labeled_only=labeled_only,
    )
    if labeled_only and any(ref.target is None for ref in index.refs):
        raise ValueError("Supervised classification dataset contains unlabeled rows")
    source = SQLiteClassificationSource(sqlite_path, child)
    dataset = source.training_dataset(
        index,
        shuffle=(split == "train") if shuffle is None else bool(shuffle),
        augment=(split == "train") if augment is None else bool(augment),
    )
    return StratumDataset(
        dimension=int(dimension), split=split, epoch=epoch,
        routing_mode=routing_mode, config=child, index=index, source=source,
        dataset=dataset,
    )


def build_stratum_datasets(
    sqlite_path: str | Path,
    config: dict[str, Any],
    split: str,
    *,
    routing_mode: RoutingMode = "canonical",
    epoch: int | None = None,
    labeled_only: bool = True,
    include_empty: bool = True,
) -> dict[int, StratumDataset]:
    """Build all configured child datasets for a split."""
    result = {
        dimension: build_stratum_dataset(
            sqlite_path, config, split, dimension, routing_mode=routing_mode,
            epoch=epoch, labeled_only=labeled_only,
        )
        for dimension in dimensions(config)
    }
    return result if include_empty else {
        dimension: value for dimension, value in result.items() if value.count
    }
