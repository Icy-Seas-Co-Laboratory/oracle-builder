"""Contract tests for the maintained, user-facing TOML recipes.

The exhaustive reference is intentionally resolved against a real frozen
classification checkpoint here.  Static parsing alone would not catch drift
between its documented component choices and the training planner.
"""

from __future__ import annotations

import sqlite3
import re
from pathlib import Path

import pytest

from oracle_builder.config import load_toml, resolve_config
from oracle_builder.data.sqlite_dataset import create_synthetic_classification
from oracle_builder.datasets.schema import set_dataset_lifecycle


PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"
REFERENCE_CONFIG = CONFIG_ROOT / "reference_v2_exhaustive.toml"
SUPPORTED_TRAINING_TASKS = {"classification", "embedding", "segmentation"}
TABLE_HEADER = re.compile(r"^\[([^]]+)\]")
CANONICAL_SECTION_ORDER = {
    "run": 0,
    "architecture": 1,
    "input": 2,
    "input.geometry": 3,
    "data": 4,
    "data.geodesic_distance": 5,
    "data.streaming": 6,
    "data.materialization": 6,
    "preprocessing": 7,
    "preprocessing.derived_channels": 8,
    "encoder": 9,
    "stem": 10,
    "normalization": 11,
    "pooling": 12,
    "image_embedding.projection": 13,
    "metadata": 14,
    "metadata.encoder": 15,
    "metadata.augmentation": 16,
    "fusion": 17,
    "classifier": 18,
    "model": 19,
    "classification.stratification": 20,
    "classification.stratification.conditioning": 21,
    "classification.stratification.training_routing": 22,
    "classification.stratification.cycle_scheduler": 23,
    "self_supervised": 24,
    "pretraining": 24,  # Read-only legacy compatibility table.
    "self_supervised.embedding_health": 25,
    "self_supervised.augmentation": 26,
    "training": 27,
    "training.class_weights": 28,
    "callbacks": 29,
    "recovery": 30,
    "augmentation": 31,
    "distribution": 32,
    "inference": 33,
    "evaluation": 34,
    "evaluation.uncertainty": 35,
    "evaluation.benchmark": 36,
    "tiling": 37,
    "evidence": 38,
    "output": 39,
    "output.intermediate_artifacts": 40,
}


@pytest.fixture
def frozen_classification_database(tmp_path: Path) -> Path:
    """Create the smallest valid immutable input for a classification recipe."""
    database = tmp_path / "classification.sqlite"
    create_synthetic_classification(database, n=9, shape=(16, 16, 1), classes=3)
    with sqlite3.connect(database) as connection:
        set_dataset_lifecycle(connection, "frozen", actor="config-reference-test")
        connection.commit()
    return database


def test_exhaustive_v2_reference_parses_and_resolves_against_frozen_dataset(
    tmp_path: Path,
    frozen_classification_database: Path,
):
    resolved = resolve_config(
        REFERENCE_CONFIG,
        frozen_classification_database,
        tmp_path / "run",
    )

    assert resolved["architecture"]["version"] == 2
    assert resolved["run"]["task"] == "classification"
    assert resolved["data"]["num_classes"] == 3
    assert resolved["dataset"]["lifecycle"] == "frozen"


@pytest.mark.parametrize(
    "path",
    sorted(CONFIG_ROOT.glob("*.toml")),
    ids=lambda path: path.name,
)
def test_every_root_config_toml_parses(path: Path):
    """Product descriptors are TOML too, even though they are not run recipes."""
    assert isinstance(load_toml(path), dict)


@pytest.mark.parametrize(
    "path",
    sorted(CONFIG_ROOT.rglob("*.toml")),
    ids=lambda path: str(path.relative_to(CONFIG_ROOT)),
)
def test_every_training_recipe_is_an_explicit_supported_v2_recipe(path: Path):
    """Only files with ``[run]`` are trainable recipes; product TOMLs are exempt."""
    document = load_toml(path)
    if "run" not in document:
        return

    assert document["architecture"]["version"] == 2
    assert document["run"]["task"] in SUPPORTED_TRAINING_TASKS


@pytest.mark.parametrize(
    "path",
    sorted(CONFIG_ROOT.rglob("*.toml")),
    ids=lambda path: str(path.relative_to(CONFIG_ROOT)),
)
def test_every_training_recipe_uses_the_canonical_v2_section_order(path: Path):
    """Keep copied recipes readable by enforcing the reference-file order."""
    document = load_toml(path)
    if "run" not in document:
        return

    tables = [
        match.group(1)
        for line in path.read_text(encoding="utf-8").splitlines()
        if (match := TABLE_HEADER.match(line))
    ]
    ranks = [CANONICAL_SECTION_ORDER[table] for table in tables]
    assert ranks == sorted(ranks), f"{path} is not in canonical V2 section order"
