"""Download CIFAR-10/CIFAR-100 and store them as Oracle Builder datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from oracle_builder.data.decoders import encode_npy
from oracle_builder.data.sqlite_dataset import ensure_schema
from oracle_builder.datasets.repository import SQLiteDatasetRepository
from oracle_builder.datasets.schema import utc_now


CIFAR10_LABELS = (
    "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck",
)

CIFAR100_LABELS = (
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
)


def _load_cifar(dataset_name: str) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[str, ...]]:
    """Load the canonical arrays through TensorFlow/Keras' verified downloader."""
    try:
        from tensorflow import keras
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("CIFAR import requires TensorFlow. Install oracle-builder training dependencies.") from exc

    if dataset_name == "cifar10":
        train, test = keras.datasets.cifar10.load_data()
        labels = CIFAR10_LABELS
    elif dataset_name == "cifar100":
        train, test = keras.datasets.cifar100.load_data(label_mode="fine")
        labels = CIFAR100_LABELS
    else:  # pragma: no cover - argparse constrains this
        raise ValueError(f"Unsupported CIFAR dataset: {dataset_name}")
    return train, test, labels


def import_cifar(options: argparse.Namespace) -> dict[str, Any]:
    output = Path(options.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output dataset already exists: {output}; choose a new output path.")

    (train_images, train_labels), (test_images, test_labels), labels = _load_cifar(options.dataset)
    train_labels = np.asarray(train_labels).reshape(-1)
    test_labels = np.asarray(test_labels).reshape(-1)
    if train_images.shape[0] != train_labels.shape[0] or test_images.shape[0] != test_labels.shape[0]:
        raise ValueError("CIFAR image and label counts do not agree")

    output.parent.mkdir(parents=True, exist_ok=True)
    dataset_id = options.dataset_id or str(uuid.uuid4())
    connection = sqlite3.connect(":memory:" if options.dry_run else output)
    info = ensure_schema(
        connection,
        "classification",
        dataset_id=dataset_id,
        name=options.dataset,
        title=options.dataset.upper().replace("CIFAR", "CIFAR-"),
        description=(
            f"Canonical {options.dataset.upper()} image classification benchmark downloaded via TensorFlow/Keras."
        ),
        version="canonical",
        metadata={
            "source": {
                "name": options.dataset.upper(),
                "provider": "TensorFlow/Keras datasets",
                "official_partitions": {"train": int(train_images.shape[0]), "test": int(test_images.shape[0])},
                "license_note": "Refer to the original CIFAR dataset terms and citation.",
            },
            "imaging": {"polarity": "unknown", "channels": "rgb", "native_shape": [32, 32, 3]},
            "split_policy": "Official CIFAR train/test membership is stored as source_partition metadata; model split manifests remain run-owned.",
        },
    )
    import_id = str(uuid.uuid4())
    report = {
        "import_id": import_id,
        "dataset_id": info["dataset_id"],
        "dataset": options.dataset,
        "output": str(output),
        "dry_run": bool(options.dry_run),
        "labels": list(labels),
        "counts": {"train": int(train_images.shape[0]), "test": int(test_images.shape[0])},
    }
    if not options.dry_run:
        try:
            repository = SQLiteDatasetRepository(connection)
            label_ids = {
                name: repository.add_classification_label(index, name, metadata={"source": options.dataset})
                for index, name in enumerate(labels)
            }
            for partition, images, target_labels in (
                ("train", train_images, train_labels),
                ("test", test_images, test_labels),
            ):
                for index, (image, class_index) in enumerate(zip(images, target_labels, strict=True)):
                    normalized_index = int(class_index)
                    if not 0 <= normalized_index < len(labels):
                        raise ValueError(f"Invalid {options.dataset} class index: {normalized_index}")
                    array = np.asarray(image, dtype=np.uint8)
                    payload = encode_npy(array)
                    source_key = f"{options.dataset}/{partition}/{index:05d}"
                    content_sha256 = hashlib.sha256(payload).hexdigest()
                    asset = repository.add_asset(
                        payload,
                        encoding="npy",
                        media_type="application/x-npy",
                        shape=array.shape,
                        dtype=str(array.dtype),
                        original_filename=f"{index:05d}.npy",
                        metadata={"source_dataset": options.dataset, "source_partition": partition},
                        content_sha256=content_sha256,
                    )
                    item_id = repository.add_item(
                        source_key=source_key,
                        metadata={
                            "source_dataset": options.dataset,
                            "source_partition": partition,
                            "source_index": index,
                            "class_name": labels[normalized_index],
                            "class_index": normalized_index,
                            "import_id": import_id,
                        },
                    )
                    repository.add_classification_item(
                        item_id=item_id,
                        image_asset_id=asset.asset_id,
                        label_id=label_ids[labels[normalized_index]],
                        source="cifar_import",
                        metadata={"import_id": import_id, "source_partition": partition},
                    )
                print(f"Imported {partition}: {len(images)} images", flush=True)
            connection.execute(
                """
                INSERT INTO import_events
                (import_id, dataset_id, revision_id, created_at, importer, source_uri, options_json, summary_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    import_id,
                    info["dataset_id"],
                    info["revision_id"],
                    utc_now(),
                    "cifar",
                    f"keras://datasets/{options.dataset}",
                    json.dumps({"dataset": options.dataset}, sort_keys=True),
                    json.dumps(report, sort_keys=True),
                ),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
    connection.close()
    if not options.dry_run:
        output.with_suffix(".import_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        output.with_suffix(".labels.json").write_text(
            json.dumps({name: index for index, name in enumerate(labels)}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download CIFAR-10 or CIFAR-100 into an Oracle Builder SQLite dataset.")
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), required=True)
    parser.add_argument("--output", required=True, help="New SQLite dataset path.")
    parser.add_argument("--dataset-id", help="Optional UUID for deterministic dataset identity.")
    parser.add_argument("--dry-run", action="store_true", help="Download and validate without writing a database.")
    return parser


def main() -> int:
    options = build_parser().parse_args()
    try:
        report = import_cifar(options)
    except (FileExistsError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
