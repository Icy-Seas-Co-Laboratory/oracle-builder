#!/usr/bin/env python3
"""Idempotently generate every focused Architecture V2 classification TOML.

The shared values are owned by ``oracle_builder.config_schema``. Running this
script is safe: it replaces only generated files in ``classification_defaults``
and leaves examples and historical provenance untouched.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oracle_builder.config import deep_merge
from oracle_builder.config_schema import (
    classification_preset_defaults,
    classification_stratification_preset,
)


@dataclass(frozen=True)
class Recipe:
    filename: str
    model: str
    family: str
    variant: str
    summary: str
    stem_kernel: int = 3
    stem_stride: int = 1
    stem_pool: str = "none"
    family_baseline: bool = False


def _variants(family: str, names: tuple[str, ...], *, kernel: int = 3, stride: int = 2, pool: str = "none") -> tuple[Recipe, ...]:
    return tuple(
        Recipe(name, name, family, name, f"{name.replace('_', '-').title()} {family} variant.", kernel, stride, pool)
        for name in names
    )


RECIPES: tuple[Recipe, ...] = (
    Recipe("simple_cnn", "simple_cnn", "simple_cnn", "simple_cnn", "Fast three-stage convolutional baseline.", family_baseline=True),
    Recipe("resnet_like", "resnet_like", "resnet_like", "resnet_like", "Compact shallow residual baseline.", family_baseline=True),
    Recipe("densenet_like", "densenet_like", "densenet_like", "densenet_like", "Compact shallow densely connected baseline.", 7, 2, "max", True),
    Recipe("resnet", "resnet", "resnet", "resnet18", "ResNet family default (ResNet-18).", family_baseline=True),
    *_variants("resnet", ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152"), kernel=3, stride=1),
    Recipe("densenet", "densenet", "densenet", "densenet121", "DenseNet family default (DenseNet-121).", 7, 2, "max", True),
    *_variants("densenet", ("densenet121", "densenet169", "densenet201"), kernel=7, stride=2, pool="max"),
    Recipe("efficientnet", "efficientnet", "efficientnet", "efficientnet_b0", "EfficientNet family default (B0).", family_baseline=True),
    *_variants("efficientnet", tuple(f"efficientnet_b{index}" for index in range(8))),
    Recipe("efficientnet_v2", "efficientnet_v2", "efficientnet_v2", "efficientnet_v2_b0", "EfficientNetV2 family default (B0)."),
    *_variants("efficientnet_v2", ("efficientnet_v2_b0", "efficientnet_v2_b1", "efficientnet_v2_b2", "efficientnet_v2_b3", "efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l")),
    Recipe("convnext", "convnext", "convnext", "convnext_tiny", "ConvNeXt family default (Tiny).", 4, 4),
    *_variants("convnext", ("convnext_tiny", "convnext_small"), kernel=4, stride=4),
    Recipe("mobilenet", "mobilenet", "mobilenet", "mobilenet_v3_small", "MobileNetV3 family default (Small)."),
    *_variants("mobilenet", ("mobilenet_v3_small", "mobilenet_v3_large")),
)


def _toml_value(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        raise ValueError("Focused recipes must not emit null TOML values")
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML value {value!r}")


SECTION_ORDER = (
    "run", "architecture", "input", "input.geometry", "data", "data.streaming",
    "preprocessing", "encoder", "stem", "normalization", "pooling",
    "image_embedding.projection", "metadata", "metadata.encoder", "metadata.augmentation",
    "fusion", "classifier", "classification.stratification",
    "classification.stratification.conditioning", "classification.stratification.training_routing",
    "classification.stratification.cycle_scheduler", "training", "training.class_weights", "callbacks",
    "augmentation", "distribution", "evidence", "output",
)

FAMILY_GUIDANCE = {
    "simple_cnn": "Smallest bundled encoder; use it to validate a dataset or streaming pipeline.",
    "resnet_like": "Compact residual encoder; use the resnet family for canonical depth variants.",
    "densenet_like": "Compact dense encoder; use the densenet family for canonical DenseNet variants.",
    "resnet": "18/34 use basic blocks; 50/101/152 use bottleneck blocks and cost more memory.",
    "densenet": "DenseNet-121 is the usual first run; larger variants increase activation memory.",
    "efficientnet": "B0 is the economical baseline; larger B variants usually benefit from larger inputs.",
    "efficientnet_v2": "Use B0 for compact inputs; S/M/L are progressively larger training targets.",
    "convnext": "Tiny is the compact baseline; Small adds capacity at a substantial compute cost.",
    "mobilenet": "Small is intended for constrained compute; Large trades speed for capacity.",
}


def _section_comments(recipe: Recipe, section: str) -> tuple[str, ...]:
    comments = {
        "data": ("# The SQLite dataset supplies class count, label order, and durable source identity.",),
        "data.streaming": ("# Decode only active batches; increase workers only when CPU decode limits throughput.",),
        "preprocessing": ("# fit_pad preserves the whole ROI. This contract travels with the saved model.",),
        "encoder": ("# V2 separates encoder family from its concrete topology.",),
        "stem": ("# Explicit native stem. Smaller stride/pool preserves fine detail at higher cost.",),
        "image_embedding.projection": ("# The fixed-size image representation consumed by fusion and the classifier.",),
        "metadata.augmentation": ("# Gaussian metadata noise is training-only; leave zero unless regularization is intended.",),
        "training.class_weights": ("# Effective-number weighting is less volatile than inverse-frequency weighting.",),
        "augmentation": ("# Online augmentation remains stochastic even when deterministic prepared-input caching is enabled.",),
        "distribution": ("# single uses one selected GPU; select mirrored only for intentional multi-GPU training.",),
        "evidence": ("# Retain embeddings, prototypes, and nearest-neighbor evidence with predictions.",),
    }
    return comments.get(section, ())


def _table(config: dict[str, Any], path: str) -> dict[str, Any]:
    value: Any = config
    for component in path.split("."):
        if not isinstance(value, dict) or component not in value:
            return {}
        value = value[component]
    assert isinstance(value, dict)
    return value


def render(recipe: Recipe) -> str:
    config = deep_merge(classification_preset_defaults(), {
        "run": {"notes": f"{recipe.variant} V2 classification preset"},
        "encoder": {"family": recipe.family, "variant": recipe.variant},
        "stem": {"type": "native", "kernel_size": recipe.stem_kernel, "stride": recipe.stem_stride, "pool": recipe.stem_pool},
    })
    if recipe.family_baseline:
        config = deep_merge(config, classification_stratification_preset())
        config["training"].update({"learning_rate": 0.0003, "weight_decay": 0.0001})
    lines = [
        f"# {recipe.summary}",
        "# Generated by scripts/generate_classification_default_configs.py; do not edit by hand.",
        "# Copy this V2 preset for routine classification. Advanced settings inherit",
        "# runtime defaults; see configs/reference_v2_exhaustive.toml for every option.",
        "",
    ]
    for section in SECTION_ORDER:
        if not _table(config, section):
            continue
        lines.append(f"[{section}]")
        lines.extend(_section_comments(recipe, section))
        for key, value in _table(config, section).items():
            if not isinstance(value, dict):
                lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate focused V2 classification presets.")
    parser.add_argument("--output", type=Path, default=Path("configs/classification_defaults"))
    parser.add_argument("--check", action="store_true", help="Fail instead of writing when a generated recipe is stale.")
    args = parser.parse_args()
    stale: list[Path] = []
    for recipe in RECIPES:
        target = args.output / f"{recipe.filename}.toml"
        rendered = render(recipe)
        if target.is_file() and target.read_text(encoding="utf-8") == rendered:
            continue
        stale.append(target)
        if not args.check:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(rendered, encoding="utf-8")
    if args.check and stale:
        print("Generated V2 presets are stale:\n" + "\n".join(str(path) for path in stale))
        return 1
    print(f"{'Verified' if args.check else 'Wrote'} {len(RECIPES)} V2 classification presets in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
