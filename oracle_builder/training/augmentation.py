from __future__ import annotations

import math
from typing import Any

import tensorflow as tf


def apply_training_augmentation(dataset, config: dict[str, Any]):
    augmentation = config.get("augmentation", {})
    if not augmentation.get("enabled", False):
        return dataset
    if config.get("training", {}).get("spatial_edge_weighting", False):
        return dataset.map(
            lambda x, y, weights: augment_batch(x, y, config, weights),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return dataset.map(lambda x, y: augment_batch(x, y, config), num_parallel_calls=tf.data.AUTOTUNE)


def augment_batch(x, y, config: dict[str, Any], sample_weight=None):
    augmentation = config.get("augmentation", {})
    task = config["run"]["task"]
    x = tf.cast(x, tf.float32)
    y_dtype = y.dtype
    y = tf.cast(y, tf.float32)

    transforms = build_random_affine_transforms(x, augmentation)
    if transforms is not None:
        x = transform_input_channels(
            x,
            transforms,
            fill_value=float(augmentation.get("fill_value", 0.0)),
            mask_channels=input_mask_channels(config, augmentation),
            distance_channels=signed_distance_input_channels(config, augmentation),
            distance_fill_value=float(augmentation.get("signed_distance_fill_value", -1.0)),
        )
        if task == "segmentation":
            y = apply_affine_transform(
                y,
                transforms,
                interpolation="NEAREST",
                fill_value=float(augmentation.get("mask_fill_value", 0.0)),
            )
            y = tf.cast(y > 0.5, tf.float32)
            if sample_weight is not None:
                sample_weight = apply_affine_transform(
                sample_weight[..., None],
                transforms,
                interpolation="BILINEAR",
                fill_value=0.0 if config.get("tiling", {}).get("enabled", False) else 1.0,
                )[..., 0]

    if bool(augmentation.get("flip_horizontal", False)):
        do_flip = tf.random.uniform(()) < 0.5
        x = tf.cond(do_flip, lambda: tf.image.flip_left_right(x), lambda: x)
        if task == "segmentation":
            y = tf.cond(do_flip, lambda: tf.image.flip_left_right(y), lambda: y)
            if sample_weight is not None:
                sample_weight = tf.cond(
                    do_flip,
                    lambda: tf.image.flip_left_right(sample_weight[..., None])[..., 0],
                    lambda: sample_weight,
                )
    if bool(augmentation.get("flip_vertical", False)):
        do_flip = tf.random.uniform(()) < 0.5
        x = tf.cond(do_flip, lambda: tf.image.flip_up_down(x), lambda: x)
        if task == "segmentation":
            y = tf.cond(do_flip, lambda: tf.image.flip_up_down(y), lambda: y)
            if sample_weight is not None:
                sample_weight = tf.cond(
                    do_flip,
                    lambda: tf.image.flip_up_down(sample_weight[..., None])[..., 0],
                    lambda: sample_weight,
                )

    x = apply_photometric_augmentation(x, config, augmentation)
    if sample_weight is not None:
        return x, tf.cast(y, y_dtype), tf.cast(sample_weight, tf.float32)
    return x, tf.cast(y, y_dtype)


def build_random_affine_transforms(x, augmentation: dict[str, Any]):
    rotation = float(augmentation.get("rotation", 0.0))
    zoom = float(augmentation.get("zoom", 0.0))
    translation = augmentation.get("translation", 0.0)
    skew = float(augmentation.get("skew", augmentation.get("shear", 0.0)))
    if not any((rotation, zoom, translation, skew)):
        return None

    batch = tf.shape(x)[0]
    height = tf.cast(tf.shape(x)[1], tf.float32)
    width = tf.cast(tf.shape(x)[2], tf.float32)
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0

    angle = tf.random.uniform([batch], -rotation * 2.0 * math.pi, rotation * 2.0 * math.pi)
    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)

    zoom = min(max(zoom, 0.0), 0.95)
    scale = tf.random.uniform([batch], 1.0 - zoom, 1.0 + zoom)
    shear_x = tf.random.uniform([batch], -skew, skew)
    shear_y = tf.zeros([batch], dtype=tf.float32)
    translate_y, translate_x = translation_fractions(translation)
    dx = tf.random.uniform([batch], -translate_x * width, translate_x * width)
    dy = tf.random.uniform([batch], -translate_y * height, translate_y * height)

    center = translation_matrix(tf.fill([batch], cx), tf.fill([batch], cy))
    uncenter = translation_matrix(tf.fill([batch], -cx), tf.fill([batch], -cy))
    move = translation_matrix(dx, dy)
    rotate = matrix_from_values(cos_a, -sin_a, tf.zeros([batch]), sin_a, cos_a, tf.zeros([batch]))
    shear = matrix_from_values(
        tf.ones([batch]),
        shear_x,
        tf.zeros([batch]),
        shear_y,
        tf.ones([batch]),
        tf.zeros([batch]),
    )
    scale_matrix = matrix_from_values(scale, tf.zeros([batch]), tf.zeros([batch]), tf.zeros([batch]), scale, tf.zeros([batch]))
    forward = tf.linalg.matmul(move, tf.linalg.matmul(center, tf.linalg.matmul(rotate, tf.linalg.matmul(shear, tf.linalg.matmul(scale_matrix, uncenter)))))
    inverse = tf.linalg.inv(forward)
    return tf.stack(
        [
            inverse[:, 0, 0],
            inverse[:, 0, 1],
            inverse[:, 0, 2],
            inverse[:, 1, 0],
            inverse[:, 1, 1],
            inverse[:, 1, 2],
            inverse[:, 2, 0],
            inverse[:, 2, 1],
        ],
        axis=1,
    )


def translation_fractions(value: Any) -> tuple[float, float]:
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError("augmentation.translation must be a scalar or [height_fraction, width_fraction].")
        return float(value[0]), float(value[1])
    fraction = float(value)
    return fraction, fraction


def translation_matrix(dx, dy):
    batch = tf.shape(dx)[0]
    zeros = tf.zeros([batch], dtype=tf.float32)
    ones = tf.ones([batch], dtype=tf.float32)
    return matrix_from_values(ones, zeros, dx, zeros, ones, dy)


def matrix_from_values(a00, a01, a02, a10, a11, a12):
    batch = tf.shape(a00)[0]
    zeros = tf.zeros([batch], dtype=tf.float32)
    ones = tf.ones([batch], dtype=tf.float32)
    return tf.reshape(
        tf.stack([a00, a01, a02, a10, a11, a12, zeros, zeros, ones], axis=1),
        [batch, 3, 3],
    )


def transform_input_channels(
    x,
    transforms,
    fill_value: float,
    mask_channels: list[int],
    distance_channels: list[int] | None = None,
    distance_fill_value: float = -1.0,
):
    channel_count = x.shape[-1]
    if channel_count is None:
        return apply_affine_transform(x, transforms, interpolation="BILINEAR", fill_value=fill_value)
    transformed_channels = []
    distance_channel_set = set(distance_channels or [])
    for index in range(int(channel_count)):
        interpolation = "NEAREST" if index in mask_channels else "BILINEAR"
        channel_fill_value = distance_fill_value if index in distance_channel_set else fill_value
        transformed = apply_affine_transform(
            x[..., index : index + 1],
            transforms,
            interpolation=interpolation,
            fill_value=channel_fill_value,
        )
        if index in mask_channels:
            transformed = tf.cast(transformed > 0.5, tf.float32)
        transformed_channels.append(transformed)
    return tf.concat(transformed_channels, axis=-1)


def apply_affine_transform(image, transforms, interpolation: str, fill_value: float):
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=image,
        transforms=transforms,
        output_shape=tf.shape(image)[1:3],
        interpolation=interpolation,
        fill_mode="CONSTANT",
        fill_value=float(fill_value),
    )


def apply_photometric_augmentation(x, config: dict[str, Any], augmentation: dict[str, Any]):
    channels = photometric_channels(config, augmentation)
    if not channels:
        return x
    values = x
    selected = gather_channels(values, channels)

    if bool(augmentation.get("invert", False)):
        selected = 1.0 - selected
    brightness = float(augmentation.get("brightness", 0.0))
    if brightness:
        delta = tf.random.uniform([tf.shape(selected)[0], 1, 1, 1], -brightness, brightness)
        selected = selected + delta
    contrast = float(augmentation.get("contrast", 0.0))
    if contrast:
        factor = tf.random.uniform([tf.shape(selected)[0], 1, 1, 1], max(0.0, 1.0 - contrast), 1.0 + contrast)
        mean = tf.reduce_mean(selected, axis=[1, 2], keepdims=True)
        selected = (selected - mean) * factor + mean
    noise = float(augmentation.get("gaussian_noise", 0.0))
    if noise:
        selected = selected + tf.random.normal(tf.shape(selected), stddev=noise)
    selected = tf.clip_by_value(selected, 0.0, 1.0)
    return replace_channels(values, channels, selected)


def gather_channels(x, channels: list[int]):
    return tf.concat([x[..., index : index + 1] for index in channels], axis=-1)


def replace_channels(x, channels: list[int], replacement):
    channel_count = x.shape[-1]
    if channel_count is None:
        return x
    pieces = []
    replacement_index = 0
    channel_set = set(channels)
    for index in range(int(channel_count)):
        if index in channel_set:
            pieces.append(replacement[..., replacement_index : replacement_index + 1])
            replacement_index += 1
        else:
            pieces.append(x[..., index : index + 1])
    return tf.concat(pieces, axis=-1)


def input_mask_channels(config: dict[str, Any], augmentation: dict[str, Any]) -> list[int]:
    if "mask_input_channels" in augmentation:
        return [int(channel) for channel in augmentation.get("mask_input_channels") or []]
    input_shape = config.get("data", {}).get("input_shape", [])
    if config.get("run", {}).get("task") == "segmentation" and input_shape and (
        int(input_shape[-1]) == 2 or config.get("data", {}).get("candidate_sdf", False)
    ):
        return [1]
    return []


def signed_distance_input_channels(config: dict[str, Any], augmentation: dict[str, Any]) -> list[int]:
    if "signed_distance_input_channels" in augmentation:
        return [int(channel) for channel in augmentation.get("signed_distance_input_channels") or []]
    if config.get("data", {}).get("candidate_sdf", False):
        return [2]
    return []


def photometric_channels(config: dict[str, Any], augmentation: dict[str, Any]) -> list[int]:
    if "photometric_channels" in augmentation:
        return [int(channel) for channel in augmentation.get("photometric_channels") or []]
    input_shape = config.get("data", {}).get("input_shape", [])
    if not input_shape:
        return []
    channels = int(input_shape[-1])
    if config.get("run", {}).get("task") == "segmentation" and (
        channels == 2 or config.get("data", {}).get("candidate_sdf", False)
    ):
        return [0]
    return list(range(channels))
