from __future__ import annotations

import math
from typing import Any

import tensorflow as tf

from oracle_builder.classification.metadata import gaussian_noise_indices
from oracle_builder.data.channels import derive_channels_tensor, has_derived_channels


def apply_training_augmentation(dataset, config: dict[str, Any]):
    augmentation = config.get("augmentation", {})
    metadata_augmentation = config.get("metadata", {}).get("augmentation", {})
    metadata_noise_enabled = bool(
        isinstance(metadata_augmentation, dict)
        and float(metadata_augmentation.get("gaussian_variance", 0.0)) > 0
    )
    if not augmentation.get("enabled", False) and not metadata_noise_enabled:
        return dataset
    element_spec = dataset.element_spec
    if isinstance(element_spec, (tuple, list)) and len(element_spec) == 3:
        return dataset.map(
            lambda x, y, weights: augment_batch(x, y, config, weights),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return dataset.map(lambda x, y: augment_batch(x, y, config), num_parallel_calls=tf.data.AUTOTUNE)


def augment_batch(x, y, config: dict[str, Any], sample_weight=None):
    augmentation = config.get("augmentation", {})
    metadata_augmentation = config.get("metadata", {}).get("augmentation", {})
    metadata_noise_enabled = bool(
        isinstance(metadata_augmentation, dict)
        and float(metadata_augmentation.get("gaussian_variance", 0.0)) > 0
    )
    if not augmentation.get("enabled", False) and not metadata_noise_enabled:
        if sample_weight is None:
            return x, y
        return x, y, sample_weight
    # Classifiers with auxiliary scalar features arrive as a named Keras input
    # dictionary. Geometric/photometric augmentation applies only to pixels;
    # ROI metadata describes the original item and must remain unchanged.
    auxiliary_inputs = x if isinstance(x, dict) else None
    # Metadata regularization deliberately does not depend on image
    # augmentation being enabled. It is training-only because this function is
    # applied solely to the train dataset by both data loaders.
    if not augmentation.get("enabled", False):
        result_x = augment_metadata_inputs(auxiliary_inputs, config)
        if sample_weight is not None:
            return result_x if result_x is not None else x, y, sample_weight
        return result_x if result_x is not None else x, y
    if auxiliary_inputs is not None:
        x = auxiliary_inputs["image"]
    preprocessing = config.get("preprocessing", {})
    # Materialized legacy datasets contain derived channels already.  Opt in to
    # this policy to discard those stale features, augment intensity, then
    # regenerate all requested channels from the augmented image.
    derive_after_augmentation = bool(preprocessing.get("derive_after_augmentation", False)) and has_derived_channels(preprocessing)
    if derive_after_augmentation:
        x = x[..., :1]
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

    crop_strength = float(augmentation.get("random_resized_crop", 0.0))
    if crop_strength:
        crop_boxes = build_random_resized_crop_boxes(x, crop_strength)
        x = crop_input_channels(
            x,
            crop_boxes,
            mask_channels=input_mask_channels(config, augmentation),
            distance_channels=signed_distance_input_channels(config, augmentation),
        )
        if task == "segmentation":
            y = apply_random_resized_crop(y, crop_boxes, interpolation="nearest")
            y = tf.cast(y > 0.5, tf.float32)
            if sample_weight is not None:
                sample_weight = apply_random_resized_crop(
                    sample_weight[..., None], crop_boxes, interpolation="bilinear"
                )[..., 0]

    if bool(augmentation.get("rotate_90", False)):
        turns = random_right_angle_turns(x)
        x = tf.image.rot90(x, turns)
        if task == "segmentation":
            y = tf.image.rot90(y, turns)
            if sample_weight is not None:
                sample_weight = tf.image.rot90(sample_weight[..., None], turns)[..., 0]

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
    if derive_after_augmentation:
        x = derive_channels_tensor(x, preprocessing)
    result_x = (
        {**auxiliary_inputs, "image": x}
        if auxiliary_inputs is not None
        else x
    )
    result_x = augment_metadata_inputs(result_x, config)
    if sample_weight is not None:
        return result_x, tf.cast(y, y_dtype), tf.cast(sample_weight, tf.float32)
    return result_x, tf.cast(y, y_dtype)


def augment_metadata_inputs(inputs, config: dict[str, Any]):
    """Add zero-mean Gaussian noise to selected *scaled* metadata columns."""
    if not isinstance(inputs, dict) or "metadata" not in inputs:
        return inputs
    settings = config.get("metadata", {}).get("augmentation", {})
    if not isinstance(settings, dict):
        return inputs
    variance = float(settings.get("gaussian_variance", 0.0))
    probability = float(settings.get("probability", 1.0))
    indices = gaussian_noise_indices(config)
    if variance <= 0 or probability <= 0 or not indices:
        return inputs
    metadata = tf.cast(inputs["metadata"], tf.float32)
    width = metadata.shape[-1]
    if width is None:
        raise ValueError("metadata Gaussian augmentation requires a known feature width")
    if any(index >= int(width) for index in indices):
        raise ValueError("metadata Gaussian augmentation index exceeds metadata input width")
    eligible = tf.reduce_sum(
        tf.one_hot(indices, depth=int(width), dtype=metadata.dtype), axis=0
    )
    noise = tf.random.normal(tf.shape(metadata), stddev=tf.sqrt(tf.cast(variance, metadata.dtype)))
    apply = tf.cast(tf.random.uniform([tf.shape(metadata)[0], 1]) < probability, metadata.dtype)
    return {**inputs, "metadata": metadata + noise * eligible[tf.newaxis, :] * apply}


def build_random_affine_transforms(x, augmentation: dict[str, Any]):
    rotation = float(augmentation.get("rotation", 0.0))
    zoom = float(augmentation.get("zoom", 0.0))
    zoom_x = augmentation.get("zoom_x", zoom)
    zoom_y = augmentation.get("zoom_y", zoom)
    zoom_x = zoom if zoom_x is None else float(zoom_x)
    zoom_y = zoom if zoom_y is None else float(zoom_y)
    translation = augmentation.get("translation", 0.0)
    skew = float(augmentation.get("skew", augmentation.get("shear", 0.0)))
    if not any((rotation, zoom_x, zoom_y, translation, skew)):
        return None

    batch = tf.shape(x)[0]
    height = tf.cast(tf.shape(x)[1], tf.float32)
    width = tf.cast(tf.shape(x)[2], tf.float32)
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0

    angle = tf.random.uniform([batch], -rotation * 2.0 * math.pi, rotation * 2.0 * math.pi)
    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)

    zoom_x = min(max(zoom_x, 0.0), 0.95)
    zoom_y = min(max(zoom_y, 0.0), 0.95)
    scale_x = tf.random.uniform([batch], 1.0 - zoom_x, 1.0 + zoom_x)
    scale_y = tf.random.uniform([batch], 1.0 - zoom_y, 1.0 + zoom_y)
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
    scale_matrix = matrix_from_values(scale_x, tf.zeros([batch]), tf.zeros([batch]), tf.zeros([batch]), scale_y, tf.zeros([batch]))
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


def build_random_resized_crop_boxes(x, strength: float):
    """Return one aspect-preserving random crop box per batch item.

    A crop is resized back to the original geometry, which makes this safe for
    batched training and keeps classification and segmentation paths aligned.
    ``strength`` controls the smallest retained side fraction.
    """
    strength = float(strength)
    if not 0 <= strength < 1:
        raise ValueError("augmentation.random_resized_crop must be in [0, 1)")
    batch = tf.shape(x)[0]
    scale = tf.random.uniform([batch], 1.0 - strength, 1.0)
    y0 = tf.random.uniform([batch], 0.0, 1.0) * (1.0 - scale)
    x0 = tf.random.uniform([batch], 0.0, 1.0) * (1.0 - scale)
    return tf.stack([y0, x0, y0 + scale, x0 + scale], axis=1)


def apply_random_resized_crop(values, boxes, *, interpolation: str):
    """Crop each batch member and resize to its original spatial shape."""
    return tf.image.crop_and_resize(
        tf.cast(values, tf.float32),
        boxes,
        tf.range(tf.shape(values)[0]),
        tf.shape(values)[1:3],
        method=interpolation,
    )


def crop_input_channels(x, boxes, *, mask_channels: list[int], distance_channels: list[int]):
    """Crop image channels with the interpolation appropriate to their role."""
    channel_count = x.shape[-1]
    if channel_count is None:
        raise ValueError("Input channel count must be known for geometric augmentation")
    mask_set = set(mask_channels)
    distance_set = set(distance_channels)
    values = []
    for index in range(int(channel_count)):
        interpolation = "nearest" if index in mask_set else "bilinear"
        # Signed distance fields are continuous, so they deliberately share
        # bilinear interpolation with intensity and derived scalar channels.
        if index in distance_set:
            interpolation = "bilinear"
        values.append(apply_random_resized_crop(x[..., index:index + 1], boxes, interpolation=interpolation))
    return tf.concat(values, axis=-1)


def random_right_angle_turns(x):
    """Select a right-angle rotation without changing non-square tensor shape."""
    height, width = x.shape[1], x.shape[2]
    if height is not None and width is not None and height == width:
        return tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    # 180 degrees has the same geometry for any rectangular input.  Dynamic
    # shapes use this conservative branch because tf.data requires static
    # batch element shapes to remain compatible after mapping.
    return 2 * tf.random.uniform((), minval=0, maxval=2, dtype=tf.int32)


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
