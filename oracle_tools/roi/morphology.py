"""Small NumPy-only morphology helpers for binary ROI masks."""
from __future__ import annotations

from collections import deque
import numpy as np


def _binary(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"Morphology expects a 2D mask, got shape {array.shape}")
    return array.astype(bool)


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, list[int]]:
    binary = _binary(mask)
    labels, sizes, label = np.zeros(binary.shape, dtype="int32"), [], 0
    height, width = binary.shape
    for y in range(height):
        for x in range(width):
            if not binary[y, x] or labels[y, x]:
                continue
            label, count, queue = label + 1, 0, deque([(y, x)])
            labels[y, x] = label
            while queue:
                cy, cx = queue.popleft(); count += 1
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if (dy or dx) and 0 <= ny < height and 0 <= nx < width and binary[ny, nx] and not labels[ny, nx]:
                            labels[ny, nx] = label; queue.append((ny, nx))
            sizes.append(count)
    return labels, sizes


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes in a binary mask."""
    binary, visited = _binary(mask), None
    background = ~binary
    visited = np.zeros(binary.shape, dtype=bool)
    height, width, queue = *binary.shape, deque()
    for y in range(height):
        for x in (0, width - 1):
            if background[y, x] and not visited[y, x]:
                visited[y, x] = True; queue.append((y, x))
    for x in range(width):
        for y in (0, height - 1):
            if background[y, x] and not visited[y, x]:
                visited[y, x] = True; queue.append((y, x))
    while queue:
        cy, cx = queue.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < height and 0 <= nx < width and background[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True; queue.append((ny, nx))
    return (binary | (background & ~visited)).astype("uint8")


def remove_small_objects(mask: np.ndarray, min_size: int) -> np.ndarray:
    labels, sizes = connected_components(mask)
    keep = np.zeros(labels.shape, dtype=bool)
    for index, size in enumerate(sizes, start=1):
        if size >= min_size:
            keep |= labels == index
    return keep.astype("uint8")


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    labels, sizes = connected_components(mask)
    return np.zeros(labels.shape, dtype="uint8") if not sizes else (labels == int(np.argmax(sizes)) + 1).astype("uint8")
