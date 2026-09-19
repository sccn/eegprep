"""Channel-label-aware map matching."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ._validation import real_array
from .matcorr import _apply_weighting, _cosine_rows, _match_correlations


def mapcorr(
    first: Any,
    second: Any,
    first_channels: Sequence[Mapping[str, Any]],
    second_channels: Sequence[Mapping[str, Any]],
    remove_mean: bool = False,
    method: int = 2,
    weighting: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Match rows after aligning map columns by common channel labels."""
    left = real_array(first, "first")
    right = real_array(second, "second")
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("map matrices must be 2-D")
    if left.shape[1] != len(first_channels) or right.shape[1] != len(second_channels):
        raise ValueError("channel-location counts must match map columns")
    left_labels = [str(channel.get("labels", "")).strip() for channel in first_channels]
    right_labels = [str(channel.get("labels", "")).strip() for channel in second_channels]
    if not all(left_labels) or not all(right_labels):
        raise ValueError("channel labels must be nonempty")
    if len(set(left_labels)) != len(left_labels) or len(set(right_labels)) != len(right_labels):
        raise ValueError("channel labels must be unique within each map")
    right_by_label = {label: index for index, label in enumerate(right_labels)}
    left_indices: list[int] = []
    right_indices: list[int] = []
    for left_index, channel in enumerate(first_channels):
        label = left_labels[left_index]
        if label in right_by_label:
            left_indices.append(left_index)
            right_indices.append(right_by_label[label])
    if not left_indices:
        raise ValueError("the channel sets have no labels in common")
    if remove_mean:
        left = left - np.mean(left, axis=1, keepdims=True)
        right = right - np.mean(right, axis=1, keepdims=True)
    correlations = _cosine_rows(left[:, left_indices], right[:, right_indices])
    correlations = _apply_weighting(correlations, weighting)
    return _match_correlations(correlations, method)


__all__ = ["mapcorr"]
