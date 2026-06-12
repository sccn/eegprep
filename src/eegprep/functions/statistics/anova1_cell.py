"""One-way unpaired ANOVA helper."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.statistics._shared import one_way_arrays, stat_mean, sum_square_residuals


def anova1_cell(data: Any, *, axis: int = -1) -> tuple[np.ndarray, tuple[int, int]]:
    """Compute one-way unpaired ANOVA F-statistics across condition arrays."""

    arrays = one_way_arrays(data, axis=axis, paired=False)
    if len(arrays) < 2:
        raise ValueError("anova1_cell requires at least two conditions")
    feature_shape = arrays[0].shape[:-1]
    for index, array in enumerate(arrays):
        if array.shape[:-1] != feature_shape:
            raise ValueError(f"condition {index} has feature shape {array.shape[:-1]}, expected {feature_shape}")
        if array.shape[-1] < 2:
            raise ValueError("anova1_cell requires at least two cases in each condition")

    counts = np.array([array.shape[-1] for array in arrays], dtype=float)
    means = np.stack([stat_mean(array, axis=-1) for array in arrays], axis=-1)
    total_n = int(np.sum(counts))
    grand_mean = np.sum(means * counts, axis=-1) / total_n
    ss_between = np.sum(counts * (means - np.expand_dims(grand_mean, -1)) ** 2, axis=-1)
    ss_within = np.zeros(feature_shape, dtype=float)
    for array, mean in zip(arrays, np.moveaxis(means, -1, 0), strict=True):
        ss_within = ss_within + sum_square_residuals(array, mean, axis=-1)

    df_between = len(arrays) - 1
    df_within = total_n - len(arrays)
    with np.errstate(divide="ignore", invalid="ignore"):
        f_values = (ss_between / df_between) / (ss_within / df_within)
    return f_values, (df_between, df_within)


__all__ = ["anova1_cell"]
