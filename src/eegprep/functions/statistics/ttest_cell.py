"""Paired t-test helper."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.statistics._shared import stat_mean, stat_std, two_arrays


def ttest_cell(a: Any, b: Any | None = None, *, axis: int = -1) -> tuple[np.ndarray, int]:
    """Compute paired t-statistics across the case axis."""

    first, second = two_arrays(a, b, "ttest_cell", axis=axis)
    if first.shape != second.shape:
        raise ValueError("ttest_cell requires paired arrays with identical shapes")
    n_cases = first.shape[-1]
    if n_cases < 2:
        raise ValueError("ttest_cell requires at least two paired cases")

    difference = first - second
    mean_difference = stat_mean(difference, axis=-1)
    sd_difference = stat_std(difference, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = mean_difference / sd_difference * np.sqrt(n_cases)
    return t_values, n_cases - 1


__all__ = ["ttest_cell"]
