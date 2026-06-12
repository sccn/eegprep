"""Unpaired t-test helper."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.statistics._shared import stat_mean, stat_std, two_arrays


def ttest2_cell(
    a: Any,
    b: Any | None = None,
    variance: str = "homogenous",
    *,
    axis: int = -1,
) -> tuple[np.ndarray, np.ndarray | int]:
    """Compute unpaired t-statistics across the case axis."""

    if isinstance(b, str):
        variance = b
        b = None
    first, second = two_arrays(a, b, "ttest2_cell", axis=axis)
    if first.shape[:-1] != second.shape[:-1]:
        raise ValueError("ttest2_cell requires matching feature shapes before the case axis")
    if first.shape[-1] < 2 or second.shape[-1] < 2:
        raise ValueError("ttest2_cell requires at least two cases in each group")

    variance_name = variance.lower()
    if variance_name not in {"homogenous", "inhomogenous"}:
        raise ValueError("variance must be 'homogenous' or 'inhomogenous'")

    first_n = first.shape[-1]
    second_n = second.shape[-1]
    first_mean = stat_mean(first, axis=-1)
    second_mean = stat_mean(second, axis=-1)
    if variance_name == "inhomogenous":
        first_scaled = np.var(first, axis=-1, ddof=1) / first_n
        second_scaled = np.var(second, axis=-1, ddof=1) / second_n
        standard_error = np.sqrt(first_scaled + second_scaled)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_values = (first_mean - second_mean) / standard_error
            df = (first_scaled + second_scaled) ** 2 / (
                first_scaled**2 / (first_n - 1) + second_scaled**2 / (second_n - 1)
            )
        return t_values, df

    first_sd = stat_std(first, axis=-1)
    second_sd = stat_std(second, axis=-1)
    pooled_sd = np.sqrt(((first_n - 1) * first_sd**2 + (second_n - 1) * second_sd**2) / (first_n + second_n - 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = (first_mean - second_mean) / pooled_sd / np.sqrt(1 / first_n + 1 / second_n)
    return t_values, first_n + second_n - 2


__all__ = ["ttest2_cell"]
