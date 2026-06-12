"""One-way repeated-measures ANOVA helper."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.statistics._shared import anova_values, one_way_arrays, require_same_shapes


def anova1rm_cell(data: Any, *, axis: int = -1) -> tuple[np.ndarray, tuple[int, int]]:
    """Compute one-way repeated-measures ANOVA F-statistics."""

    arrays = one_way_arrays(data, axis=axis, paired=True)
    if len(arrays) < 2:
        raise ValueError("anova1rm_cell requires at least two conditions")
    require_same_shapes(arrays, "anova1rm_cell")
    n_cases = arrays[0].shape[-1]
    if n_cases < 2:
        raise ValueError("anova1rm_cell requires at least two repeated cases")

    stacked = np.stack(arrays, axis=-2)
    values = anova_values(stacked)
    n_conditions = len(arrays)
    condition_subject = values
    condition_sums = np.sum(condition_subject, axis=-1)
    subject_sums = np.sum(condition_subject, axis=-2)
    total_sum = np.sum(condition_sums, axis=-1)

    df_condition = n_conditions - 1
    df_error = (n_conditions - 1) * (n_cases - 1)
    expected_condition = np.sum(condition_sums**2, axis=-1) / n_cases
    expected_subject = np.sum(subject_sums**2, axis=-1) / n_conditions
    expected_condition_subject = np.sum(condition_subject**2, axis=(-2, -1))
    expected_total = total_sum**2 / (n_conditions * n_cases)

    ss_condition = expected_condition - expected_total
    ss_error = expected_condition_subject - expected_condition - expected_subject + expected_total
    with np.errstate(divide="ignore", invalid="ignore"):
        f_values = (ss_condition / df_condition) / (ss_error / df_error)
    return f_values, (df_condition, df_error)


__all__ = ["anova1rm_cell"]
