"""Two-way unpaired ANOVA helper."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.statistics._shared import TwoWayAnovaResult, anova_values, stat_mean, two_way_stack


def anova2_cell(data: Any, *, axis: int = -1) -> TwoWayAnovaResult:
    """Compute balanced two-way unpaired ANOVA F-statistics."""

    stacked = two_way_stack(data, axis=axis, name="anova2_cell")
    values = anova_values(stacked)
    n_rows = stacked.shape[-3]
    n_columns = stacked.shape[-2]
    n_cases = stacked.shape[-1]
    if n_rows < 2 or n_columns < 2:
        raise ValueError("anova2_cell requires at least two rows and two columns")
    if n_cases < 2:
        raise ValueError("anova2_cell requires at least two cases in each cell")

    means = stat_mean(stacked, axis=-1)
    residual_ss = np.sum((values - np.expand_dims(means, -1)) ** 2, axis=-1)
    error_ss = np.sum(residual_ss, axis=(-2, -1))
    grand = np.mean(means, axis=(-2, -1))
    row_means = np.mean(means, axis=-1)
    column_means = np.mean(means, axis=-2)

    row_ss = n_columns * n_cases * np.sum((row_means - np.expand_dims(grand, -1)) ** 2, axis=-1)
    column_ss = n_rows * n_cases * np.sum((column_means - np.expand_dims(grand, -1)) ** 2, axis=-1)
    interaction_terms = (
        means
        - np.expand_dims(row_means, -1)
        - np.expand_dims(column_means, -2)
        + np.expand_dims(np.expand_dims(grand, -1), -1)
    )
    interaction_ss = n_cases * np.sum(interaction_terms**2, axis=(-2, -1))

    df_error = n_rows * n_columns * (n_cases - 1)
    df_rows = (n_rows - 1, df_error)
    df_columns = (n_columns - 1, df_error)
    df_interaction = ((n_rows - 1) * (n_columns - 1), df_error)
    with np.errstate(divide="ignore", invalid="ignore"):
        row_f = (row_ss / df_rows[0]) / (error_ss / df_error)
        column_f = (column_ss / df_columns[0]) / (error_ss / df_error)
        interaction_f = (interaction_ss / df_interaction[0]) / (error_ss / df_error)
    return TwoWayAnovaResult(row_f, column_f, interaction_f, df_rows, df_columns, df_interaction)


__all__ = ["TwoWayAnovaResult", "anova2_cell"]
