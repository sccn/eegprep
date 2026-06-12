"""Two-way repeated-measures ANOVA helper."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.statistics._shared import TwoWayAnovaResult, anova_values, two_way_stack


def anova2rm_cell(data: Any, *, axis: int = -1) -> TwoWayAnovaResult:
    """Compute two-way repeated-measures ANOVA F-statistics."""

    stacked = two_way_stack(data, axis=axis, name="anova2rm_cell")
    values = anova_values(stacked)
    n_rows = stacked.shape[-3]
    n_columns = stacked.shape[-2]
    n_subjects = stacked.shape[-1]
    if n_rows < 2 or n_columns < 2:
        raise ValueError("anova2rm_cell requires at least two rows and two columns")
    if n_subjects < 2:
        raise ValueError("anova2rm_cell requires at least two repeated cases")

    ab_sums = np.sum(values, axis=-1)
    row_subject_sums = np.sum(values, axis=-2)
    column_subject_sums = np.sum(values, axis=-3)
    row_sums = np.sum(ab_sums, axis=-1)
    column_sums = np.sum(ab_sums, axis=-2)
    subject_sums = np.sum(row_subject_sums, axis=-2)
    total_sum = np.sum(row_sums, axis=-1)

    df_rows_num = n_rows - 1
    df_columns_num = n_columns - 1
    df_interaction_num = (n_rows - 1) * (n_columns - 1)
    df_row_subject = (n_rows - 1) * (n_subjects - 1)
    df_column_subject = (n_columns - 1) * (n_subjects - 1)
    df_interaction_subject = (n_rows - 1) * (n_columns - 1) * (n_subjects - 1)

    expected_rows = np.sum(row_sums**2, axis=-1) / (n_columns * n_subjects)
    expected_columns = np.sum(column_sums**2, axis=-1) / (n_rows * n_subjects)
    expected_ab = np.sum(ab_sums**2, axis=(-2, -1)) / n_subjects
    expected_subjects = np.sum(subject_sums**2, axis=-1) / (n_rows * n_columns)
    expected_row_subject = np.sum(row_subject_sums**2, axis=(-2, -1)) / n_columns
    expected_column_subject = np.sum(column_subject_sums**2, axis=(-2, -1)) / n_rows
    expected_y = np.sum(values**2, axis=(-3, -2, -1))
    expected_total = total_sum**2 / (n_rows * n_columns * n_subjects)

    ss_rows = expected_rows - expected_total
    ss_columns = expected_columns - expected_total
    ss_interaction = expected_ab - expected_rows - expected_columns + expected_total
    ss_subjects = expected_subjects - expected_total
    ss_row_subject = expected_row_subject - expected_rows - expected_subjects + expected_total
    ss_column_subject = expected_column_subject - expected_columns - expected_subjects + expected_total
    ss_interaction_subject = (
        expected_y
        - expected_ab
        - expected_row_subject
        - expected_column_subject
        + expected_rows
        + expected_columns
        + expected_subjects
        - expected_total
    )
    del ss_subjects

    with np.errstate(divide="ignore", invalid="ignore"):
        row_f = (ss_rows / df_rows_num) / (ss_row_subject / df_row_subject)
        column_f = (ss_columns / df_columns_num) / (ss_column_subject / df_column_subject)
        interaction_f = (ss_interaction / df_interaction_num) / (ss_interaction_subject / df_interaction_subject)
    return TwoWayAnovaResult(
        row_f,
        column_f,
        interaction_f,
        (df_rows_num, df_row_subject),
        (df_columns_num, df_column_subject),
        (df_interaction_num, df_interaction_subject),
    )


__all__ = ["TwoWayAnovaResult", "anova2rm_cell"]
