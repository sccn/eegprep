"""NumPy implementations of EEGLAB-style statistics helpers."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats as scipy_stats


@dataclass(frozen=True)
class FDRResult:
    """False discovery rate threshold and mask."""

    threshold: np.ndarray | float
    mask: np.ndarray

    def __iter__(self) -> Iterator[np.ndarray | float]:
        yield self.threshold
        yield self.mask


@dataclass(frozen=True)
class ConcatenatedData:
    """Data concatenated across condition case axes."""

    data: np.ndarray
    lengths: np.ndarray
    grid_shape: tuple[int, int]

    def __iter__(self) -> Iterator[np.ndarray | tuple[int, int]]:
        yield self.data
        yield self.lengths
        yield self.grid_shape


@dataclass(frozen=True)
class TwoWayEffects:
    """Row, column, and interaction values for a two-way design."""

    rows: Any
    columns: Any
    interaction: Any

    def __iter__(self) -> Iterator[Any]:
        yield self.rows
        yield self.columns
        yield self.interaction


@dataclass(frozen=True)
class TwoWayAnovaResult:
    """Two-way ANOVA statistics and degrees of freedom."""

    rows: np.ndarray
    columns: np.ndarray
    interaction: np.ndarray
    df_rows: tuple[int, int]
    df_columns: tuple[int, int]
    df_interaction: tuple[int, int]

    def as_effects(self) -> TwoWayEffects:
        return TwoWayEffects(self.rows, self.columns, self.interaction)

    def df_effects(self) -> TwoWayEffects:
        return TwoWayEffects(self.df_rows, self.df_columns, self.df_interaction)


@dataclass(frozen=True)
class SurrogateDistribution:
    """Surrogate condition grids produced by permutation or bootstrap."""

    samples: tuple[tuple[tuple[np.ndarray, ...], ...], ...]

    def __iter__(self) -> Iterator[tuple[tuple[np.ndarray, ...], ...]]:
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)


@dataclass(frozen=True)
class StatcondResult:
    """Result returned by :func:`statcond`."""

    stat: Any
    df: Any
    pvalue: Any
    surrogate: Any
    method: str
    paired: bool
    ci: Any = None
    mask: Any = None

    def __iter__(self) -> Iterator[Any]:
        yield self.stat
        yield self.df
        yield self.pvalue
        yield self.surrogate


def fdr(pvals: Any, q: float | None = None, fdr_type: str = "parametric") -> FDRResult:
    """Compute Benjamini-Hochberg or Benjamini-Yekutieli FDR thresholds.

    Args:
        pvals: Numeric p-value array with values in the closed interval [0, 1].
        q: Desired false discovery rate. If omitted, an EEGLAB-style array of
            corrected thresholds is returned.
        fdr_type: ``"parametric"`` for Benjamini-Hochberg or
            ``"nonparametric"``/``"nonParametric"`` for Benjamini-Yekutieli.

    Returns:
        Threshold and boolean mask with the same shape as ``pvals``.
    """

    values = _as_numeric_array(pvals, "pvals", require_axis=False)
    if values.size == 0:
        return FDRResult(np.array([], dtype=float), np.array([], dtype=bool))
    if np.any((values < 0) | (values > 1)):
        raise ValueError("pvals must contain probabilities between 0 and 1")

    if q is None:
        threshold = np.ones(values.shape, dtype=float)
        thresholds = np.exp(np.linspace(np.log(0.1), np.log(0.000001), 1000))
        for current in thresholds:
            current_result = fdr(values, float(current), fdr_type=fdr_type)
            threshold[current_result.mask] = current
        return FDRResult(threshold, values <= threshold)

    q_value = float(q)
    if not 0 <= q_value <= 1:
        raise ValueError("q must be between 0 and 1")

    fdr_type_name = fdr_type.lower()
    if fdr_type_name not in {"parametric", "nonparametric"}:
        raise ValueError("fdr_type must be 'parametric' or 'nonparametric'")

    flat = np.sort(values.reshape(-1))
    count = flat.size
    indices = np.arange(1, count + 1, dtype=float)
    correction = 1.0 if fdr_type_name == "parametric" else float(np.sum(1.0 / indices))
    accepted = flat <= indices / count * q_value / correction
    threshold_value = float(flat[np.flatnonzero(accepted).max()]) if np.any(accepted) else 0.0
    return FDRResult(threshold_value, values <= threshold_value)


def stat_surrogate_pvals(distribution: Any, observed: Any, tail: str = "both") -> np.ndarray:
    """Compute empirical p-values against a surrogate distribution.

    Args:
        distribution: Surrogate statistic array whose last axis stores
            surrogate replications.
        observed: Observed statistic array matching ``distribution.shape[:-1]``.
        tail: ``"right"``/``"upper"``/``"one"``, ``"left"``/``"lower"``, or
            ``"both"``.
    """

    surrogates = _as_numeric_array(distribution, "distribution")
    observed_values = _as_numeric_array(observed, "observed", require_axis=False)
    if surrogates.ndim < 1:
        raise ValueError("distribution must have a surrogate axis")
    if observed_values.shape != surrogates.shape[:-1]:
        raise ValueError("observed shape must match distribution without its last axis")

    n_samples = surrogates.shape[-1]
    expanded = np.expand_dims(observed_values, axis=-1)
    p_right = np.sum(surrogates >= expanded, axis=-1) / n_samples
    tail_name = tail.lower()
    if tail_name in {"right", "upper", "one"}:
        return p_right

    p_left = 1 - p_right + np.sum(surrogates == expanded, axis=-1) / n_samples
    if tail_name in {"left", "lower"}:
        return p_left
    if tail_name == "both":
        return np.minimum(2 * np.minimum(p_right, p_left), 1.0)
    raise ValueError("tail must be 'right', 'upper', 'left', 'lower', 'one', or 'both'")


def stat_surrogate_ci(distribution: Any, alpha: float = 0.05, tail: str = "both") -> np.ndarray:
    """Compute surrogate confidence intervals along the last axis.

    Args:
        distribution: Surrogate statistic array whose last axis stores
            surrogate replications.
        alpha: Type-I error rate.
        tail: ``"upper"``, ``"lower"``, ``"one"``, or ``"both"``.
    """

    values = _as_numeric_array(distribution, "distribution")
    if values.ndim < 1:
        raise ValueError("distribution must have a surrogate axis")
    alpha_value = float(alpha)
    if not 0 <= alpha_value <= 1:
        raise ValueError("alpha must be between 0 and 1")

    sorted_values = np.sort(values, axis=-1)
    n_samples = sorted_values.shape[-1]
    ci_alpha = alpha_value / 2 if tail.lower() == "both" else alpha_value
    low = int(np.floor(ci_alpha * n_samples + 0.5))
    high = n_samples - low
    low_index = min(max(low, 0), n_samples - 1)
    high_index = min(max(high - 1, 0), n_samples - 1)
    mean_values = np.mean(sorted_values, axis=-1)

    tail_name = tail.lower()
    if tail_name == "upper":
        lower = mean_values
        upper = sorted_values[..., high_index]
    elif tail_name == "lower":
        lower = sorted_values[..., low_index]
        upper = mean_values
    elif tail_name in {"both", "one"}:
        lower = sorted_values[..., low_index]
        upper = sorted_values[..., high_index]
    else:
        raise ValueError("tail must be 'upper', 'lower', 'one', or 'both'")

    return np.stack((lower, upper), axis=0)


def concatdata(data: Any, *, axis: int = -1) -> ConcatenatedData:
    """Concatenate condition arrays along their case axis.

    Args:
        data: One- or two-dimensional sequence of condition arrays.
        axis: Axis in each condition array that stores cases.
    """

    grid = _condition_grid(data, axis=axis, min_cases=1)
    arrays = _flatten_grid(grid)
    feature_shape = arrays[0].shape[:-1]
    for index, array in enumerate(arrays):
        if array.shape[:-1] != feature_shape:
            raise ValueError(f"condition {index} has feature shape {array.shape[:-1]}, expected {feature_shape}")

    lengths = np.zeros(len(arrays) + 1, dtype=int)
    lengths[1:] = np.cumsum([array.shape[-1] for array in arrays])
    concatenated = np.concatenate(arrays, axis=-1)
    return ConcatenatedData(concatenated, lengths, (len(grid), len(grid[0])))


def corrcoef_cell(a: Any, b: Any | None = None, *, axis: int = -1) -> np.ndarray:
    """Compute pairwise correlations along a case axis."""

    first, second = _two_arrays(a, b, "corrcoef_cell", axis=axis)
    if first.shape != second.shape:
        raise ValueError("corrcoef_cell requires arrays with identical shapes")
    if first.shape[-1] < 2:
        raise ValueError("corrcoef_cell requires at least two cases")

    first_centered = first - np.mean(first, axis=-1, keepdims=True)
    second_centered = second - np.mean(second, axis=-1, keepdims=True)
    covariance = np.sum(first_centered * second_centered, axis=-1)
    first_power = np.sum(first_centered * first_centered, axis=-1)
    second_power = np.sum(second_centered * second_centered, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return covariance / np.sqrt(first_power * second_power)


def ttest_cell(a: Any, b: Any | None = None, *, axis: int = -1) -> tuple[np.ndarray, int]:
    """Compute paired t-statistics across the case axis."""

    first, second = _two_arrays(a, b, "ttest_cell", axis=axis)
    if first.shape != second.shape:
        raise ValueError("ttest_cell requires paired arrays with identical shapes")
    n_cases = first.shape[-1]
    if n_cases < 2:
        raise ValueError("ttest_cell requires at least two paired cases")

    difference = first - second
    mean_difference = _stat_mean(difference, axis=-1)
    sd_difference = _stat_std(difference, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = mean_difference / sd_difference * np.sqrt(n_cases)
    return t_values, n_cases - 1


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
    first, second = _two_arrays(a, b, "ttest2_cell", axis=axis)
    if first.shape[:-1] != second.shape[:-1]:
        raise ValueError("ttest2_cell requires matching feature shapes before the case axis")
    if first.shape[-1] < 2 or second.shape[-1] < 2:
        raise ValueError("ttest2_cell requires at least two cases in each group")

    variance_name = variance.lower()
    if variance_name not in {"homogenous", "inhomogenous"}:
        raise ValueError("variance must be 'homogenous' or 'inhomogenous'")

    first_n = first.shape[-1]
    second_n = second.shape[-1]
    first_mean = _stat_mean(first, axis=-1)
    second_mean = _stat_mean(second, axis=-1)
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

    first_sd = _stat_std(first, axis=-1)
    second_sd = _stat_std(second, axis=-1)
    pooled_sd = np.sqrt(((first_n - 1) * first_sd**2 + (second_n - 1) * second_sd**2) / (first_n + second_n - 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = (first_mean - second_mean) / pooled_sd / np.sqrt(1 / first_n + 1 / second_n)
    return t_values, first_n + second_n - 2


def anova1_cell(data: Any, *, axis: int = -1) -> tuple[np.ndarray, tuple[int, int]]:
    """Compute one-way unpaired ANOVA F-statistics across condition arrays."""

    arrays = _one_way_arrays(data, axis=axis, paired=False)
    if len(arrays) < 2:
        raise ValueError("anova1_cell requires at least two conditions")
    feature_shape = arrays[0].shape[:-1]
    for index, array in enumerate(arrays):
        if array.shape[:-1] != feature_shape:
            raise ValueError(f"condition {index} has feature shape {array.shape[:-1]}, expected {feature_shape}")
        if array.shape[-1] < 2:
            raise ValueError("anova1_cell requires at least two cases in each condition")

    counts = np.array([array.shape[-1] for array in arrays], dtype=float)
    means = np.stack([_stat_mean(array, axis=-1) for array in arrays], axis=-1)
    total_n = int(np.sum(counts))
    grand_mean = np.sum(means * counts, axis=-1) / total_n
    ss_between = np.sum(counts * (means - np.expand_dims(grand_mean, -1)) ** 2, axis=-1)
    ss_within = np.zeros(feature_shape, dtype=float)
    for array, mean in zip(arrays, np.moveaxis(means, -1, 0), strict=True):
        ss_within = ss_within + _sum_square_residuals(array, mean, axis=-1)

    df_between = len(arrays) - 1
    df_within = total_n - len(arrays)
    with np.errstate(divide="ignore", invalid="ignore"):
        f_values = (ss_between / df_between) / (ss_within / df_within)
    return f_values, (df_between, df_within)


def anova1rm_cell(data: Any, *, axis: int = -1) -> tuple[np.ndarray, tuple[int, int]]:
    """Compute one-way repeated-measures ANOVA F-statistics."""

    arrays = _one_way_arrays(data, axis=axis, paired=True)
    if len(arrays) < 2:
        raise ValueError("anova1rm_cell requires at least two conditions")
    _require_same_shapes(arrays, "anova1rm_cell")
    n_cases = arrays[0].shape[-1]
    if n_cases < 2:
        raise ValueError("anova1rm_cell requires at least two repeated cases")

    stacked = np.stack(arrays, axis=-2)
    values = _anova_values(stacked)
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


def anova2_cell(data: Any, *, axis: int = -1) -> TwoWayAnovaResult:
    """Compute balanced two-way unpaired ANOVA F-statistics."""

    stacked = _two_way_stack(data, axis=axis, name="anova2_cell")
    values = _anova_values(stacked)
    n_rows = stacked.shape[-3]
    n_columns = stacked.shape[-2]
    n_cases = stacked.shape[-1]
    if n_rows < 2 or n_columns < 2:
        raise ValueError("anova2_cell requires at least two rows and two columns")
    if n_cases < 2:
        raise ValueError("anova2_cell requires at least two cases in each cell")

    means = _stat_mean(stacked, axis=-1)
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


def anova2rm_cell(data: Any, *, axis: int = -1) -> TwoWayAnovaResult:
    """Compute two-way repeated-measures ANOVA F-statistics."""

    stacked = _two_way_stack(data, axis=axis, name="anova2rm_cell")
    values = _anova_values(stacked)
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


def surrogdistrib(
    data: Any,
    *,
    method: str = "perm",
    pairing: str = "on",
    naccu: int = 1,
    axis: int = -1,
    rng: np.random.Generator | int | None = None,
) -> SurrogateDistribution:
    """Build bootstrap or permutation surrogate condition grids.

    Args:
        data: One- or two-dimensional sequence of condition arrays.
        method: ``"perm"``/``"permutation"`` or ``"bootstrap"``.
        pairing: ``"on"`` to preserve case identity across conditions or
            ``"off"`` to resample from the pooled case axis.
        naccu: Number of surrogate grids to generate.
        axis: Axis in each condition array that stores cases.
        rng: Optional NumPy generator or seed for deterministic resampling.
    """

    method_name = _normalize_method(method)
    if method_name == "param":
        raise ValueError("surrogdistrib only supports permutation or bootstrap methods")
    pairing_name = pairing.lower()
    if pairing_name not in {"on", "off"}:
        raise ValueError("pairing must be 'on' or 'off'")
    count = int(naccu)
    if count < 1:
        raise ValueError("naccu must be at least 1")

    generator = _rng(rng)
    grid = _condition_grid(data, axis=axis, min_cases=1)
    samples = tuple(
        _resampled_grid(grid, bootstrap=method_name == "bootstrap", paired=pairing_name == "on", rng=generator)
        for _ in range(count)
    )
    return SurrogateDistribution(samples)


def statcond(
    data: Any,
    *,
    paired: str | bool = "auto",
    method: str = "param",
    mode: str | None = None,
    naccu: int = 200,
    variance: str = "homogenous",
    forceanova: bool = False,
    tail: str = "both",
    axis: int = -1,
    rng: np.random.Generator | int | None = None,
    alpha: float | None = None,
    surrog: Any = None,
    stats: Any = None,
    return_resampling_array: bool = False,
) -> StatcondResult | SurrogateDistribution:
    """Compare condition arrays using EEGLAB-style t-tests or ANOVAs.

    Args:
        data: One- or two-dimensional sequence of condition arrays. The case
            dimension is the last axis by default.
        paired: ``"auto"``, ``"on"``/``True``, or ``"off"``/``False``.
        method: ``"param"``, ``"perm"``, or ``"bootstrap"``.
        mode: Legacy alias for ``method``.
        naccu: Number of surrogate samples for nonparametric methods.
        variance: ``"homogenous"`` or ``"inhomogenous"`` for unpaired t-tests.
        forceanova: Use one-way ANOVA instead of a two-condition t-test.
        tail: Empirical-tail mode for supplied or computed surrogates.
        axis: Axis in each condition array that stores cases.
        rng: Optional NumPy generator or seed for deterministic resampling.
        alpha: Optional threshold for confidence intervals and masks; requires
            a nonparametric method or supplied surrogate statistics.
        surrog: Precomputed surrogate statistic array.
        stats: Observed statistic to pair with ``surrog``.
        return_resampling_array: Return surrogate condition grids instead of
            computing statistics.
    """

    method_name = _normalize_method(mode or method)
    grid = _condition_grid(data, axis=axis, min_cases=2)
    paired_flag = _paired_flag(grid, paired)
    if return_resampling_array:
        if method_name == "param":
            raise ValueError("return_resampling_array requires 'perm' or 'bootstrap'")
        return surrogdistrib(
            grid,
            method=method_name,
            pairing="on" if paired_flag else "off",
            naccu=naccu,
            rng=rng,
        )

    if surrog is not None:
        if stats is None:
            raise ValueError("stats must be supplied when surrog is supplied")
        observed_stat = stats
        observed_df = None
        surrogate_stat = surrog
        pvalue = _surrogate_pvalues(surrogate_stat, observed_stat, tail)
        ci = None
        mask = None
        if alpha is not None:
            ci = _surrogate_ci(surrogate_stat, alpha, _ci_tail(tail))
            mask = _effect_map(pvalue, lambda value: value < alpha)
        return StatcondResult(
            observed_stat, observed_df, pvalue, surrogate_stat, method_name, paired_flag, ci=ci, mask=mask
        )

    observed_stat, observed_df, statistic_kind = _compute_statistic(
        grid,
        paired=paired_flag,
        variance=variance,
        forceanova=forceanova,
    )
    surrogate_stat = None
    if method_name == "param":
        pvalue = _parametric_pvalues(observed_stat, observed_df, statistic_kind)
    else:
        surrogate_stat = _compute_surrogate_statistics(
            grid,
            paired=paired_flag,
            method=method_name,
            naccu=naccu,
            variance=variance,
            forceanova=forceanova,
            rng=rng,
        )
        empirical_tail = "one" if statistic_kind.startswith("f") else tail
        pvalue = _surrogate_pvalues(surrogate_stat, observed_stat, empirical_tail)

    ci = None
    mask = None
    if alpha is not None:
        if surrogate_stat is None:
            raise ValueError("alpha confidence intervals require a nonparametric method or supplied surrogates")
        empirical_tail = "one" if statistic_kind.startswith("f") else tail
        ci = _surrogate_ci(surrogate_stat, alpha, _ci_tail(empirical_tail))
        mask = _effect_map(pvalue, lambda value: value < alpha)

    return StatcondResult(
        observed_stat, observed_df, pvalue, surrogate_stat, method_name, paired_flag, ci=ci, mask=mask
    )


def teststat(seed: int = 0) -> dict[str, float]:
    """Run deterministic smoke checks for the EEGPrep statistics package."""

    rng = np.random.default_rng(seed)
    first = rng.normal(size=(3, 12))
    second = first + rng.normal(loc=0.25, scale=0.2, size=(3, 12))
    paired_result = statcond([first, second], paired="on", method="param")
    if not isinstance(paired_result, StatcondResult):
        raise AssertionError("paired statcond unexpectedly returned surrogate grids")
    t_values, df = ttest_cell(first, second)
    np.testing.assert_allclose(paired_result.stat, t_values)
    if paired_result.df != df:
        raise AssertionError("paired t-test degrees of freedom changed")

    groups = [rng.normal(loc=offset, size=(3, 10)) for offset in (0.0, 0.2, 0.5)]
    one_way = statcond(groups, paired="off", method="param")
    if not isinstance(one_way, StatcondResult):
        raise AssertionError("one-way statcond unexpectedly returned surrogate grids")
    direct_one_way, one_way_df = anova1_cell(groups)
    np.testing.assert_allclose(one_way.stat, direct_one_way)
    if one_way.df != one_way_df:
        raise AssertionError("one-way ANOVA degrees of freedom changed")

    grid = (
        (rng.normal(size=(2, 9)), rng.normal(loc=0.1, size=(2, 9))),
        (rng.normal(loc=0.2, size=(2, 9)), rng.normal(loc=0.4, size=(2, 9))),
    )
    two_way = statcond(grid, paired="on", method="param")
    if not isinstance(two_way, StatcondResult):
        raise AssertionError("two-way statcond unexpectedly returned surrogate grids")
    if not isinstance(two_way.stat, TwoWayEffects):
        raise AssertionError("two-way statcond did not return factor effects")

    return {
        "paired_t_mean": float(np.mean(paired_result.stat)),
        "one_way_f_mean": float(np.mean(one_way.stat)),
        "two_way_interaction_mean": float(np.mean(two_way.stat.interaction)),
    }


def _as_numeric_array(value: Any, name: str, *, axis: int = -1, require_axis: bool = True) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric")
    if require_axis and array.ndim == 0:
        raise ValueError(f"{name} must have at least one dimension")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if not require_axis:
        return array.astype(np.complex128 if np.iscomplexobj(array) else np.float64, copy=False)

    if not -array.ndim <= axis < array.ndim:
        raise ValueError(f"axis {axis} is out of bounds for array with {array.ndim} dimensions")
    normalized_axis = axis % array.ndim
    if normalized_axis != array.ndim - 1:
        array = np.moveaxis(array, normalized_axis, -1)
    return array.astype(np.complex128 if np.iscomplexobj(array) else np.float64, copy=False)


def _condition_grid(data: Any, *, axis: int = -1, min_cases: int = 1) -> tuple[tuple[np.ndarray, ...], ...]:
    raw_grid = _raw_condition_grid(data)
    grid: list[tuple[np.ndarray, ...]] = []
    for row_index, row in enumerate(raw_grid):
        converted_row = []
        for column_index, value in enumerate(row):
            array = _as_numeric_array(value, f"condition ({row_index}, {column_index})", axis=axis)
            if array.shape[-1] < min_cases:
                raise ValueError(
                    f"condition ({row_index}, {column_index}) must have at least {min_cases} cases on the case axis"
                )
            converted_row.append(array)
        grid.append(tuple(converted_row))

    if len(grid) > 1 and len(grid[0]) == 1:
        grid = [tuple(row[0] for row in grid)]
    return tuple(grid)


def _raw_condition_grid(data: Any) -> tuple[tuple[Any, ...], ...]:
    if isinstance(data, np.ndarray) and data.dtype == object:
        if data.ndim == 1:
            return (tuple(data.tolist()),)
        if data.ndim == 2:
            return tuple(tuple(row) for row in data.tolist())
        raise ValueError("object-array condition grids must be one- or two-dimensional")
    if not isinstance(data, Sequence) or isinstance(data, str | bytes):
        raise TypeError("data must be a sequence of condition arrays")
    if len(data) == 0:
        raise ValueError("data must contain at least one condition")

    first = data[0]
    if _looks_like_condition_row(first):
        rows = []
        expected_width: int | None = None
        for row in data:
            if not _looks_like_condition_row(row):
                raise ValueError("data rows must all be condition sequences")
            row_tuple = tuple(row)
            if len(row_tuple) == 0:
                raise ValueError("condition rows must not be empty")
            if expected_width is None:
                expected_width = len(row_tuple)
            elif len(row_tuple) != expected_width:
                raise ValueError("all condition rows must have the same length")
            rows.append(row_tuple)
        return tuple(rows)
    return (tuple(data),)


def _looks_like_condition_row(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, np.ndarray | str | bytes)


def _flatten_grid(grid: tuple[tuple[np.ndarray, ...], ...]) -> list[np.ndarray]:
    return [array for row in grid for array in row]


def _two_arrays(a: Any, b: Any | None, caller: str, *, axis: int) -> tuple[np.ndarray, np.ndarray]:
    if b is None:
        grid = _condition_grid(a, axis=axis, min_cases=1)
        arrays = _flatten_grid(grid)
        if len(arrays) != 2:
            raise ValueError(f"{caller} requires exactly two arrays")
        return arrays[0], arrays[1]
    return (
        _as_numeric_array(a, "a", axis=axis),
        _as_numeric_array(b, "b", axis=axis),
    )


def _one_way_arrays(data: Any, *, axis: int, paired: bool) -> list[np.ndarray]:
    grid = _condition_grid(data, axis=axis, min_cases=2 if paired else 1)
    if len(grid) != 1:
        raise ValueError("one-way ANOVA helpers require a one-dimensional condition sequence")
    return list(grid[0])


def _require_same_shapes(arrays: Sequence[np.ndarray], name: str) -> None:
    expected_shape = arrays[0].shape
    for index, array in enumerate(arrays):
        if array.shape != expected_shape:
            raise ValueError(f"{name} requires identical shapes; condition {index} has shape {array.shape}")


def _two_way_stack(data: Any, *, axis: int, name: str) -> np.ndarray:
    grid = _condition_grid(data, axis=axis, min_cases=2)
    if len(grid) < 1 or len(grid[0]) < 1:
        raise ValueError(f"{name} requires a non-empty condition grid")
    expected_shape = grid[0][0].shape
    for row_index, row in enumerate(grid):
        for column_index, array in enumerate(row):
            if array.shape != expected_shape:
                raise ValueError(
                    f"{name} requires balanced cell shapes; condition ({row_index}, {column_index}) has "
                    f"shape {array.shape}, expected {expected_shape}"
                )
    return np.stack([np.stack(row, axis=-2) for row in grid], axis=-3)


def _stat_mean(array: np.ndarray, *, axis: int) -> np.ndarray:
    result = np.mean(array, axis=axis)
    if np.iscomplexobj(array):
        return np.abs(result)
    return result


def _stat_std(array: np.ndarray, *, axis: int) -> np.ndarray:
    if np.iscomplexobj(array):
        return np.std(np.abs(array), axis=axis, ddof=1)
    return np.sqrt(np.sum((array - np.mean(array, axis=axis, keepdims=True)) ** 2, axis=axis) / (array.shape[axis] - 1))


def _anova_values(array: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(array):
        return np.abs(array)
    return array


def _sum_square_residuals(array: np.ndarray, mean: np.ndarray, *, axis: int) -> np.ndarray:
    values = _anova_values(array)
    return np.sum((values - np.expand_dims(mean, axis=axis)) ** 2, axis=axis)


def _normalize_method(method: str) -> str:
    method_name = method.lower()
    if method_name == "parametric":
        return "param"
    if method_name == "permutation":
        return "perm"
    if method_name in {"param", "perm", "bootstrap"}:
        return method_name
    raise ValueError("method must be 'param', 'perm', 'permutation', 'parametric', or 'bootstrap'")


def _paired_flag(grid: tuple[tuple[np.ndarray, ...], ...], paired: str | bool) -> bool:
    counts = [array.shape[-1] for array in _flatten_grid(grid)]
    can_pair = len(set(counts)) == 1
    if paired == "auto":
        return can_pair
    if isinstance(paired, str):
        paired_name = paired.lower()
        if paired_name not in {"on", "off"}:
            raise ValueError("paired must be 'auto', 'on', 'off', True, or False")
        requested = paired_name == "on"
    else:
        requested = bool(paired)
    if requested and not can_pair:
        raise ValueError("paired statistics require the same number of cases in every condition")
    return requested


def _compute_statistic(
    grid: tuple[tuple[np.ndarray, ...], ...],
    *,
    paired: bool,
    variance: str,
    forceanova: bool,
) -> tuple[Any, Any, str]:
    rows = len(grid)
    columns = len(grid[0])
    if rows == 1:
        if columns == 2 and not forceanova:
            if paired:
                stat, df = ttest_cell(grid[0][0], grid[0][1])
            else:
                stat, df = ttest2_cell(grid[0][0], grid[0][1], variance=variance)
            return stat, df, "t"
        if paired:
            stat, df = anova1rm_cell(grid[0])
        else:
            stat, df = anova1_cell(grid[0])
        return stat, df, "f_one_way"

    anova = anova2rm_cell(grid) if paired else anova2_cell(grid)
    return anova.as_effects(), anova.df_effects(), "f_two_way"


def _parametric_pvalues(stat: Any, df: Any, statistic_kind: str) -> Any:
    if statistic_kind == "t":
        return 2 * scipy_stats.t.sf(np.abs(stat), df)
    if isinstance(stat, TwoWayEffects):
        return TwoWayEffects(
            scipy_stats.f.sf(stat.rows, df.rows[0], df.rows[1]),
            scipy_stats.f.sf(stat.columns, df.columns[0], df.columns[1]),
            scipy_stats.f.sf(stat.interaction, df.interaction[0], df.interaction[1]),
        )
    return scipy_stats.f.sf(stat, df[0], df[1])


def _compute_surrogate_statistics(
    grid: tuple[tuple[np.ndarray, ...], ...],
    *,
    paired: bool,
    method: str,
    naccu: int,
    variance: str,
    forceanova: bool,
    rng: np.random.Generator | int | None,
) -> Any:
    distribution = surrogdistrib(
        grid,
        method=method,
        pairing="on" if paired else "off",
        naccu=naccu,
        rng=rng,
    )
    stats = []
    for sample in distribution:
        sample_stat, _sample_df, _kind = _compute_statistic(
            sample,
            paired=paired,
            variance=variance,
            forceanova=forceanova,
        )
        stats.append(sample_stat)
    return _stack_effects(stats)


def _stack_effects(values: Sequence[Any]) -> Any:
    first = values[0]
    if isinstance(first, TwoWayEffects):
        return TwoWayEffects(
            np.stack([value.rows for value in values], axis=-1),
            np.stack([value.columns for value in values], axis=-1),
            np.stack([value.interaction for value in values], axis=-1),
        )
    return np.stack(values, axis=-1)


def _surrogate_pvalues(surrogate: Any, observed: Any, tail: str) -> Any:
    if isinstance(surrogate, TwoWayEffects):
        return TwoWayEffects(
            stat_surrogate_pvals(surrogate.rows, observed.rows, tail),
            stat_surrogate_pvals(surrogate.columns, observed.columns, tail),
            stat_surrogate_pvals(surrogate.interaction, observed.interaction, tail),
        )
    return stat_surrogate_pvals(surrogate, observed, tail)


def _surrogate_ci(surrogate: Any, alpha: float, tail: str) -> Any:
    if isinstance(surrogate, TwoWayEffects):
        return TwoWayEffects(
            stat_surrogate_ci(surrogate.rows, alpha, tail),
            stat_surrogate_ci(surrogate.columns, alpha, tail),
            stat_surrogate_ci(surrogate.interaction, alpha, tail),
        )
    return stat_surrogate_ci(surrogate, alpha, tail)


def _ci_tail(tail: str) -> str:
    tail_name = tail.lower()
    if tail_name == "right":
        return "upper"
    if tail_name == "left":
        return "lower"
    return tail_name


def _effect_map(effect: Any, func: Any) -> Any:
    if isinstance(effect, TwoWayEffects):
        return TwoWayEffects(func(effect.rows), func(effect.columns), func(effect.interaction))
    return func(effect)


def _rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _resampled_grid(
    grid: tuple[tuple[np.ndarray, ...], ...],
    *,
    bootstrap: bool,
    paired: bool,
    rng: np.random.Generator,
) -> tuple[tuple[np.ndarray, ...], ...]:
    arrays = _flatten_grid(grid)
    feature_shape = arrays[0].shape[:-1]
    for index, array in enumerate(arrays):
        if array.shape[:-1] != feature_shape:
            raise ValueError(f"condition {index} has feature shape {array.shape[:-1]}, expected {feature_shape}")
    counts = [array.shape[-1] for array in arrays]

    if paired:
        if len(set(counts)) != 1:
            raise ValueError("paired surrogate resampling requires equal case counts")
        resampled = _paired_resample(arrays, bootstrap=bootstrap, rng=rng)
    else:
        resampled = _unpaired_resample(arrays, counts, bootstrap=bootstrap, rng=rng)

    iterator = iter(resampled)
    return tuple(tuple(next(iterator) for _column in row) for row in grid)


def _paired_resample(
    arrays: Sequence[np.ndarray],
    *,
    bootstrap: bool,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    n_conditions = len(arrays)
    n_cases = arrays[0].shape[-1]
    output = [np.empty_like(array) for array in arrays]
    for case_index in range(n_cases):
        source_conditions = (
            rng.integers(0, n_conditions, size=n_conditions) if bootstrap else rng.permutation(n_conditions)
        )
        for target_condition, source_condition in enumerate(source_conditions):
            output[target_condition][..., case_index] = arrays[int(source_condition)][..., case_index]
    return output


def _unpaired_resample(
    arrays: Sequence[np.ndarray],
    counts: Sequence[int],
    *,
    bootstrap: bool,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    pooled = np.concatenate(arrays, axis=-1)
    total_cases = pooled.shape[-1]
    if bootstrap:
        indices = rng.integers(0, total_cases, size=total_cases)
    else:
        indices = rng.permutation(total_cases)

    output = []
    start = 0
    for count in counts:
        stop = start + count
        output.append(np.take(pooled, indices[start:stop], axis=-1))
        start = stop
    return output
