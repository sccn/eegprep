"""Faithful reference contracts and additional Python ``statcond`` checks."""

from __future__ import annotations

from collections.abc import Sequence
import warnings

import numpy as np
import pytest
from scipy.io import loadmat
from scipy import stats as scipy_stats

from eegprep.functions.statistics import StatcondResult, SurrogateDistribution, TwoWayEffects, statcond
from eegprep.functions.statistics._matlab_rand import reset_matlab_rng
from tests.eeglab_tests import eeglab_test


STATCOND_CLASS = "unittesting_statistics/statcond/statcondTest.m"
STATCOND_REGRESSION = "regression_tests/t_statcond.m"
STATCOND_WRAPPER = "unittesting_statistics/statcond/statistics_statcond_wrapperTest.m"


@pytest.fixture(scope="module")
def statcond_regression_data(eeglab_suite_root):
    # mat_dtype=True restores MATLAB classes; never squeeze cells or dimensions.
    return loadmat(eeglab_suite_root / "regression_tests/t_statcond.mat", mat_dtype=True)


@pytest.fixture
def seeded_statcond(eeglab_backend, request):
    if request.config.getoption("--eeglab-backend") == "matlab":
        request.getfixturevalue("eeglab_matlab_engine").rng("default", nargout=0)
    else:
        reset_matlab_rng()
    return eeglab_backend


def _assert_matlab_value(actual, expected, *, atol, rtol):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    if expected.dtype == object:
        for index in np.ndindex(expected.shape):
            _assert_matlab_value(actual[index], expected[index], atol=atol, rtol=rtol)
    else:
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


# Cases 1, 2, and 7-14 are double t-tests with stored zero tolerances.
# The Python backend derives a tolerance from float64 eps and the t-test;
# the MATLAB backend still uses the stored zero tolerance.
_PYTHON_TTEST_REGRESSION_CASES = frozenset({1, 2, *range(7, 15)})


def _ttest_accumulation_length(arguments: list) -> int:
    """Return the number of cases reduced into the t statistic.

    A paired test accumulates one difference per case, so ``n`` is that case
    count. An unpaired test accumulates both samples, because each case enters
    a mean and the pooled variance, so ``n`` is the sum of the two case counts.
    """

    conditions = [np.asarray(cell) for cell in np.asarray(arguments[0]).ravel()]
    paired = next(arguments[index + 1] for index in range(1, len(arguments) - 1, 2) if arguments[index] == "paired")
    case_counts = [condition.shape[-1] for condition in conditions]
    if paired == "on":
        return case_counts[0]
    return sum(case_counts)


def _relative_p_mask(pvalue: np.ndarray) -> np.ndarray:
    """Return where a relative p tolerance is defined.

    Zero and subnormal reference probabilities are excluded. Dividing by zero
    makes the condition number infinite, and one such element would loosen the
    whole array.
    """

    magnitude = np.abs(np.asarray(pvalue, dtype=np.float64))
    return np.isfinite(magnitude) & (magnitude >= np.finfo(np.float64).tiny)


def _two_tailed_t_condition(stat: np.ndarray, df: np.ndarray, pvalue: np.ndarray) -> float:
    """Return ``max |t| * 2 * pdf(|t|, df) / |p|`` where ``p`` is normal.

    The two-tailed p-value is ``2 * sf(|t|, df)``, so this factor is
    ``|d log p / d log t|``. The maximum is one bound for the whole array: a
    per-element bound is tighter than the last-bit ``tcdf`` difference on
    elements whose ``t`` already matches. On the ten double fixtures, ``t``
    matches bit for bit in nine, so the bound covers that ``tcdf`` difference.
    Probabilities that are zero or subnormal do not enter the maximum.
    """

    t_abs = np.abs(np.asarray(stat, dtype=np.float64))
    degrees = np.asarray(df, dtype=np.float64)
    probability = np.abs(np.asarray(pvalue, dtype=np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        numerator = np.asarray(t_abs * (2.0 * scipy_stats.t.pdf(t_abs, degrees)), dtype=np.float64)
    numerator, probability = np.broadcast_arrays(numerator, probability)
    usable = _relative_p_mask(probability)
    quotient = np.zeros(numerator.shape, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(numerator, probability, out=quotient, where=usable)
    if not np.any(usable):
        return 0.0
    bound = float(np.max(quotient[usable]))
    if not np.isfinite(bound):
        raise AssertionError("two-tailed t p-value tolerance is not finite")
    return bound


def _assert_python_ttest_regression(actual, reference, arguments) -> None:
    """Compare a double t-test at ``n * eps`` and the p-value condition bound.

    ``eps`` is float64 machine epsilon. The statistic uses ``n * eps``. The
    p-value uses that tolerance times :func:`_two_tailed_t_condition`. Degrees
    of freedom and the empty parametric surrogate stay exact, with absolute and
    relative tolerance zero.
    """

    assert actual.shape == reference.shape
    assert actual.dtype == object and reference.dtype == object
    stat_rtol = _ttest_accumulation_length(arguments) * float(np.finfo(np.float64).eps)
    _assert_matlab_value(actual[0, 0], reference[0, 0], atol=0.0, rtol=stat_rtol)
    _assert_matlab_value(actual[0, 1], reference[0, 1], atol=0.0, rtol=0.0)
    assert actual[0, 2].shape == reference[0, 2].shape
    assert actual[0, 2].dtype == reference[0, 2].dtype
    _assert_derived_p(actual[0, 2], reference[0, 0], reference[0, 1], reference[0, 2], stat_rtol=stat_rtol)
    if reference.shape[1] == 4:
        _assert_matlab_value(actual[0, 3], reference[0, 3], atol=0.0, rtol=0.0)


# Cases 3-6 and 15-30 are parametric ANOVA fixtures stored at zero tolerance.
# The Python backend derives a tolerance from float64 roundoff, the sum-of-squares
# length, and repeated-measures cancellation. The MATLAB backend keeps the
# stored zero tolerance.
_PYTHON_ANOVA_REGRESSION_CASES = frozenset({*range(3, 7), *range(15, 31)})
_FLOAT64_UNIT_ROUNDOFF = np.float64(2.0**-53)
# Measured betainc evaluation allowance, in multiples of u = 2^{-53}.
# Not derived from the sum-of-squares operation count. One constant covers
# the full tail. Box MATLAB R2025b vs mpmath reaches 1966.5u, and SciPy
# 1.15.3 reaches 69.9u down to p = 1e-300. Their sum is 2036u. 1024 does
# not cover it; 2048 does.
_C_BETA = np.float64(2048)


def _regression_option(arguments: list, name: str):
    return next(arguments[index + 1] for index in range(1, len(arguments) - 1, 2) if arguments[index] == name)


def _regression_grid(arguments: list) -> tuple[tuple[np.ndarray, ...], ...]:
    raw = np.asarray(arguments[0])
    return tuple(
        tuple(np.asarray(raw[row, column], dtype=np.float64) for column in range(raw.shape[1]))
        for row in range(raw.shape[0])
    )


def _observation_count(grid: tuple[tuple[np.ndarray, ...], ...]) -> int:
    """Return how many observations enter one feature's sum of squares."""

    if len(grid) == 1:
        return sum(array.shape[-1] for array in grid[0])
    return len(grid) * len(grid[0]) * grid[0][0].shape[-1]


def _cancellation_factor(parts: Sequence[np.ndarray], total: np.ndarray) -> np.ndarray:
    """Return ``max(1, Σ|term| / |sum|)`` for a bracket of sums of squares.

    A sum of positive terms has factor 1. A repeated-measures difference such as
    ``expA - expT`` has a larger factor when those terms nearly cancel.
    """

    magnitude = np.zeros(np.shape(total), dtype=np.float64)
    for part in parts:
        magnitude = magnitude + np.abs(np.asarray(part, dtype=np.float64))
    result = np.abs(np.asarray(total, dtype=np.float64))
    factor = np.ones(result.shape, dtype=np.float64)
    np.divide(magnitude, result, out=factor, where=result > 0)
    return np.maximum(factor, 1.0)


def _repeated_measures_factor(
    numerator_parts: Sequence[np.ndarray],
    numerator: np.ndarray,
    denominator_parts: Sequence[np.ndarray],
    denominator: np.ndarray,
) -> np.ndarray:
    return _cancellation_factor(numerator_parts, numerator) + _cancellation_factor(denominator_parts, denominator)


def _one_way_repeated_tolerances(grid: tuple[tuple[np.ndarray, ...], ...]) -> list:
    stacked = np.stack([np.asarray(array, dtype=np.float64) for array in grid[0]], axis=-2)
    n_conditions = stacked.shape[-2]
    n_cases = stacked.shape[-1]
    n_observations = n_conditions * n_cases
    condition_sums = np.sum(stacked, axis=-1)
    subject_sums = np.sum(stacked, axis=-2)
    total = np.sum(condition_sums, axis=-1)
    expected_condition = np.sum(condition_sums**2, axis=-1) / n_cases
    expected_subject = np.sum(subject_sums**2, axis=-1) / n_conditions
    expected_within = np.sum(stacked**2, axis=(-2, -1))
    expected_total = total**2 / n_observations
    numerator = expected_condition - expected_total
    denominator = expected_within - expected_condition - expected_subject + expected_total
    factor = _repeated_measures_factor(
        (expected_condition, expected_total),
        numerator,
        (expected_within, expected_condition, expected_subject, expected_total),
        denominator,
    )
    return [2.0 * n_observations * _FLOAT64_UNIT_ROUNDOFF * factor]


def _two_way_repeated_tolerances(grid: tuple[tuple[np.ndarray, ...], ...]) -> list:
    values = np.stack(
        [np.stack([np.asarray(array, dtype=np.float64) for array in row], axis=-2) for row in grid],
        axis=-3,
    )
    n_rows, n_columns, n_subjects = values.shape[-3:]
    n_observations = n_rows * n_columns * n_subjects
    cell_sums = np.sum(values, axis=-1)
    row_subject = np.sum(values, axis=-2)
    column_subject = np.sum(values, axis=-3)
    row_sums = np.sum(cell_sums, axis=-1)
    column_sums = np.sum(cell_sums, axis=-2)
    subject_sums = np.sum(row_subject, axis=-2)
    total = np.sum(row_sums, axis=-1)
    expected_rows = np.sum(row_sums**2, axis=-1) / (n_columns * n_subjects)
    expected_columns = np.sum(column_sums**2, axis=-1) / (n_rows * n_subjects)
    expected_cells = np.sum(cell_sums**2, axis=(-2, -1)) / n_subjects
    expected_subjects = np.sum(subject_sums**2, axis=-1) / (n_rows * n_columns)
    expected_row_subject = np.sum(row_subject**2, axis=(-2, -1)) / n_columns
    expected_column_subject = np.sum(column_subject**2, axis=(-2, -1)) / n_rows
    expected_values = np.sum(values**2, axis=(-3, -2, -1))
    expected_total = total**2 / n_observations
    row_ss = expected_rows - expected_total
    column_ss = expected_columns - expected_total
    interaction_ss = expected_cells - expected_rows - expected_columns + expected_total
    row_error = expected_row_subject - expected_rows - expected_subjects + expected_total
    column_error = expected_column_subject - expected_columns - expected_subjects + expected_total
    interaction_error = (
        expected_values
        - expected_cells
        - expected_row_subject
        - expected_column_subject
        + expected_rows
        + expected_columns
        + expected_subjects
        - expected_total
    )
    scale = 2.0 * n_observations * _FLOAT64_UNIT_ROUNDOFF
    return [
        scale
        * _repeated_measures_factor(
            (expected_rows, expected_total),
            row_ss,
            (expected_row_subject, expected_rows, expected_subjects, expected_total),
            row_error,
        ),
        scale
        * _repeated_measures_factor(
            (expected_columns, expected_total),
            column_ss,
            (expected_column_subject, expected_columns, expected_subjects, expected_total),
            column_error,
        ),
        scale
        * _repeated_measures_factor(
            (expected_cells, expected_rows, expected_columns, expected_total),
            interaction_ss,
            (
                expected_values,
                expected_cells,
                expected_row_subject,
                expected_column_subject,
                expected_rows,
                expected_columns,
                expected_subjects,
                expected_total,
            ),
            interaction_error,
        ),
    ]


def _anova_f_tolerances(arguments: list) -> list:
    """Return one relative F tolerance per MATLAB effect.

    Each sum of squares over ``N`` observations uses the t-test accumulation
    rule ``N * eps = 2 * N * u``, with ``u = 2^{-53}``. An F ratio adds the
    numerator and denominator. Repeated-measures brackets multiply that by
    ``Σ|term| / |sum|`` per element, because ``anova1rm_cell.m`` and
    ``anova2rm_cell.m`` subtract those terms. The unpaired factor is 1 because
    EEGLAB's ``anova1_cell`` delegates to the two-pass ``anova2_cell``, and the
    tolerance is ``4 * N * u``. Paired two-way effects are row, column,
    interaction. Unpaired two-way effects follow ``anova2_cell`` and are
    column, row, interaction.
    """

    grid = _regression_grid(arguments)
    paired = _regression_option(arguments, "paired") == "on"
    if len(grid) == 1:
        if paired:
            return _one_way_repeated_tolerances(grid)
        return [4.0 * _observation_count(grid) * _FLOAT64_UNIT_ROUNDOFF]
    if paired:
        return _two_way_repeated_tolerances(grid)
    tolerance = 4.0 * _observation_count(grid) * _FLOAT64_UNIT_ROUNDOFF
    return [tolerance, tolerance, tolerance]


def _f_tail_condition(stat: np.ndarray, df: np.ndarray, pvalue: np.ndarray) -> np.ndarray:
    """Return ``|F| * pdf(F; df1, df2) / |p|`` per element where ``p`` is normal.

    The parametric tail is the F survival function, so this factor is
    ``|d log p / d log F|``. Zero and subnormal probabilities stay 0 here and
    are compared exactly by the caller.
    """

    f_values = np.abs(np.asarray(stat, dtype=np.float64))
    degrees = np.asarray(df, dtype=np.float64).reshape(-1)
    probability = np.abs(np.asarray(pvalue, dtype=np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        numerator = f_values * scipy_stats.f.pdf(f_values, float(degrees[0]), float(degrees[1]))
    numerator, probability = np.broadcast_arrays(np.asarray(numerator, dtype=np.float64), probability)
    usable = _relative_p_mask(probability)
    quotient = np.zeros(numerator.shape, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(numerator, probability, out=quotient, where=usable)
    if np.any(usable) and not np.all(np.isfinite(quotient[usable])):
        raise AssertionError("F p-value tolerance is not finite")
    return quotient


def _align_tolerance(tolerance, shape: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(tolerance, dtype=np.float64)
    if values.shape == shape:
        return values
    if values.ndim == 0 or values.size == 1:
        return np.broadcast_to(values, shape)
    if values.size == int(np.prod(shape)):
        return np.reshape(values, shape, order="F")
    raise AssertionError(f"tolerance shape {values.shape} does not match {shape}")


def _assert_within_limit(actual, expected, limit) -> None:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    delta = np.abs(np.asarray(actual, dtype=np.float64) - np.asarray(expected, dtype=np.float64))
    bound = _align_tolerance(limit, expected.shape)
    if not np.all(np.isfinite(bound)):
        raise AssertionError("tolerance is not finite")
    if np.any(delta > bound):
        ratio = np.divide(delta, bound, out=np.full(delta.shape, np.inf), where=bound > 0)
        raise AssertionError(f"max measured/limit ratio {float(np.max(ratio))}")


def _assert_relative_p(actual, reference_p, bound: float) -> None:
    actual = np.asarray(actual)
    reference_p = np.asarray(reference_p)
    assert actual.shape == reference_p.shape
    if not np.isfinite(bound):
        raise AssertionError("p-value tolerance is not finite")
    usable = _relative_p_mask(reference_p)
    if np.any(~usable):
        np.testing.assert_array_equal(actual[~usable], reference_p[~usable])
    if np.any(usable):
        np.testing.assert_allclose(actual[usable], reference_p[usable], rtol=bound, atol=0.0)


def _assert_derived_p(actual, reference_stat, reference_df, reference_p, *, stat_rtol: float) -> None:
    """Compare p at ``stat_rtol`` times the two-tailed condition number.

    Reference probabilities that are zero or subnormal must match exactly.
    A non-finite bound is an assertion failure rather than an infinite
    tolerance.
    """

    bound = stat_rtol * _two_tailed_t_condition(reference_stat, reference_df, reference_p)
    _assert_relative_p(actual, reference_p, bound)


def _assert_anova_p(actual, reference_stat, reference_df, reference_p, f_tol) -> None:
    """Compare p at the derived F propagation plus the measured betainc term.

    For each normal reference probability the absolute limit is
    ``F_tol * |F| * pdf(F; df1, df2) + _C_BETA * u * |p|``. The first term is
    the per-element propagation of the F tolerance through
    ``|d log p / d log F|``. ``_C_BETA`` is a measured allowance for ``betainc``
    evaluation error between MATLAB and SciPy. Zero and subnormal reference
    probabilities must match exactly.
    """

    actual = np.asarray(actual)
    reference_p = np.asarray(reference_p)
    assert actual.shape == reference_p.shape
    assert actual.dtype == reference_p.dtype
    usable = _relative_p_mask(reference_p)
    if np.any(~usable):
        np.testing.assert_array_equal(actual[~usable], reference_p[~usable])
    if not np.any(usable):
        return
    condition = _f_tail_condition(reference_stat, reference_df, reference_p)
    probability = np.abs(np.asarray(reference_p, dtype=np.float64))
    derived = _align_tolerance(f_tol, probability.shape) * _align_tolerance(condition, probability.shape) * probability
    measured = _C_BETA * _FLOAT64_UNIT_ROUNDOFF * probability
    limit = np.array(derived + measured, copy=True)
    limit[~usable] = 0.0
    if np.any(usable & ~np.isfinite(limit)):
        raise AssertionError("F p-value tolerance is not finite")
    delta = np.abs(np.asarray(actual, dtype=np.float64) - np.asarray(reference_p, dtype=np.float64))
    if np.any(usable & (delta > limit)):
        ratio = np.divide(delta, limit, out=np.zeros(delta.shape, dtype=np.float64), where=usable & (limit > 0))
        raise AssertionError(f"max measured/limit ratio {float(np.max(ratio))}")


def _effect_leaves(value) -> list:
    array = np.asarray(value)
    if array.dtype == object:
        return [array[index] for index in np.ndindex(array.shape)]
    return [array]


def _assert_python_anova_regression(actual, reference, arguments) -> None:
    """Compare parametric ANOVA output at the derived F and p tolerances.

    Degrees of freedom and the empty parametric surrogate stay exact.
    """

    assert actual.shape == reference.shape
    assert actual.dtype == object and reference.dtype == object
    tolerances = _anova_f_tolerances(arguments)
    statistics = _effect_leaves(actual[0, 0])
    reference_statistics = _effect_leaves(reference[0, 0])
    assert len(statistics) == len(tolerances) == len(reference_statistics)
    for got, expected, rtol in zip(statistics, reference_statistics, tolerances, strict=True):
        magnitude = np.abs(np.asarray(expected, dtype=np.float64))
        _assert_within_limit(got, expected, _align_tolerance(rtol, magnitude.shape) * magnitude)
    _assert_matlab_value(actual[0, 1], reference[0, 1], atol=0.0, rtol=0.0)
    probabilities = _effect_leaves(actual[0, 2])
    reference_probabilities = _effect_leaves(reference[0, 2])
    reference_degrees = _effect_leaves(reference[0, 1])
    assert len(probabilities) == len(tolerances)
    for got, expected_f, expected_df, expected_p, rtol in zip(
        probabilities,
        reference_statistics,
        reference_degrees,
        reference_probabilities,
        tolerances,
        strict=True,
    ):
        _assert_anova_p(got, expected_f, expected_df, expected_p, rtol)
    if reference.shape[1] == 4:
        _assert_matlab_value(actual[0, 3], reference[0, 3], atol=0.0, rtol=0.0)


def _statcond_regression_test(number):
    @eeglab_test(STATCOND_REGRESSION, f"test_{number}")
    def test(seeded_statcond, statcond_regression_data, request):
        fixture = statcond_regression_data
        arguments = list(fixture["inputs"][0, number - 1][0])
        # This spelling correction is performed by t_statcond.TestClassSetup.
        for index, value in enumerate(arguments[1:], start=1):
            if value.dtype.kind in "US":
                arguments[index] = value.item()
        for index in range(1, len(arguments) - 1, 2):
            if arguments[index] == "variance":
                arguments[index + 1] = arguments[index + 1].replace("homogeneous", "homogenous")
        nargout = 4 if number <= 6 else 3 if number <= 30 else 1
        actual = seeded_statcond("statcond", *arguments, nargout=nargout)
        if nargout > 1:
            outputs = np.empty((1, nargout), dtype=object)
            for index, output in enumerate(actual):
                outputs[0, index] = output
            actual = outputs
        reference = fixture[f"test_{number}"]
        expected = reference["value"][0, 0]
        if request.config.getoption("--eeglab-backend") == "python":
            if number in _PYTHON_TTEST_REGRESSION_CASES:
                _assert_python_ttest_regression(actual, expected, arguments)
                return
            if number in _PYTHON_ANOVA_REGRESSION_CASES:
                _assert_python_anova_regression(actual, expected, arguments)
                return
        _assert_matlab_value(
            actual,
            expected,
            atol=reference["absTol"][0, 0].item(),
            rtol=reference["relTol"][0, 0].item(),
        )

    return test


# Keep one collected test and exact provenance per native regression method.
for _case_number in range(1, 79):
    globals()[f"test_reference_statcond_regression_{_case_number:02d}"] = _statcond_regression_test(_case_number)


def _ordinary_two_tailed_p() -> tuple[np.ndarray, np.ndarray, float]:
    stat = np.array([1.5, 1.5], dtype=np.float64)
    df = np.array([10.0, 10.0], dtype=np.float64)
    probability = float(2.0 * scipy_stats.t.sf(1.5, 10.0))
    return stat, df, probability


def test_two_tailed_t_condition_ignores_zero_and_subnormal_p():
    stat, df, probability = _ordinary_two_tailed_p()
    reference_p = np.array([probability, 0.0], dtype=np.float64)
    expected = 1.5 * 2.0 * scipy_stats.t.pdf(1.5, 10.0) / probability
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert _two_tailed_t_condition(stat, df, reference_p) == pytest.approx(expected)
        subnormal = np.array([probability, np.nextafter(0.0, 1.0)], dtype=np.float64)
        assert _two_tailed_t_condition(stat, df, subnormal) == pytest.approx(expected)
        assert _two_tailed_t_condition(stat[:1], df[:1], np.array([0.0])) == 0.0


def test_derived_p_requires_exact_zero_and_subnormal_values():
    stat, df, probability = _ordinary_two_tailed_p()
    stat_rtol = 10 * float(np.finfo(np.float64).eps)
    reference_p = np.array([probability, 0.0, np.nextafter(0.0, 1.0)], dtype=np.float64)
    reference_stat = np.array([1.5, 1.5, 1.5], dtype=np.float64)
    reference_df = np.array([10.0, 10.0, 10.0], dtype=np.float64)
    _assert_derived_p(reference_p, reference_stat, reference_df, reference_p, stat_rtol=stat_rtol)

    wrong_normal = reference_p.copy()
    wrong_normal[0] = probability * 2.0
    with pytest.raises(AssertionError):
        _assert_derived_p(wrong_normal, reference_stat, reference_df, reference_p, stat_rtol=stat_rtol)

    wrong_zero = reference_p.copy()
    wrong_zero[1] = np.finfo(np.float64).tiny
    with pytest.raises(AssertionError):
        _assert_derived_p(wrong_zero, reference_stat, reference_df, reference_p, stat_rtol=stat_rtol)

    wrong_subnormal = reference_p.copy()
    wrong_subnormal[2] = wrong_subnormal[2] * 2.0
    with pytest.raises(AssertionError):
        _assert_derived_p(wrong_subnormal, reference_stat, reference_df, reference_p, stat_rtol=stat_rtol)


def test_two_tailed_t_condition_rejects_nonfinite_bound():
    with pytest.raises(AssertionError, match="not finite"):
        _two_tailed_t_condition(np.array([np.inf]), np.array([10.0]), np.array([0.2]))


def test_two_tailed_t_condition_keeps_tiny_normal_p_finite():
    # 2e-300 is normal. It stays in the factor, must not warn, and must not
    # become a non-finite tolerance or excuse a zero element from matching.
    stat = np.array([1.0, 1.0], dtype=np.float64)
    df = np.array([10.0, 10.0], dtype=np.float64)
    reference_p = np.array([2e-300, 0.0], dtype=np.float64)
    expected = 1.0 * 2.0 * scipy_stats.t.pdf(1.0, 10.0) / 2e-300
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        factor = _two_tailed_t_condition(stat, df, reference_p)
    assert factor == pytest.approx(expected)
    assert np.isfinite(factor)
    stat_rtol = 21 * float(np.finfo(np.float64).eps)
    wrong_zero = reference_p.copy()
    wrong_zero[1] = np.finfo(np.float64).tiny
    with pytest.raises(AssertionError):
        _assert_derived_p(wrong_zero, stat, df, reference_p, stat_rtol=stat_rtol)


def _reference_conditions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(114)
    first = rng.random((1, 10))
    second = rng.random((1, 10)) + 0.5
    third = rng.random((1, 10)) + 0.2
    return first, second, third


def _assert_paired_t_reference() -> None:
    first, second, _third = _reference_conditions()
    result = statcond([first, second], method="param", paired="on")
    expected = scipy_stats.ttest_rel(first, second, axis=-1)

    np.testing.assert_allclose(result.stat, expected.statistic, rtol=1e-13, atol=1e-13)
    assert result.df == 9
    np.testing.assert_allclose(result.pvalue, expected.pvalue, rtol=1e-13, atol=1e-13)
    assert result.surrogate is None


def _assert_unpaired_t_reference() -> None:
    first, second, _third = _reference_conditions()
    result = statcond([first, second], method="param", paired="off", variance="homogenous")
    expected = scipy_stats.ttest_ind(first, second, axis=-1, equal_var=True)

    np.testing.assert_allclose(result.stat, expected.statistic, rtol=1e-13, atol=1e-13)
    assert result.df == 18
    np.testing.assert_allclose(result.pvalue, expected.pvalue, rtol=1e-13, atol=1e-13)
    assert result.surrogate is None


def _one_way_repeated_reference(groups: Sequence[np.ndarray]) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
    values = np.stack(groups, axis=-2)
    n_conditions, n_subjects = values.shape[-2:]
    grand_mean = np.mean(values, axis=(-2, -1))
    condition_ss = n_subjects * np.sum((np.mean(values, axis=-1) - grand_mean[..., np.newaxis]) ** 2, axis=-1)
    subject_ss = n_conditions * np.sum((np.mean(values, axis=-2) - grand_mean[..., np.newaxis]) ** 2, axis=-1)
    total_ss = np.sum((values - grand_mean[..., np.newaxis, np.newaxis]) ** 2, axis=(-2, -1))
    error_ss = total_ss - condition_ss - subject_ss
    df = (n_conditions - 1, (n_conditions - 1) * (n_subjects - 1))
    statistic = (condition_ss / df[0]) / (error_ss / df[1])
    return statistic, df, scipy_stats.f.sf(statistic, *df)


def _assert_paired_one_way_reference() -> None:
    groups = _reference_conditions()
    result = statcond(groups, method="param", paired="on")
    expected_stat, expected_df, expected_pvalue = _one_way_repeated_reference(groups)

    np.testing.assert_allclose(result.stat, expected_stat, rtol=1e-13, atol=1e-13)
    assert result.df == expected_df
    np.testing.assert_allclose(result.pvalue, expected_pvalue, rtol=1e-13, atol=1e-13)


def _assert_unpaired_one_way_reference() -> None:
    groups = _reference_conditions()
    result = statcond(groups, method="param", paired="off")
    expected = scipy_stats.f_oneway(*groups, axis=-1)

    np.testing.assert_allclose(result.stat, expected.statistic, rtol=1e-13, atol=1e-13)
    assert result.df == (2, 27)
    np.testing.assert_allclose(result.pvalue, expected.pvalue, rtol=1e-13, atol=1e-13)


def _two_way_unpaired_reference(
    grid: Sequence[Sequence[np.ndarray]],
) -> tuple[TwoWayEffects, TwoWayEffects, TwoWayEffects]:
    values = np.stack([np.stack(row, axis=-2) for row in grid], axis=-3)
    n_rows, n_columns, n_cases = values.shape[-3:]
    cell_means = np.mean(values, axis=-1)
    grand_mean = np.mean(cell_means, axis=(-2, -1))
    row_means = np.mean(cell_means, axis=-1)
    column_means = np.mean(cell_means, axis=-2)
    error_ss = np.sum((values - cell_means[..., np.newaxis]) ** 2, axis=(-3, -2, -1))
    row_ss = n_columns * n_cases * np.sum((row_means - grand_mean[..., np.newaxis]) ** 2, axis=-1)
    column_ss = n_rows * n_cases * np.sum((column_means - grand_mean[..., np.newaxis]) ** 2, axis=-1)
    interaction_ss = n_cases * np.sum(
        (
            cell_means
            - row_means[..., :, np.newaxis]
            - column_means[..., np.newaxis, :]
            + grand_mean[..., np.newaxis, np.newaxis]
        )
        ** 2,
        axis=(-2, -1),
    )
    error_df = n_rows * n_columns * (n_cases - 1)
    dfs = TwoWayEffects(
        (n_rows - 1, error_df),
        (n_columns - 1, error_df),
        ((n_rows - 1) * (n_columns - 1), error_df),
    )
    statistics = TwoWayEffects(
        (row_ss / dfs.rows[0]) / (error_ss / error_df),
        (column_ss / dfs.columns[0]) / (error_ss / error_df),
        (interaction_ss / dfs.interaction[0]) / (error_ss / error_df),
    )
    pvalues = TwoWayEffects(
        scipy_stats.f.sf(statistics.rows, *dfs.rows),
        scipy_stats.f.sf(statistics.columns, *dfs.columns),
        scipy_stats.f.sf(statistics.interaction, *dfs.interaction),
    )
    return statistics, dfs, pvalues


def _two_way_repeated_reference(
    grid: Sequence[Sequence[np.ndarray]],
) -> tuple[TwoWayEffects, TwoWayEffects, TwoWayEffects]:
    values = np.stack([np.stack(row, axis=-2) for row in grid], axis=-3)
    n_rows, n_columns, n_subjects = values.shape[-3:]
    ab_sums = np.sum(values, axis=-1)
    row_subject_sums = np.sum(values, axis=-2)
    column_subject_sums = np.sum(values, axis=-3)
    row_sums = np.sum(ab_sums, axis=-1)
    column_sums = np.sum(ab_sums, axis=-2)
    subject_sums = np.sum(row_subject_sums, axis=-2)
    total = np.sum(values, axis=(-3, -2, -1))

    expected_rows = np.sum(row_sums**2, axis=-1) / (n_columns * n_subjects)
    expected_columns = np.sum(column_sums**2, axis=-1) / (n_rows * n_subjects)
    expected_ab = np.sum(ab_sums**2, axis=(-2, -1)) / n_subjects
    expected_subjects = np.sum(subject_sums**2, axis=-1) / (n_rows * n_columns)
    expected_row_subject = np.sum(row_subject_sums**2, axis=(-2, -1)) / n_columns
    expected_column_subject = np.sum(column_subject_sums**2, axis=(-2, -1)) / n_rows
    expected_y = np.sum(values**2, axis=(-3, -2, -1))
    expected_total = total**2 / (n_rows * n_columns * n_subjects)

    row_ss = expected_rows - expected_total
    column_ss = expected_columns - expected_total
    interaction_ss = expected_ab - expected_rows - expected_columns + expected_total
    row_subject_ss = expected_row_subject - expected_rows - expected_subjects + expected_total
    column_subject_ss = expected_column_subject - expected_columns - expected_subjects + expected_total
    interaction_subject_ss = (
        expected_y
        - expected_ab
        - expected_row_subject
        - expected_column_subject
        + expected_rows
        + expected_columns
        + expected_subjects
        - expected_total
    )
    dfs = TwoWayEffects(
        (n_rows - 1, (n_rows - 1) * (n_subjects - 1)),
        (n_columns - 1, (n_columns - 1) * (n_subjects - 1)),
        (
            (n_rows - 1) * (n_columns - 1),
            (n_rows - 1) * (n_columns - 1) * (n_subjects - 1),
        ),
    )
    statistics = TwoWayEffects(
        (row_ss / dfs.rows[0]) / (row_subject_ss / dfs.rows[1]),
        (column_ss / dfs.columns[0]) / (column_subject_ss / dfs.columns[1]),
        (interaction_ss / dfs.interaction[0]) / (interaction_subject_ss / dfs.interaction[1]),
    )
    pvalues = TwoWayEffects(
        scipy_stats.f.sf(statistics.rows, *dfs.rows),
        scipy_stats.f.sf(statistics.columns, *dfs.columns),
        scipy_stats.f.sf(statistics.interaction, *dfs.interaction),
    )
    return statistics, dfs, pvalues


def _reference_grid() -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    first, second, third = _reference_conditions()
    return (
        (first / 2, second, third),
        (first.copy(), second.copy(), third.copy()),
    )


def _unpaired_factor_order_grid() -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    return (
        (
            np.array([1, 2, 4, 3, 5], dtype=float),
            np.array([2, 2, 3, 5, 4], dtype=float),
            np.array([3, 4, 2, 5, 6], dtype=float),
        ),
        (
            np.array([2, 3, 1, 4, 5], dtype=float),
            np.array([4, 2, 5, 3, 6], dtype=float),
            np.array([5, 6, 3, 7, 4], dtype=float),
        ),
    )


def _assert_unpaired_two_way_matlab_golden() -> None:
    result = statcond(_unpaired_factor_order_grid(), method="param", paired="off")

    # Direct MATLAB output from EEGLAB 8ac485f is positionally columns,
    # rows, interaction for the unpaired branch. Map those values to the
    # documented factor meanings instead of preserving the positional defect.
    matlab_statistics = (2.40845084190369, 1.14084577560425, 0.295774638652802)
    matlab_dfs = ((2, 24), (1, 24), (2, 24))
    matlab_pvalues = (0.111369788646698, 0.296099960803986, 0.746627926826477)
    expected_statistics = TwoWayEffects(matlab_statistics[1], matlab_statistics[0], matlab_statistics[2])
    expected_dfs = TwoWayEffects(matlab_dfs[1], matlab_dfs[0], matlab_dfs[2])
    expected_pvalues = TwoWayEffects(matlab_pvalues[1], matlab_pvalues[0], matlab_pvalues[2])

    assert isinstance(result.stat, TwoWayEffects)
    assert result.df == expected_dfs
    assert isinstance(result.pvalue, TwoWayEffects)
    for actual, expected in zip(result.stat, expected_statistics, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    for actual, expected in zip(result.pvalue, expected_pvalues, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def _assert_effects(result: StatcondResult, expected: tuple[TwoWayEffects, TwoWayEffects, TwoWayEffects]) -> None:
    expected_statistics, expected_dfs, expected_pvalues = expected
    assert isinstance(result.stat, TwoWayEffects)
    assert isinstance(result.df, TwoWayEffects)
    assert isinstance(result.pvalue, TwoWayEffects)
    for actual, reference in zip(result.stat, expected_statistics, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)
    for actual, reference in zip(result.df, expected_dfs, strict=True):
        assert actual == reference
    for actual, reference in zip(result.pvalue, expected_pvalues, strict=True):
        np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)


def _assert_paired_two_way_reference() -> None:
    grid = _reference_grid()
    _assert_effects(statcond(grid, method="param", paired="on"), _two_way_repeated_reference(grid))


def _assert_unpaired_two_way_reference() -> None:
    grid = _reference_grid()
    _assert_effects(statcond(grid, method="param", paired="off"), _two_way_unpaired_reference(grid))


def test_statcond_paired_t_matches_independent_reference():
    _assert_paired_t_reference()


def test_statcond_unpaired_t_matches_independent_reference():
    _assert_unpaired_t_reference()


def test_statcond_paired_one_way_anova_matches_independent_reference():
    _assert_paired_one_way_reference()


def test_statcond_paired_two_way_anova_matches_independent_reference():
    _assert_paired_two_way_reference()


def test_statcond_unpaired_one_way_anova_matches_independent_reference():
    _assert_unpaired_one_way_reference()


def test_statcond_unpaired_two_way_anova_uses_documented_factor_order():
    _assert_unpaired_two_way_reference()
    _assert_unpaired_two_way_matlab_golden()


def _dimensional_conditions() -> tuple[tuple[tuple[np.ndarray, ...], tuple[int, ...]], ...]:
    rng = np.random.default_rng(518)
    base = tuple(rng.random((1, 10)) + offset for offset in (0.0, 0.5, 0.2))
    matrix = tuple(rng.random((10, 10)) + offset for offset in (0.0, 0.5, 0.2))
    cube = tuple(rng.random((5, 10, 10)) + offset for offset in (0.0, 0.5, 0.2))
    hypercube = tuple(rng.random((2, 5, 10, 10)) + offset for offset in (0.0, 0.5, 0.2))
    for index in range(3):
        matrix[index][3, :] = base[index][0]
        cube[index][1, 3, :] = base[index][0]
        hypercube[index][0, 1, 3, :] = base[index][0]
    return (
        (base, (0,)),
        (matrix, (3,)),
        (cube, (1, 3)),
        (hypercube, (0, 1, 3)),
    )


def _value_at(value: np.ndarray | float, index: tuple[int, ...]) -> float:
    array = np.asarray(value)
    return float(array[index] if index else array)


def _assert_dimension_invariance(*, paired: str, design: str) -> None:
    conditions_by_dimension = _dimensional_conditions()
    results: list[tuple[StatcondResult, tuple[int, ...]]] = []
    for conditions, index in conditions_by_dimension:
        data: Sequence[Sequence[np.ndarray]] | Sequence[np.ndarray]
        if design == "t":
            data = conditions[:2]
        elif design == "one-way":
            data = conditions
        else:
            data = (
                (conditions[0] / 2, conditions[1], conditions[2]),
                tuple(condition.copy() for condition in conditions),
            )
        result = statcond(data, method="param", paired=paired, variance="homogenous")
        expected_shape = conditions[0].shape[:-1]
        if design == "two-way":
            assert all(np.asarray(effect).shape == expected_shape for effect in result.stat)
        else:
            assert np.asarray(result.stat).shape == expected_shape
        results.append((result, index))

    baseline, _baseline_index = results[0]
    for result, index in results[1:]:
        if design == "two-way":
            for actual, expected in zip(result.stat, baseline.stat, strict=True):
                np.testing.assert_allclose(_value_at(actual, index), expected, rtol=1e-12, atol=1e-12)
            for actual, expected in zip(result.pvalue, baseline.pvalue, strict=True):
                np.testing.assert_allclose(_value_at(actual, index), expected, rtol=1e-12, atol=1e-12)
            assert result.df == baseline.df
        else:
            np.testing.assert_allclose(_value_at(result.stat, index), baseline.stat, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(_value_at(result.pvalue, index), baseline.pvalue, rtol=1e-12, atol=1e-12)
            assert result.df == baseline.df


def test_statcond_paired_t_is_invariant_across_feature_dimensions():
    _assert_dimension_invariance(paired="on", design="t")


def test_statcond_unpaired_t_is_invariant_across_feature_dimensions():
    _assert_dimension_invariance(paired="off", design="t")


def test_statcond_paired_one_way_is_invariant_across_feature_dimensions():
    _assert_dimension_invariance(paired="on", design="one-way")


def test_statcond_unpaired_one_way_is_invariant_across_feature_dimensions():
    _assert_dimension_invariance(paired="off", design="one-way")


def test_statcond_paired_two_way_is_invariant_across_feature_dimensions():
    _assert_dimension_invariance(paired="on", design="two-way")


def test_statcond_unpaired_two_way_is_invariant_across_feature_dimensions():
    _assert_dimension_invariance(paired="off", design="two-way")


def _resampling_conditions(feature_shape: tuple[int, ...], n_conditions: int) -> tuple[np.ndarray, ...]:
    offsets = (0, 100, 1000)
    arrays = []
    for offset in offsets[:n_conditions]:
        values = np.zeros((*feature_shape, 10), dtype=np.float64)
        values.reshape(-1, 10)[-1] = np.arange(1, 11) + offset
        arrays.append(values)
    return tuple(arrays)


def _last_feature_trace(array: np.ndarray) -> np.ndarray:
    return np.asarray(array).reshape(-1, array.shape[-1])[-1]


def _assert_resampling_case(
    conditions: tuple[np.ndarray, ...],
    *,
    method: str,
    paired: str,
    naccu: int,
    seed: int,
    arraycomp: bool,
) -> None:
    originals = tuple(condition.copy() for condition in conditions)
    result = statcond(
        conditions,
        method=method,
        paired=paired,
        naccu=naccu,
        rng=seed,
        return_resampling_array=True,
        arraycomp=arraycomp,
    )
    repeated = statcond(
        conditions,
        method=method,
        paired=paired,
        naccu=naccu,
        rng=seed,
        return_resampling_array=True,
        arraycomp=arraycomp,
    )
    assert isinstance(result, SurrogateDistribution)
    assert isinstance(repeated, SurrogateDistribution)
    assert len(result) == (naccu if arraycomp else 1)
    source_traces = np.stack([_last_feature_trace(condition) for condition in conditions])
    pooled_source = source_traces.ravel()

    for sample, repeated_sample in zip(result, repeated, strict=True):
        output_conditions = sample[0]
        repeated_conditions = repeated_sample[0]
        assert len(output_conditions) == len(conditions)
        for source, output, repeated_output in zip(conditions, output_conditions, repeated_conditions, strict=True):
            assert output.shape == source.shape
            assert output.dtype == source.dtype
            np.testing.assert_array_equal(output, repeated_output)
        output_traces = np.stack([_last_feature_trace(output) for output in output_conditions])

        if paired == "on":
            for case_index in range(source_traces.shape[1]):
                assert np.isin(output_traces[:, case_index], source_traces[:, case_index]).all()
            if method == "perm":
                np.testing.assert_array_equal(np.sort(output_traces, axis=0), np.sort(source_traces, axis=0))
        else:
            assert np.isin(output_traces, pooled_source).all()
            if method == "perm":
                np.testing.assert_array_equal(np.sort(output_traces.ravel()), np.sort(pooled_source))
            else:
                assert all(np.unique(trace).size > 3 for trace in output_traces)

    for original, condition in zip(originals, conditions, strict=True):
        np.testing.assert_array_equal(condition, original)


def _assert_resampling_suite() -> None:
    seed = 720
    for feature_shape in ((1,), (10,), (9, 8)):
        for n_conditions in (2, 3):
            conditions = _resampling_conditions(feature_shape, n_conditions)
            for vectorized in (True, False):
                for method, paired in (
                    ("bootstrap", "on"),
                    ("perm", "on"),
                    ("bootstrap", "off"),
                    ("perm", "off"),
                ):
                    seed += 1
                    _assert_resampling_case(
                        conditions,
                        method=method,
                        paired=paired,
                        naccu=10,
                        seed=seed,
                        arraycomp=vectorized,
                    )


def test_statcond_resampling_preserves_assignment_invariants():
    _assert_resampling_suite()


def _matlab_cells(rows):
    cells = np.empty((len(rows), len(rows[0])), dtype=object)
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            cells[row_index, column_index] = value
    return cells


def _anova_resampling_grid(design: str) -> np.ndarray:
    def block(seed: int) -> np.ndarray:
        return np.random.default_rng(seed).normal(size=(2, 6))

    if design == "t":
        rows = [[block(1), block(2)]]
    elif design == "one-way":
        rows = [[block(1), block(2), block(3)]]
    else:
        rows = [[block(1), block(2)], [block(3), block(4)]]
    return _matlab_cells(rows)


def _expected_resampling_dtype(design: str, arraycomp: str):
    # concatdata.m is single. ttest_cell keeps that class for both arraycomp
    # modes. ANOVA arraycomp on replaces it with the double anova*_cell output.
    if design != "t" and arraycomp == "on":
        return np.float64
    return np.float32


def _assert_numeric_dtype(value, dtype) -> None:
    array = np.asarray(value)
    if array.dtype == object:
        assert array.size > 0
        for item in array.ravel():
            _assert_numeric_dtype(item, dtype)
        return
    assert array.size > 0
    assert array.dtype == dtype


@pytest.mark.parametrize("method", ["perm", "bootstrap"])
@pytest.mark.parametrize("arraycomp", ["on", "off"])
@pytest.mark.parametrize("design", ["t", "one-way", "two-way"])
def test_statcond_f_resampling_dtype_follows_arraycomp(method, design, arraycomp):
    grid = _anova_resampling_grid(design)
    paired = "on" if design == "two-way" else "off"
    expected = _expected_resampling_dtype(design, arraycomp)
    result = statcond(grid, method=method, paired=paired, naccu=8, arraycomp=arraycomp, rng=1, verbose="off")
    _assert_numeric_dtype(result.matlab_surrogate, expected)
    # statcond.m:511 asks stat_surrogate_ci for a numeric array. A two-way
    # surrogate is a cell, so that call errors. Check the cell class only.
    if design == "two-way":
        return
    structured = statcond(
        grid,
        method=method,
        paired=paired,
        naccu=8,
        arraycomp=arraycomp,
        rng=1,
        verbose="off",
        structoutput="on",
        alpha=0.05,
    )
    _assert_numeric_dtype(structured.matlab_primary["surrog"], expected)
    _assert_numeric_dtype(structured.matlab_primary["ci"], expected)


@pytest.mark.parametrize("method", ["perm", "bootstrap"])
@pytest.mark.parametrize("arraycomp", ["on", "off"])
@pytest.mark.parametrize("design", ["t", "one-way", "two-way"])
def test_reference_statcond_f_resampling_dtype_follows_arraycomp(eeglab_backend, method, design, arraycomp):
    grid = _anova_resampling_grid(design)
    paired = "on" if design == "two-way" else "off"
    expected = _expected_resampling_dtype(design, arraycomp)
    _statistic, _degrees, _probability, surrogate = eeglab_backend(
        "statcond",
        grid,
        method=method,
        paired=paired,
        naccu=8,
        arraycomp=arraycomp,
        verbose="off",
        nargout=4,
    )
    _assert_numeric_dtype(surrogate, expected)
    if design == "two-way":
        return
    output = eeglab_backend(
        "statcond",
        grid,
        method=method,
        paired=paired,
        naccu=8,
        arraycomp=arraycomp,
        verbose="off",
        structoutput="on",
        alpha=0.05,
        nargout=1,
    )
    _assert_numeric_dtype(output["surrog"], expected)
    _assert_numeric_dtype(output["ci"], expected)


@pytest.mark.parametrize("stat_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("surrog_dtype", [np.float32, np.float64])
def test_statcond_supplied_arrays_keep_their_class(stat_dtype, surrog_dtype):
    # statcond.m:444-445 assigns the supplied arrays. reshape_results keeps
    # the class when naccu matches the surrogate length. pvals stay double.
    naccu = 4
    data = _matlab_cells([[np.ones((2, 6)), np.ones((2, 6))]])
    stats = np.arange(2, dtype=stat_dtype)
    surrog = np.arange(2 * naccu, dtype=surrog_dtype).reshape(2, naccu)
    result = statcond(
        data,
        method="perm",
        surrog=surrog,
        stats=stats,
        naccu=naccu,
        verbose="off",
        structoutput="on",
        alpha=0.05,
    )
    _assert_numeric_dtype(result.matlab_primary["stat"], stat_dtype)
    _assert_numeric_dtype(result.matlab_primary["surrog"], surrog_dtype)
    _assert_numeric_dtype(result.matlab_primary["pval"], np.float64)
    _assert_numeric_dtype(result.matlab_primary["ci"], surrog_dtype)
    _assert_numeric_dtype(result.matlab_surrogate, surrog_dtype)


@pytest.mark.parametrize("stat_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("surrog_dtype", [np.float32, np.float64])
def test_reference_statcond_supplied_arrays_keep_their_class(eeglab_backend, stat_dtype, surrog_dtype):
    naccu = 4
    data = _matlab_cells([[np.ones((2, 6)), np.ones((2, 6))]])
    stats = np.arange(2, dtype=stat_dtype)
    surrog = np.arange(2 * naccu, dtype=surrog_dtype).reshape(2, naccu)
    statistic, _degrees, probability, surrogate = eeglab_backend(
        "statcond",
        data,
        method="perm",
        surrog=surrog,
        stats=stats,
        naccu=naccu,
        verbose="off",
        nargout=4,
    )
    _assert_numeric_dtype(statistic, stat_dtype)
    _assert_numeric_dtype(surrogate, surrog_dtype)
    _assert_numeric_dtype(probability, np.float64)
    output = eeglab_backend(
        "statcond",
        data,
        method="perm",
        surrog=surrog,
        stats=stats,
        naccu=naccu,
        verbose="off",
        structoutput="on",
        alpha=0.05,
        nargout=1,
    )
    _assert_numeric_dtype(output["stat"], stat_dtype)
    _assert_numeric_dtype(output["surrog"], surrog_dtype)
    _assert_numeric_dtype(output["pval"], np.float64)
    _assert_numeric_dtype(output["ci"], surrog_dtype)


def _assert_class_same(actual, expected):
    # statcondTest.verifySame uses RelTol=1e-2, without an absolute tolerance.
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=1e-2, atol=0)


def _statcond_class_vector_test(name, paired, design):
    @eeglab_test(STATCOND_CLASS, name)
    def test(eeglab_backend):
        rng = np.random.default_rng(114)
        if design == "t":
            rows = [[rng.random((1, 10)), rng.random((1, 10)) + 0.5]]
            kwargs = {"variance": "homogenous"} if paired == "off" else {}
            statistic, df, pvalue, _surrogate = eeglab_backend(
                "statcond", _matlab_cells(rows), mode="param", verbose="off", paired=paired, nargout=4, **kwargs
            )
            reference = (
                scipy_stats.ttest_rel(*rows[0], axis=-1)
                if paired == "on"
                else scipy_stats.ttest_ind(*rows[0], axis=-1, equal_var=True)
            )
            expected = reference.statistic, 9.0 if paired == "on" else 18.0, reference.pvalue
        else:
            # Source anova_a contains six independently generated arrays.
            rows = [
                [rng.random((1, 10)), rng.random((1, 10)), rng.random((1, 10)) + 0.2],
                [rng.random((1, 10)), rng.random((1, 10)) + 0.2, rng.random((1, 10))],
            ]
            data = rows[:1] if design == "one-way" else rows
            statistic, df, pvalue, _surrogate = eeglab_backend(
                "statcond", _matlab_cells(data), mode="param", verbose="off", paired=paired, nargout=4
            )
            if design == "one-way":
                if paired == "on":
                    expected = _one_way_repeated_reference(rows[0])
                else:
                    reference = scipy_stats.f_oneway(*rows[0], axis=-1)
                    expected = reference.statistic, (2, 27), reference.pvalue
            else:
                reference = _two_way_repeated_reference(rows) if paired == "on" else _two_way_unpaired_reference(rows)
                expected = tuple(value.interaction for value in reference)
                statistic, df, pvalue = (value[0, 2] for value in (statistic, df, pvalue))
        # Source vector tests compare scalar F/t and p values, and each df
        # entry separately; ANOVA explicitly converts F and p to double.
        if design != "t":
            statistic, pvalue = statistic.astype(float), pvalue.astype(float)
        _assert_class_same(statistic.flat[0], np.asarray(expected[0]).flat[0])
        for actual, reference in zip(df.flat, np.asarray(expected[1], dtype=float).flat, strict=True):
            _assert_class_same(actual, reference)
        _assert_class_same(pvalue.flat[0], np.asarray(expected[2]).flat[0])

    return test


def _statcond_class_dimension_test(name, paired, design):
    @eeglab_test(STATCOND_CLASS, name)
    def test(eeglab_backend):
        rng = np.random.default_rng(518)
        conditions = [
            [rng.random((*shape, 10)) + offset for offset in (0, 0.5, 0)]
            for shape in ((1,), (10,), (5, 10), (2, 5, 10))
        ]
        indices = ((0,), (3,), (1, 3), (0, 1, 3))
        for arrays, index in zip(conditions[1:], indices[1:], strict=True):
            for source, target in zip(conditions[0], arrays, strict=True):
                target[index] = source[0]
        baseline = None
        for arrays, index in zip(conditions, indices, strict=True):
            rows = [arrays[:2]] if design == "t" else [arrays]
            if design == "two-way":
                rows = [[arrays[0] / 2, arrays[1], arrays[2]], arrays]
            kwargs = {"variance": "homogenous"} if design == "t" and paired == "off" else {}
            statistic, df, pvalue = eeglab_backend(
                "statcond", _matlab_cells(rows), mode="param", verbose="off", paired=paired, nargout=3, **kwargs
            )
            if design == "two-way":
                statistic, df, pvalue = (value[0, 2] for value in (statistic, df, pvalue))
            # MATLAB indexes scalar / vector / matrix / 3-D statistic maps.
            feature_index = () if statistic.size == 1 else index
            current = (
                np.asarray(statistic if not feature_index else statistic[feature_index]).flat[0],
                df,
                np.asarray(pvalue if not feature_index else pvalue[feature_index]).flat[0],
            )
            if design == "two-way":
                # Source compares interaction df entries individually here.
                current = (current[0], df[0, 0], df[0, 1], current[2])
            if baseline is None:
                baseline = current
            else:
                for actual, reference in zip(current, baseline, strict=True):
                    _assert_class_same(actual, reference)

    return test


for _design, _suffix in (("t", "TTest"), ("one-way", "1Anova"), ("two-way", "2Anova")):
    for _paired, _prefix in (("on", "paired"), ("off", "unpaired")):
        _name = f"{_prefix}{_suffix}"
        globals()[f"test_reference_statcond_{_name}"] = _statcond_class_vector_test(_name, _paired, _design)
        _name = f"{_prefix}Dim{_suffix}"
        globals()[f"test_reference_statcond_{_name}"] = _statcond_class_dimension_test(_name, _paired, _design)


@pytest.mark.parametrize("arraycomp", ["on", "off"])
@pytest.mark.parametrize(
    "feature_shape,n_conditions", [((1,), 2), ((1,), 3), ((10,), 2), ((10,), 3), ((9, 8), 2), ((9, 8), 3)]
)
@eeglab_test(STATCOND_CLASS, "shuffleAndPermutation")
def test_reference_statcond_shuffle_and_permutation(eeglab_backend, feature_shape, n_conditions, arraycomp):
    sa1, sa2, sa3, sa4 = _reference_shuffle_arrays(eeglab_backend, feature_shape, n_conditions, arraycomp)
    for value in sa1[:2]:
        assert value.dtype == np.float32  # Source verifyEqual expects single.
        np.testing.assert_array_equal(np.remainder(value, 10), [1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
    means = np.mean(sa2, axis=0)
    assert means.dtype == np.float32
    # Source repeats the same two sa1 checks in its permutation section.
    np.testing.assert_array_equal(np.round(means - means[0]), np.arange(10))
    assert all(np.unique(value).size == 10 for value in sa2[:2])
    assert all(np.unique(value).size > 3 for value in sa3[:2])
    assert all(np.unique(value).size == 10 for value in sa4[:2])
    assert np.floor(np.mean(np.mean(sa4, axis=0))) in (55, 372)


def _reference_shuffle_arrays(eeglab_backend, feature_shape, n_conditions, arraycomp):
    conditions = _resampling_conditions(feature_shape, n_conditions)
    data = _matlab_cells([conditions])
    arrays = []
    for mode, paired in (("bootstrap", "on"), ("perm", "on"), ("bootstrap", "off"), ("perm", "off")):
        kwargs = {} if arraycomp == "on" else {"arraycomp": "off"}
        result = eeglab_backend(
            "statcond", data, mode=mode, verbose="off", paired=paired, returnresamplingarray="on", naccu=10.0, **kwargs
        )
        traces = []
        for value in result.ravel():
            # Preserve the source's ndims branches, including its selection
            # of the last replicate when arraycomp adds a resampling dimension.
            if value.ndim == 2 and value.shape[1] > 1:
                value = value[-1, :]
            elif value.ndim == 3:
                value = value[-1, -1, :]
            elif value.ndim == 4:
                value = value[-1, -1, -1, :]
            traces.append(value.ravel())
        arrays.append(traces)
    return arrays


def _legacy_assertsame(*values):
    # Source's early return checks only the first argument when it has >2
    # entries. Preserve this blind spot; the newer class separately checks df/p.
    if len(values[0]) > 2:
        for index in range(len(values[0]) - 1):
            _legacy_assertsame(values[0][index : index + 2])
        return
    for value in values:
        assert not abs(value[0] - value[1]) > abs(np.mean(value)) * 0.01


@pytest.fixture
def legacy_statcond_oracle_path(request, eeglab_suite_root):
    if request.config.getoption("--eeglab-backend") != "matlab":
        yield
        return
    engine = request.getfixturevalue("eeglab_matlab_engine")
    original_path = engine.path()
    try:
        # This standalone oracle is identical to the local copy in test_statcond.
        engine.addpath(str(eeglab_suite_root / "unittesting_statistics" / "statcond"), nargout=0)
        yield
    finally:
        engine.path(original_path, nargout=0)


def _legacy_matlab_reference(eeglab_backend, t_data, anova_data, paired, design):
    if design == "t":
        _h, pvalue, _interval, statistics = eeglab_backend("ttest" if paired == "on" else "ttest2", *t_data, nargout=4)
        return statistics["tstat"], (statistics["df"].item(),), pvalue

    if paired == "on":
        z, o, t = np.zeros((10, 1)), np.ones((10, 1)), np.full((10, 1), 2.0)
        rows = anova_data[:1] if design == "one-way" else anova_data
        statistics = eeglab_backend(
            "rm_anova2",
            np.concatenate([value.T for row in rows for value in row]),
            np.tile(np.arange(1.0, 11.0)[:, None], (3 * len(rows), 1)),
            np.concatenate([o, o, o] if design == "one-way" else [o, o, o, z, z, z]),
            np.concatenate([z, o, t] * len(rows)),
            _matlab_cells([["a", "b"]]),
        )
        effect_row, error_row = (2, 5) if design == "one-way" else (3, 6)
    elif design == "one-way":
        _pvalue, statistics = eeglab_backend(
            "anova1",
            np.concatenate([value.T for value in anova_data[0]], axis=1),
            np.empty((0, 0), dtype=object),
            "off",
            nargout=2,
        )
        effect_row, error_row = 1, 2
    else:
        _pvalue, statistics = eeglab_backend(
            "anova2", np.block([[value.T for value in row] for row in anova_data]), 10.0, "off", nargout=2
        )
        effect_row, error_row = 3, 4
    return (
        statistics[effect_row, 4],
        (statistics[effect_row, 2].item(), statistics[error_row, 2].item()),
        statistics[effect_row, 5],
    )


@eeglab_test(STATCOND_WRAPPER, "test_test_statcond")
def test_reference_legacy_statcond_workflow(eeglab_backend, request, legacy_statcond_oracle_path):
    native = request.config.getoption("--eeglab-backend") == "matlab"
    # MATLAB's licensed toolbox prerequisite has no Python licensing analogue;
    # SciPy, imported above, supplies the independent Python statistical oracles.
    available = not native or (
        eeglab_backend("license", "checkout", "statistics_toolbox").item()
        and eeglab_backend("exist", "kmeans", "file").item()
    )
    request.node.user_properties.append(("eeglab_source_body_entered", bool(available)))
    if not available:
        return
    rng = np.random.default_rng(114)
    t_data = [rng.random((1, 10)), rng.random((1, 10)) + 0.5]
    anova_data = [
        [rng.random((1, 10)), rng.random((1, 10)), rng.random((1, 10)) + 0.2],
        [rng.random((1, 10)), rng.random((1, 10)) + 0.2, rng.random((1, 10))],
    ]
    for paired, design in (
        ("on", "t"),
        ("off", "t"),
        ("on", "one-way"),
        ("on", "two-way"),
        ("off", "one-way"),
        ("off", "two-way"),
    ):
        rows = [t_data] if design == "t" else anova_data[:1] if design == "one-way" else anova_data
        kwargs = {"variance": "homogenous"} if design == "t" and paired == "off" else {}
        statistic, df, pvalue, _surrogate = eeglab_backend(
            "statcond", _matlab_cells(rows), mode="param", verbose="off", paired=paired, nargout=4, **kwargs
        )
        if native:
            expected = _legacy_matlab_reference(eeglab_backend, t_data, anova_data, paired, design)
        elif design == "t":
            reference = (
                scipy_stats.ttest_rel(*t_data, axis=-1)
                if paired == "on"
                else scipy_stats.ttest_ind(*t_data, axis=-1, equal_var=True)
            )
            expected = reference.statistic, (9.0 if paired == "on" else 18.0,), reference.pvalue
        elif design == "one-way":
            if paired == "on":
                expected = _one_way_repeated_reference(anova_data[0])
            else:
                reference = scipy_stats.f_oneway(*anova_data[0], axis=-1)
                expected = reference.statistic, (2.0, 27.0), reference.pvalue
        else:
            reference = (
                _two_way_repeated_reference(anova_data) if paired == "on" else _two_way_unpaired_reference(anova_data)
            )
            expected = tuple(value.interaction for value in reference)
        if design == "two-way":
            statistic, df, pvalue = (value[0, 2] for value in (statistic, df, pvalue))
        pairs = [np.array([statistic.flat[0], np.asarray(expected[0]).flat[0]], dtype=statistic.dtype)]
        pairs.extend(
            np.array([actual, value], dtype=df.dtype) for actual, value in zip(df.flat, expected[1], strict=True)
        )
        pairs.append(np.array([pvalue.flat[0], np.asarray(expected[2]).flat[0]], dtype=pvalue.dtype))
        _legacy_assertsame(*pairs)

    conditions = [
        [rng.random((*shape, 10)) + offset for offset in (0, 0.5, 0)] for shape in ((1,), (10,), (5, 10), (2, 5, 10))
    ]
    indices = ((0,), (3,), (1, 3), (0, 1, 3))
    for arrays, index in zip(conditions[1:], indices[1:], strict=True):
        for source, target in zip(conditions[0], arrays, strict=True):
            target[index] = source[0]
    for design in ("t", "one-way", "two-way"):
        for paired in ("on", "off"):
            values = []
            for arrays, index in zip(conditions, indices, strict=True):
                rows = (
                    [arrays[:2]]
                    if design == "t"
                    else [arrays]
                    if design == "one-way"
                    else [[arrays[0] / 2, arrays[1], arrays[2]], arrays]
                )
                kwargs = {"variance": "homogenous"} if design == "t" and paired == "off" else {}
                statistic, df, pvalue = eeglab_backend(
                    "statcond", _matlab_cells(rows), mode="param", verbose="off", paired=paired, nargout=3, **kwargs
                )
                if design == "two-way":
                    statistic, df, pvalue = (value[0, 2] for value in (statistic, df, pvalue))
                feature = () if statistic.size == 1 else index
                values.append((np.asarray(statistic[feature]).flat[0], df, np.asarray(pvalue[feature]).flat[0]))
            _legacy_assertsame(
                np.array([value[0] for value in values]),
                np.concatenate([value[1].ravel() for value in values]),
                np.array([value[2] for value in values]),
            )
    for arraycomp in ("on", "off"):
        for feature_shape, n_conditions in (((1,), 2), ((1,), 3), ((10,), 2), ((10,), 3), ((9, 8), 2), ((9, 8), 3)):
            sa1, sa2, sa3, sa4 = _reference_shuffle_arrays(eeglab_backend, feature_shape, n_conditions, arraycomp)
            for value in sa1[:2]:
                np.testing.assert_array_equal(np.remainder(value, 10), [1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
            means = np.mean(sa2, axis=0)
            np.testing.assert_array_equal(np.round(means - means[0]), np.arange(10))
            assert all(np.unique(value).size == 10 for value in sa2[:2])
            assert all(np.unique(value).size > 3 for value in sa3[:2])
            assert all(np.unique(value).size == 10 for value in sa4[:2])
            assert np.floor(np.mean(np.mean(sa4, axis=0))) in (55, 372)
