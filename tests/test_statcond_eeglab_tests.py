"""Faithful reference contracts and additional Python ``statcond`` checks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from scipy.io import loadmat
from scipy import stats as scipy_stats

from eegprep.functions.statistics import StatcondResult, SurrogateDistribution, TwoWayEffects, statcond
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
        np.random.seed(0)
    return eeglab_backend


def _assert_matlab_value(actual, expected, *, atol, rtol):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    if expected.dtype == object:
        for index in np.ndindex(expected.shape):
            _assert_matlab_value(actual[index], expected[index], atol=atol, rtol=rtol)
    else:
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def _statcond_regression_test(number):
    @eeglab_test(STATCOND_REGRESSION, f"test_{number}")
    def test(seeded_statcond, statcond_regression_data):
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
        _assert_matlab_value(
            actual,
            reference["value"][0, 0],
            atol=reference["absTol"][0, 0].item(),
            rtol=reference["relTol"][0, 0].item(),
        )

    return test


# Keep one collected test and exact provenance per native regression method.
for _case_number in range(1, 79):
    globals()[f"test_reference_statcond_regression_{_case_number:02d}"] = _statcond_regression_test(_case_number)


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


def test_additional_statcond_python_workflow():
    _assert_paired_t_reference()
    _assert_unpaired_t_reference()
    _assert_paired_one_way_reference()
    _assert_unpaired_one_way_reference()
    _assert_paired_two_way_reference()
    _assert_unpaired_two_way_reference()
    for paired in ("on", "off"):
        for design in ("t", "one-way", "two-way"):
            _assert_dimension_invariance(paired=paired, design=design)
    _assert_resampling_suite()


def _matlab_cells(rows):
    cells = np.empty((len(rows), len(rows[0])), dtype=object)
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            cells[row_index, column_index] = value
    return cells


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


@eeglab_test(STATCOND_WRAPPER, "test_test_statcond")
def test_reference_legacy_statcond_workflow(eeglab_backend):
    if (
        not eeglab_backend("license", "checkout", "statistics_toolbox").item()
        or not eeglab_backend("exist", "kmeans", "file").item()
    ):
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
        if design == "t":
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
