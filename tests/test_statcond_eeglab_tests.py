"""Ports of the current EEGLAB ``statcond`` MATLAB tests."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats as scipy_stats

from eegprep.functions.statistics import StatcondResult, SurrogateDistribution, TwoWayEffects, statcond
from tests.eeglab_tests import eeglab_test


STATCOND_CLASS = "unittesting_statistics/statcond/statcondTest.m"
STATCOND_WRAPPER = "unittesting_statistics/statcond/statistics_statcond_wrapperTest.m"
STATCOND_REGRESSION = "regression_tests/t_statcond.m"


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


@eeglab_test(STATCOND_REGRESSION, "test_1")
@eeglab_test(STATCOND_CLASS, "pairedTTest")
def test_statcond_paired_t_matches_independent_reference():
    _assert_paired_t_reference()


@eeglab_test(STATCOND_REGRESSION, "test_2")
@eeglab_test(STATCOND_CLASS, "unpairedTTest")
def test_statcond_unpaired_t_matches_independent_reference():
    _assert_unpaired_t_reference()


@eeglab_test(STATCOND_REGRESSION, "test_3")
@eeglab_test(STATCOND_CLASS, "paired1Anova")
def test_statcond_paired_one_way_anova_matches_independent_reference():
    _assert_paired_one_way_reference()


@eeglab_test(STATCOND_REGRESSION, "test_4")
@eeglab_test(STATCOND_CLASS, "paired2Anova")
def test_statcond_paired_two_way_anova_matches_independent_reference():
    _assert_paired_two_way_reference()


@eeglab_test(STATCOND_REGRESSION, "test_5")
@eeglab_test(STATCOND_CLASS, "unpaired1Anova")
def test_statcond_unpaired_one_way_anova_matches_independent_reference():
    _assert_unpaired_one_way_reference()


@eeglab_test(STATCOND_REGRESSION, "test_6")
@eeglab_test(STATCOND_CLASS, "unpaired2Anova")
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


@eeglab_test(STATCOND_REGRESSION, "test_10")
@eeglab_test(STATCOND_REGRESSION, "test_9")
@eeglab_test(STATCOND_REGRESSION, "test_8")
@eeglab_test(STATCOND_REGRESSION, "test_7")
@eeglab_test(STATCOND_CLASS, "pairedDimTTest")
def test_statcond_paired_t_is_invariant_across_feature_dimensions():
    _assert_dimension_invariance(paired="on", design="t")


@eeglab_test(STATCOND_REGRESSION, "test_14")
@eeglab_test(STATCOND_REGRESSION, "test_13")
@eeglab_test(STATCOND_REGRESSION, "test_12")
@eeglab_test(STATCOND_REGRESSION, "test_11")
@eeglab_test(STATCOND_CLASS, "unpairedDimTTest")
def test_statcond_unpaired_t_is_invariant_across_feature_dimensions():
    _assert_dimension_invariance(paired="off", design="t")


@eeglab_test(STATCOND_REGRESSION, "test_18")
@eeglab_test(STATCOND_REGRESSION, "test_17")
@eeglab_test(STATCOND_REGRESSION, "test_16")
@eeglab_test(STATCOND_REGRESSION, "test_15")
@eeglab_test(STATCOND_CLASS, "pairedDim1Anova")
def test_statcond_paired_one_way_is_invariant_across_feature_dimensions():
    _assert_dimension_invariance(paired="on", design="one-way")


@eeglab_test(STATCOND_REGRESSION, "test_22")
@eeglab_test(STATCOND_REGRESSION, "test_21")
@eeglab_test(STATCOND_REGRESSION, "test_20")
@eeglab_test(STATCOND_REGRESSION, "test_19")
@eeglab_test(STATCOND_CLASS, "unpairedDim1Anova")
def test_statcond_unpaired_one_way_is_invariant_across_feature_dimensions():
    _assert_dimension_invariance(paired="off", design="one-way")


@eeglab_test(STATCOND_REGRESSION, "test_26")
@eeglab_test(STATCOND_REGRESSION, "test_25")
@eeglab_test(STATCOND_REGRESSION, "test_24")
@eeglab_test(STATCOND_REGRESSION, "test_23")
@eeglab_test(STATCOND_CLASS, "pairedDim2Anova")
def test_statcond_paired_two_way_is_invariant_across_feature_dimensions():
    _assert_dimension_invariance(paired="on", design="two-way")


@eeglab_test(STATCOND_REGRESSION, "test_30")
@eeglab_test(STATCOND_REGRESSION, "test_29")
@eeglab_test(STATCOND_REGRESSION, "test_28")
@eeglab_test(STATCOND_REGRESSION, "test_27")
@eeglab_test(STATCOND_CLASS, "unpairedDim2Anova")
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


@eeglab_test(STATCOND_REGRESSION, "test_78")
@eeglab_test(STATCOND_REGRESSION, "test_77")
@eeglab_test(STATCOND_REGRESSION, "test_76")
@eeglab_test(STATCOND_REGRESSION, "test_75")
@eeglab_test(STATCOND_REGRESSION, "test_74")
@eeglab_test(STATCOND_REGRESSION, "test_73")
@eeglab_test(STATCOND_REGRESSION, "test_72")
@eeglab_test(STATCOND_REGRESSION, "test_71")
@eeglab_test(STATCOND_REGRESSION, "test_70")
@eeglab_test(STATCOND_REGRESSION, "test_69")
@eeglab_test(STATCOND_REGRESSION, "test_68")
@eeglab_test(STATCOND_REGRESSION, "test_67")
@eeglab_test(STATCOND_REGRESSION, "test_66")
@eeglab_test(STATCOND_REGRESSION, "test_65")
@eeglab_test(STATCOND_REGRESSION, "test_64")
@eeglab_test(STATCOND_REGRESSION, "test_63")
@eeglab_test(STATCOND_REGRESSION, "test_62")
@eeglab_test(STATCOND_REGRESSION, "test_61")
@eeglab_test(STATCOND_REGRESSION, "test_60")
@eeglab_test(STATCOND_REGRESSION, "test_59")
@eeglab_test(STATCOND_REGRESSION, "test_58")
@eeglab_test(STATCOND_REGRESSION, "test_57")
@eeglab_test(STATCOND_REGRESSION, "test_56")
@eeglab_test(STATCOND_REGRESSION, "test_55")
@eeglab_test(STATCOND_REGRESSION, "test_54")
@eeglab_test(STATCOND_REGRESSION, "test_53")
@eeglab_test(STATCOND_REGRESSION, "test_52")
@eeglab_test(STATCOND_REGRESSION, "test_51")
@eeglab_test(STATCOND_REGRESSION, "test_50")
@eeglab_test(STATCOND_REGRESSION, "test_49")
@eeglab_test(STATCOND_REGRESSION, "test_48")
@eeglab_test(STATCOND_REGRESSION, "test_47")
@eeglab_test(STATCOND_REGRESSION, "test_46")
@eeglab_test(STATCOND_REGRESSION, "test_45")
@eeglab_test(STATCOND_REGRESSION, "test_44")
@eeglab_test(STATCOND_REGRESSION, "test_43")
@eeglab_test(STATCOND_REGRESSION, "test_42")
@eeglab_test(STATCOND_REGRESSION, "test_41")
@eeglab_test(STATCOND_REGRESSION, "test_40")
@eeglab_test(STATCOND_REGRESSION, "test_39")
@eeglab_test(STATCOND_REGRESSION, "test_38")
@eeglab_test(STATCOND_REGRESSION, "test_37")
@eeglab_test(STATCOND_REGRESSION, "test_36")
@eeglab_test(STATCOND_REGRESSION, "test_35")
@eeglab_test(STATCOND_REGRESSION, "test_34")
@eeglab_test(STATCOND_REGRESSION, "test_33")
@eeglab_test(STATCOND_REGRESSION, "test_32")
@eeglab_test(STATCOND_REGRESSION, "test_31")
@eeglab_test(STATCOND_CLASS, "shuffleAndPermutation")
def test_statcond_resampling_preserves_assignment_invariants():
    _assert_resampling_suite()


@eeglab_test(STATCOND_WRAPPER, "test_test_statcond")
def test_legacy_statcond_suite_workflow_is_preserved():
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
