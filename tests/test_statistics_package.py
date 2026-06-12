from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

import numpy as np
import numpy.testing as npt
import pytest
from scipy import io as scipy_io
from scipy import stats as scipy_stats

from eegprep.functions.statistics import (
    TwoWayEffects,
    anova1_cell,
    anova1rm_cell,
    anova2_cell,
    anova2rm_cell,
    concatdata,
    corrcoef_cell,
    fdr,
    stat_surrogate_ci,
    stat_surrogate_pvals,
    statcond,
    surrogdistrib,
    teststat as statistics_teststat,
    ttest2_cell,
    ttest_cell,
)
from tests.fixtures import matlab_engine_available


ANOVA_PARITY_RTOL = 1e-4
ANOVA_PARITY_ATOL = 1e-5


def test_fdr_matches_bh_and_by_thresholds():
    pvals = np.array([0.001, 0.01, 0.03, 0.2])

    bh = fdr(pvals, 0.05)
    by = fdr(pvals, 0.05, fdr_type="nonParametric")

    assert bh.threshold == pytest.approx(0.03)
    npt.assert_array_equal(bh.mask, [True, True, True, False])
    assert by.threshold == pytest.approx(0.01)
    npt.assert_array_equal(by.mask, [True, True, False, False])


def test_fdr_uses_finite_pvalues_for_threshold_denominator():
    pvals = np.array([0.01, 0.03, np.nan, np.inf])

    result = fdr(pvals, 0.05)

    assert result.threshold == pytest.approx(0.03)
    npt.assert_array_equal(result.mask, [True, True, False, False])


def test_fdr_no_finite_pvalues_returns_empty_mask():
    result = fdr(np.array([np.nan, np.inf]), 0.05)

    assert result.threshold == 0
    npt.assert_array_equal(result.mask, [False, False])


def test_surrogate_pvals_and_ci_use_last_axis():
    distribution = np.array([[1, 2, 3, 4], [4, 5, 6, 7]], dtype=float)
    observed = np.array([3, 5], dtype=float)
    tied_distribution = np.array([[1, 2, 2, 3]], dtype=float)
    tied_observed = np.array([2], dtype=float)

    npt.assert_allclose(stat_surrogate_pvals(distribution, observed, "right"), [0.5, 0.75])
    npt.assert_allclose(stat_surrogate_pvals(distribution, observed, "left"), [0.75, 0.5])
    npt.assert_allclose(stat_surrogate_pvals(distribution, observed, "both"), [1.0, 1.0])
    npt.assert_allclose(stat_surrogate_pvals(tied_distribution, tied_observed, "both"), [1.0])

    ci = stat_surrogate_ci(np.arange(1, 11, dtype=float), alpha=0.2, tail="both")
    npt.assert_allclose(ci, [2, 9])
    upper = stat_surrogate_ci(np.arange(1, 11, dtype=float), alpha=0.2, tail="upper")
    npt.assert_allclose(upper, [5.5, 8])


def test_ttest_helpers_match_scipy_statistics():
    rng = np.random.default_rng(1)
    first = rng.normal(size=(4, 12))
    second = first + rng.normal(loc=0.25, scale=0.5, size=(4, 12))
    third = rng.normal(loc=0.1, scale=1.2, size=(4, 9))

    paired_t, paired_df = ttest_cell(first, second)
    scipy_paired = scipy_stats.ttest_rel(first, second, axis=-1).statistic
    unequal_t, unequal_df = ttest2_cell(first, third, variance="inhomogenous")
    scipy_unequal = scipy_stats.ttest_ind(first, third, axis=-1, equal_var=False)
    pooled_t, pooled_df = ttest2_cell(first, third)
    scipy_pooled = scipy_stats.ttest_ind(first, third, axis=-1, equal_var=True)

    npt.assert_allclose(paired_t, scipy_paired)
    assert paired_df == 11
    npt.assert_allclose(unequal_t, scipy_unequal.statistic)
    npt.assert_allclose(unequal_df, scipy_unequal.df)
    npt.assert_allclose(pooled_t, scipy_pooled.statistic)
    assert pooled_df == 19


def test_corrcoef_and_concatdata_contracts():
    first = np.array([[1, 2, 3, 4], [1, 3, 5, 7]], dtype=float)
    second = np.array([[4, 3, 2, 1], [2, 4, 6, 8]], dtype=float)

    npt.assert_allclose(corrcoef_cell(first, second), [-1, 1])
    concatenated = concatdata([first, second[:, :2]])

    assert concatenated.data.shape == (2, 6)
    npt.assert_array_equal(concatenated.lengths, [0, 4, 6])
    assert concatenated.grid_shape == (1, 2)


def test_anova1_cell_matches_scipy_f_oneway():
    groups = [
        np.array([[1, 2, 1, 3], [2, 2, 3, 4]], dtype=float),
        np.array([[2, 3, 4, 5], [3, 4, 5, 6]], dtype=float),
        np.array([[4, 4, 5, 6], [6, 6, 7, 8]], dtype=float),
    ]

    f_values, df = anova1_cell(groups)
    expected = np.array([scipy_stats.f_oneway(*(group[row] for group in groups)).statistic for row in range(2)])

    npt.assert_allclose(f_values, expected)
    assert df == (2, 9)


def test_repeated_measure_anova_helpers_return_factor_results():
    grid = (
        (
            np.array([[1, 2, 3, 4], [2, 2, 4, 5]], dtype=float),
            np.array([[2, 3, 5, 6], [3, 4, 6, 7]], dtype=float),
        ),
        (
            np.array([[3, 3, 4, 5], [4, 5, 5, 6]], dtype=float),
            np.array([[4, 6, 6, 8], [6, 7, 8, 9]], dtype=float),
        ),
    )

    one_way, one_way_df = anova1rm_cell(grid[0])
    two_way = anova2rm_cell(grid)
    unpaired = anova2_cell(grid)

    assert one_way.shape == (2,)
    assert one_way_df == (1, 3)
    assert two_way.rows.shape == two_way.columns.shape == two_way.interaction.shape == (2,)
    assert two_way.df_rows == (1, 3)
    assert two_way.df_columns == (1, 3)
    assert two_way.df_interaction == (1, 3)
    assert unpaired.df_interaction == (1, 12)


def test_statcond_selects_tests_and_returns_named_two_way_effects():
    rng = np.random.default_rng(2)
    first = rng.normal(size=(3, 10))
    second = first + 0.2
    third = rng.normal(size=(3, 8))
    grid = (
        (rng.normal(size=(3, 9)), rng.normal(loc=0.2, size=(3, 9))),
        (rng.normal(loc=0.1, size=(3, 9)), rng.normal(loc=0.4, size=(3, 9))),
    )

    paired = statcond([first, second], paired="on")
    unpaired = statcond([first, third], paired="auto")
    two_way = statcond(grid, paired="on")

    assert paired.paired is True
    assert paired.stat.shape == (3,)
    assert unpaired.paired is False
    assert isinstance(two_way.stat, TwoWayEffects)
    assert two_way.pvalue.interaction.shape == (3,)


def test_statcond_unpaired_default_uses_eeglab_homogenous_variance():
    first = np.array([[1.0, 2.0, 3.0, 4.0], [1.5, 1.7, 2.2, 2.9]])
    second = np.array([[2.0, 5.0, 8.0, 11.0, 14.0], [1.4, 2.1, 2.4, 4.8, 7.0]])

    result = statcond([first, second], paired="off")
    pooled_stat, pooled_df = ttest2_cell(first, second, variance="homogenous")
    welch_stat, welch_df = ttest2_cell(first, second, variance="inhomogenous")

    npt.assert_allclose(result.stat, pooled_stat)
    assert result.df == pooled_df
    assert not np.array_equal(result.df, welch_df)
    assert not np.allclose(pooled_stat, welch_stat)


def test_nonparametric_statcond_and_surrogdistrib_are_seeded():
    rng = np.random.default_rng(3)
    first = rng.normal(size=(2, 8))
    second = first + rng.normal(size=(2, 8))

    first_result = statcond([first, second], method="perm", naccu=20, rng=10)
    second_result = statcond([first, second], method="perm", naccu=20, rng=10)
    surrogates = surrogdistrib([first, second], method="bootstrap", naccu=3, rng=10)

    npt.assert_allclose(first_result.surrogate, second_result.surrogate)
    npt.assert_allclose(first_result.pvalue, second_result.pvalue)
    assert len(surrogates) == 3
    assert all(sample[0][0].shape == first.shape for sample in surrogates)


def test_statcond_supplied_surrogates_return_alpha_ci_and_mask():
    surrogate = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 5.0, 6.0, 7.0]])
    observed = np.array([3.5, 7.5])

    result = statcond([np.ones((2, 4)), np.ones((2, 4))], surrog=surrogate, stats=observed, alpha=0.25, tail="right")

    npt.assert_allclose(result.pvalue, stat_surrogate_pvals(surrogate, observed, "right"))
    npt.assert_allclose(result.ci, stat_surrogate_ci(surrogate, alpha=0.25, tail="upper"))
    npt.assert_array_equal(result.mask, result.pvalue < 0.25)


def test_nonparametric_two_way_statcond_matches_manual_surrogate_assembly():
    rng = np.random.default_rng(4)
    grid = (
        (rng.normal(size=(2, 7)), rng.normal(loc=0.2, size=(2, 7))),
        (rng.normal(loc=0.4, size=(2, 7)), rng.normal(loc=0.6, size=(2, 7))),
    )

    result = statcond(grid, paired="on", method="perm", naccu=8, rng=24)
    assert isinstance(result.stat, TwoWayEffects)
    assert isinstance(result.surrogate, TwoWayEffects)
    assert isinstance(result.pvalue, TwoWayEffects)

    manual_rows = []
    manual_columns = []
    manual_interactions = []
    for sample in surrogdistrib(grid, method="perm", pairing="on", naccu=8, rng=24):
        effects = anova2rm_cell(sample).as_effects()
        manual_rows.append(effects.rows)
        manual_columns.append(effects.columns)
        manual_interactions.append(effects.interaction)

    expected_surrogates = TwoWayEffects(
        np.stack(manual_rows, axis=-1),
        np.stack(manual_columns, axis=-1),
        np.stack(manual_interactions, axis=-1),
    )

    npt.assert_allclose(result.surrogate.rows, expected_surrogates.rows)
    npt.assert_allclose(result.surrogate.columns, expected_surrogates.columns)
    npt.assert_allclose(result.surrogate.interaction, expected_surrogates.interaction)
    npt.assert_allclose(
        result.pvalue.interaction,
        stat_surrogate_pvals(expected_surrogates.interaction, result.stat.interaction, "one"),
    )


def test_invalid_inputs_fail_clearly():
    with pytest.raises(ValueError, match="identical shapes"):
        ttest_cell(np.ones((2, 4)), np.ones((2, 5)))
    with pytest.raises(ValueError, match="same number of cases"):
        statcond([np.ones((2, 4)), np.ones((2, 5))], paired="on")
    with pytest.raises(TypeError, match="numeric"):
        fdr(np.array(["bad"], dtype=object), 0.05)
    with pytest.raises(ValueError, match="fdr_type"):
        fdr(np.array([0.01, 0.02]), 0.05, fdr_type="unknown")
    with pytest.raises(ValueError, match="observed shape"):
        stat_surrogate_pvals(np.ones((2, 4)), np.ones((3,)), "both")


def test_teststat_smoke_helper_runs_deterministic_checks():
    result = statistics_teststat(seed=0)

    assert sorted(result) == ["one_way_f_mean", "paired_t_mean", "two_way_interaction_mean"]


def test_statistics_package_exports_remain_functions_after_submodule_imports():
    import eegprep.functions.statistics as statistics

    fdr_module = importlib.import_module("eegprep.functions.statistics.fdr")
    statcond_module = importlib.import_module("eegprep.functions.statistics.statcond")
    core_module = importlib.import_module("eegprep.functions.statistics._core")

    assert statistics.fdr is fdr_module.fdr
    assert statistics.statcond is statcond_module.statcond
    assert statistics.FDRResult is fdr_module.FDRResult
    assert statistics.StatcondResult is statcond_module.StatcondResult
    assert core_module.fdr is fdr_module.fdr


@pytest.fixture(scope="module")
def matlab_statistics_engine():
    stats_dir = _eeglab_statistics_reference_dir()
    if stats_dir is None:
        pytest.skip("EEGLAB statistics reference checkout not available; set EEGPREP_EEGLAB_ROOT")
    if not matlab_engine_available():
        pytest.skip("MATLAB engine not available or skipped")
    try:
        import matlab.engine

        engine = matlab.engine.start_matlab()
    except Exception as exc:  # pragma: no cover - depends on local MATLAB setup
        pytest.skip(f"MATLAB engine not available: {exc}")

    engine.addpath(str(stats_dir), nargout=0)
    try:
        yield engine
    finally:
        engine.quit()


def _eeglab_statistics_reference_dir() -> Path | None:
    repo_root = Path(__file__).resolve().parents[1]
    candidates: list[Path] = []
    if os.environ.get("EEGPREP_EEGLAB_ROOT"):
        candidates.append(Path(os.environ["EEGPREP_EEGLAB_ROOT"]).expanduser() / "functions" / "statistics")
    candidates.extend(
        [
            repo_root / "src" / "eegprep" / "eeglab" / "functions" / "statistics",
            repo_root.parent / "eeglab" / "functions" / "statistics",
        ]
    )
    for candidate in candidates:
        if (candidate / "fdr.m").is_file() and (candidate / "ttest_cell.m").is_file():
            return candidate
    return None


def _run_matlab_statistics(
    engine: Any,
    tmp_path: Path,
    inputs: dict[str, Any],
    body: str,
    outputs: list[str],
) -> dict[str, Any]:
    input_path = tmp_path / "matlab_statistics_input.mat"
    output_path = tmp_path / "matlab_statistics_output.mat"
    scipy_io.savemat(input_path, inputs)
    save_outputs = ", ".join(f"'{output}'" for output in outputs)
    engine.eval(
        f"load('{input_path.as_posix()}'); {body}; save('-mat', '{output_path.as_posix()}', {save_outputs});",
        nargout=0,
    )
    return scipy_io.loadmat(output_path, squeeze_me=True)


@pytest.mark.matlab
def test_matlab_parity_for_fdr_and_surrogate_helpers(matlab_statistics_engine, tmp_path):
    pvals = np.array([0.001, 0.01, 0.03, 0.2], dtype=float)
    distribution = np.array([[1, 2, 3, 4], [4, 5, 6, 7]], dtype=float)
    observed = np.array([[3], [5]], dtype=float)

    matlab = _run_matlab_statistics(
        matlab_statistics_engine,
        tmp_path,
        {"pvals": pvals, "distribution": distribution, "observed": observed},
        (
            "[pID, pMask] = fdr(pvals, 0.05); "
            "pRight = stat_surrogate_pvals(distribution, observed, 'right'); "
            "pBoth = stat_surrogate_pvals(distribution, observed, 'both'); "
            "ciBoth = stat_surrogate_ci(distribution, 0.2, 'both');"
        ),
        ["pID", "pMask", "pRight", "pBoth", "ciBoth"],
    )

    python_fdr = fdr(pvals, 0.05)
    npt.assert_allclose(python_fdr.threshold, matlab["pID"])
    npt.assert_array_equal(python_fdr.mask, matlab["pMask"].astype(bool).reshape(pvals.shape))
    npt.assert_allclose(stat_surrogate_pvals(distribution, observed[:, 0], "right"), matlab["pRight"])
    npt.assert_allclose(stat_surrogate_pvals(distribution, observed[:, 0], "both"), matlab["pBoth"])
    npt.assert_allclose(stat_surrogate_ci(distribution, 0.2, "both"), matlab["ciBoth"])


@pytest.mark.matlab
def test_matlab_parity_for_ttests_corrcoef_and_anova(matlab_statistics_engine, tmp_path):
    first = np.array([[1, 2, 3, 4, 5], [2, 3, 5, 7, 11]], dtype=float)
    second = np.array([[2, 2, 4, 4, 6], [1, 4, 4, 8, 10]], dtype=float)
    third = np.array([[0, 1, 1, 2, 3], [2, 2, 3, 3, 4]], dtype=float)
    fourth = np.array([[3, 4, 5, 7, 8], [4, 4, 6, 8, 9]], dtype=float)

    matlab = _run_matlab_statistics(
        matlab_statistics_engine,
        tmp_path,
        {"first": first, "second": second, "third": third, "fourth": fourth},
        (
            "[pairedT, pairedDf] = ttest_cell(first, second); "
            "[welchT, welchDf] = ttest2_cell(first, third, 'inhomogenous'); "
            "corrVals = corrcoef_cell(first, second); "
            "oneway = {first, second, fourth}; "
            "[anovaF, anovaDf] = anova1_cell(oneway); "
            "twoway = {first, second; third(:,1:5), fourth}; "
            "[fC, fR, fI, dfC, dfR, dfI] = anova2_cell(twoway); "
            "[rmR, rmC, rmI, rmDfR, rmDfC, rmDfI] = anova2rm_cell(twoway);"
        ),
        [
            "pairedT",
            "pairedDf",
            "welchT",
            "welchDf",
            "corrVals",
            "anovaF",
            "anovaDf",
            "fC",
            "fR",
            "fI",
            "dfC",
            "dfR",
            "dfI",
            "rmR",
            "rmC",
            "rmI",
            "rmDfR",
            "rmDfC",
            "rmDfI",
        ],
    )

    paired_t, paired_df = ttest_cell(first, second)
    welch_t, welch_df = ttest2_cell(first, third, variance="inhomogenous")
    one_way_f, one_way_df = anova1_cell([first, second, fourth])
    two_way = anova2_cell(((first, second), (third[:, :5], fourth)))
    repeated = anova2rm_cell(((first, second), (third[:, :5], fourth)))

    npt.assert_allclose(paired_t, matlab["pairedT"])
    npt.assert_allclose(paired_df, matlab["pairedDf"])
    npt.assert_allclose(welch_t, matlab["welchT"])
    npt.assert_allclose(welch_df, matlab["welchDf"])
    npt.assert_allclose(corrcoef_cell(first, second), matlab["corrVals"])
    npt.assert_allclose(one_way_f, matlab["anovaF"], rtol=ANOVA_PARITY_RTOL, atol=ANOVA_PARITY_ATOL)
    npt.assert_allclose(one_way_df, matlab["anovaDf"])
    npt.assert_allclose(two_way.columns, matlab["fC"], rtol=ANOVA_PARITY_RTOL, atol=ANOVA_PARITY_ATOL)
    npt.assert_allclose(two_way.rows, matlab["fR"], rtol=ANOVA_PARITY_RTOL, atol=ANOVA_PARITY_ATOL)
    npt.assert_allclose(two_way.interaction, matlab["fI"], rtol=ANOVA_PARITY_RTOL, atol=ANOVA_PARITY_ATOL)
    npt.assert_allclose(two_way.df_columns, matlab["dfC"])
    npt.assert_allclose(two_way.df_rows, matlab["dfR"])
    npt.assert_allclose(two_way.df_interaction, matlab["dfI"])
    npt.assert_allclose(repeated.rows, matlab["rmR"], rtol=ANOVA_PARITY_RTOL, atol=ANOVA_PARITY_ATOL)
    npt.assert_allclose(repeated.columns, matlab["rmC"], rtol=ANOVA_PARITY_RTOL, atol=ANOVA_PARITY_ATOL)
    npt.assert_allclose(repeated.interaction, matlab["rmI"], rtol=ANOVA_PARITY_RTOL, atol=ANOVA_PARITY_ATOL)
    npt.assert_allclose(repeated.df_rows, matlab["rmDfR"])
    npt.assert_allclose(repeated.df_columns, matlab["rmDfC"])
    npt.assert_allclose(repeated.df_interaction, matlab["rmDfI"])
