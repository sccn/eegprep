from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import numpy.testing as npt
import pytest
from scipy import io as scipy_io
from scipy import stats as scipy_stats

from tests.eeglab_tests import assert_matlab_near, eeglab_test
from tests.test_statcond_eeglab_tests import _assert_derived_p
from eegprep.functions.statistics import (
    TwoWayEffects,
    anova1_cell,
    anova2_cell,
    anova2rm_cell,
    concatdata,
    corrcoef_cell,
    fdr,
    stat_surrogate_ci,
    stat_surrogate_pvals,
    statcond,
    surrogdistrib,
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


@eeglab_test(
    "unittesting_sigprocfunc/fdr/sigprocfunc_fdr_wrapperTest.m",
    "test_test_fdr",
)
def test_reference_fdr_original_correlated_random_data(eeglab_backend, request):
    data = np.random.default_rng().standard_normal((30, 4))
    data[:, 3] = data.sum(axis=1)
    if request.config.getoption("--eeglab-backend") == "matlab":
        _, probabilities = eeglab_backend("corrcoef", data, nargout=2)
    else:
        probabilities = scipy_stats.pearsonr(data.T[:, None, :], data.T[None, :, :], axis=-1).pvalue
    for options in ((), (0.8,), (0.05,), (0.5,), (0.5, "nonParametric")):
        eeglab_backend("fdr", probabilities, *options, nargout=2)


def test_fdr_uses_finite_pvalues_for_threshold_denominator():
    pvals = np.array([0.01, 0.03, np.nan, np.inf])

    result = fdr(pvals, 0.05)

    assert result.threshold == pytest.approx(0.03)
    npt.assert_array_equal(result.mask, [True, True, False, False])


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


@eeglab_test("unittesting_sigprocfunc/concatdata/sigprocfunc_concatdata_wrapperTest.m", "test_test_concatdata")
def test_reference_concatdata_original_arrays(eeglab_backend):
    rng = np.random.default_rng()
    first, second = rng.random((32, 100)), rng.random((32, 1000))
    data = np.empty((1, 2), dtype=object)
    data[0, 0], data[0, 1] = first, second
    result, lengths, _dimensions = eeglab_backend("concatdata", data, nargout=3)
    assert_matlab_near([result.shape], [[32.0, 1100.0]])
    assert_matlab_near(result[:, :100], first)
    assert_matlab_near(result[:, 100:1100], second)
    assert_matlab_near(lengths, [[0.0, 100.0, 1100.0]])


def test_corrcoef_and_concatdata_contracts():
    first = np.array([[1, 2, 3, 4], [1, 3, 5, 7]], dtype=float)
    second = np.array([[4, 3, 2, 1], [2, 4, 6, 8]], dtype=float)

    npt.assert_allclose(corrcoef_cell(first, second), [-1, 1])
    concatenated = concatdata([first, second[:, :2]])

    assert concatenated.data.shape == (2, 6)
    npt.assert_array_equal(concatenated.lengths, [0, 4, 6])
    assert concatenated.grid_shape == (1, 2)


def _unpaired_statcond_sample() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    return rng.normal(size=(3, 12)), rng.normal(loc=0.2, scale=1.4, size=(3, 9))


def test_statcond_unpaired_default_matches_explicit_inhomogenous_variance():
    first, second = _unpaired_statcond_sample()
    default = statcond([first, second], paired="off")
    explicit = statcond([first, second], paired="off", variance="inhomogenous")
    pooled = statcond([first, second], paired="off", variance="homogenous")

    npt.assert_array_equal(default.stat, explicit.stat)
    npt.assert_array_equal(np.asarray(default.df), np.asarray(explicit.df))
    npt.assert_array_equal(default.pvalue, explicit.pvalue)
    assert not np.allclose(default.stat, pooled.stat)


def test_statcond_arraycomp_off_streams_the_same_seeded_statistics():
    rng = np.random.default_rng(30)
    data = [rng.normal(size=(3, 8)), rng.normal(size=(3, 8))]

    batched = statcond(data, method="perm", paired="on", naccu=12, rng=42, arraycomp="on")
    iterative = statcond(data, method="perm", paired="on", naccu=12, rng=42, arraycomp="off")

    npt.assert_array_equal(iterative.surrogate, batched.surrogate)
    npt.assert_array_equal(iterative.pvalue, batched.pvalue)


def test_matlab_twister_matches_rng_default_draws():
    from eegprep.functions.statistics._matlab_rand import MatlabTwister

    stream = MatlabTwister()
    expected = np.array(
        [
            0.8147236863931789,
            0.9057919370756192,
            0.12698681629350606,
            0.9133758561390194,
            0.6323592462254095,
        ]
    )
    npt.assert_array_equal(np.array([stream.rand() for _ in range(5)]), expected)


def test_statcond_accepts_matlab_name_value_pairs():
    first = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    second = first + 1.0
    keyword = statcond([first, second], mode="param", verbose="off", paired="on")
    positional = statcond([first, second], "mode", "param", "verbose", "off", "paired", "on")

    npt.assert_array_equal(keyword.stat, positional.stat)
    npt.assert_array_equal(keyword.pvalue, positional.pvalue)
    assert keyword.df == positional.df
    with pytest.raises(ValueError, match="unsupported option"):
        statcond([first, second], "notanoption", 1)


def test_statcond_unpaired_matlab_cells_follow_anova2_order():
    grid = (
        (np.array([1.0, 2, 4, 3, 5]), np.array([2.0, 2, 3, 5, 4]), np.array([3.0, 4, 2, 5, 6])),
        (np.array([2.0, 3, 1, 4, 5]), np.array([4.0, 2, 5, 3, 6]), np.array([5.0, 6, 3, 7, 4])),
    )
    result = statcond(grid, method="param", paired="off")

    assert result.matlab_stat.shape == (1, 3)
    assert result.matlab_stat[0, 0].dtype == np.float32
    npt.assert_array_equal(
        result.matlab_stat[0, 0],
        np.asarray(result.stat.columns, dtype=np.float32).reshape(1, 1),
    )
    npt.assert_array_equal(
        result.matlab_stat[0, 1],
        np.asarray(result.stat.rows, dtype=np.float32).reshape(1, 1),
    )
    npt.assert_array_equal(
        result.matlab_stat[0, 2],
        np.asarray(result.stat.interaction, dtype=np.float32).reshape(1, 1),
    )
    assert result.matlab_df[0, 0].shape == (1, 2)
    assert result.matlab_df[0, 0].dtype == np.float64


def test_statcond_cluster_uses_test_degrees_of_freedom():
    n_features = 4
    n_cases = 11
    alpha = 0.25
    data = [np.zeros((n_features, n_cases)), np.zeros((n_features, n_cases))]
    real_critical = scipy_stats.t.ppf(1 - alpha / 2, n_cases - 1)
    feature_count_critical = scipy_stats.t.ppf(1 - alpha / 2, n_features)
    statistic = (real_critical + feature_count_critical) / 2
    surrogate = np.zeros((n_features, 4))
    surrogate[0, 1] = statistic
    surrogate[:2, 2] = statistic
    surrogate[:, 3] = statistic

    clustered = statcond(
        data,
        method="perm",
        paired="on",
        cluster="on",
        alpha=alpha,
        surrog=surrogate,
        stats=np.array([0.0, statistic, statistic, 0.0]),
    )
    isolated = statcond(
        data,
        method="perm",
        paired="on",
        cluster="on",
        alpha=alpha,
        surrog=surrogate,
        stats=np.array([statistic, 0.0, 0.0, 0.0]),
    )

    npt.assert_array_equal(clustered.pvalue, [1.0, 0.0, 0.0, 1.0])
    npt.assert_array_equal(isolated.pvalue, [1.0, 1.0, 1.0, 1.0])


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


@pytest.mark.matlab
def test_matlab_statcond_unpaired_default_matches_eeglab(matlab_statistics_engine, tmp_path):
    stats_dir = _eeglab_statistics_reference_dir()
    assert stats_dir is not None
    guifunc = stats_dir.parent / "guifunc"
    if not (guifunc / "finputcheck.m").is_file():
        pytest.fail(f"statcond default parity needs {guifunc / 'finputcheck.m'}")
    matlab_statistics_engine.addpath(str(guifunc), nargout=0)

    first, second = _unpaired_statcond_sample()
    matlab = _run_matlab_statistics(
        matlab_statistics_engine,
        tmp_path,
        {"first": first, "second": second},
        (
            "if exist('finputcheck','file') ~= 2, "
            "error('finputcheck must be on the path so statcond defaults to inhomogenous'); "
            "end; "
            "[defaultT, defaultDf, defaultP] = statcond({first, second}, 'paired', 'off'); "
            "[pooledT, pooledDf, pooledP] = statcond({first, second}, 'paired', 'off', "
            "'variance', 'homogenous');"
        ),
        ["defaultT", "defaultDf", "defaultP", "pooledT", "pooledDf", "pooledP"],
    )
    default = statcond([first, second], paired="off")
    pooled = statcond([first, second], paired="off", variance="homogenous")
    # Both samples enter the means and the variance, so t and df use
    # (n1 + n2) * float64 eps. p uses that times the two-tailed condition
    # number on the MATLAB reference.
    stat_rtol = (first.shape[-1] + second.shape[-1]) * float(np.finfo(np.float64).eps)
    for actual, prefix in ((default, "default"), (pooled, "pooled")):
        npt.assert_allclose(np.ravel(actual.stat), np.ravel(matlab[f"{prefix}T"]), rtol=stat_rtol, atol=0)
        npt.assert_allclose(np.ravel(actual.df), np.ravel(matlab[f"{prefix}Df"]), rtol=stat_rtol, atol=0)
        _assert_derived_p(
            np.ravel(actual.pvalue),
            np.ravel(matlab[f"{prefix}T"]),
            np.ravel(matlab[f"{prefix}Df"]),
            np.ravel(matlab[f"{prefix}P"]),
            stat_rtol=stat_rtol,
        )
