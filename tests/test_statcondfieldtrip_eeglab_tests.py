"""Ports of the maintained EEGLAB ``statcondfieldtrip`` wrapper test."""

from __future__ import annotations

import importlib
from collections.abc import Sequence

import numpy as np
import pytest
from scipy import stats as scipy_stats

from eegprep.functions.statistics import StatcondFieldtripResult, statcondfieldtrip
from tests.eeglab_tests import eeglab_test


STATCONDFIELDTRIP_SCRIPT = "unittesting_statistics/statcondfieldtrip/test_statcondfieldtrip.m"
STATCONDFIELDTRIP_WRAPPER = "unittesting_statistics/statcondfieldtrip/statistics_statcondfieldtrip_wrapperTest.m"


def _reference_conditions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(941)
    first = rng.normal(size=10)
    second = rng.normal(loc=0.5, size=10)
    third = rng.normal(loc=0.2, size=10)
    return first, second, third


def _assert_active_vector_calls() -> None:
    first, second, third = _reference_conditions()

    paired = statcondfieldtrip([first, second], paired="on", mode="param", method="analytic")
    paired_reference = scipy_stats.ttest_rel(first, second)
    np.testing.assert_allclose(paired.stat, paired_reference.statistic, rtol=1e-13, atol=1e-13)
    assert paired.df == 9
    np.testing.assert_allclose(paired.pvalue, paired_reference.pvalue, rtol=1e-13, atol=1e-13)

    unpaired = statcondfieldtrip(
        [first, second],
        paired="off",
        mode="param",
        method="analytic",
        variance="homogenous",
    )
    unpaired_reference = scipy_stats.ttest_ind(first, second, equal_var=True)
    np.testing.assert_allclose(unpaired.stat, unpaired_reference.statistic, rtol=1e-13, atol=1e-13)
    assert unpaired.df == 18
    np.testing.assert_allclose(unpaired.pvalue, unpaired_reference.pvalue, rtol=1e-13, atol=1e-13)

    one_way = statcondfieldtrip([first, second, third], paired="off", mode="param", method="analytic")
    one_way_reference = scipy_stats.f_oneway(first, second, third)
    np.testing.assert_allclose(one_way.stat, one_way_reference.statistic, rtol=1e-13, atol=1e-13)
    assert one_way.df == (2, 27)
    np.testing.assert_allclose(one_way.pvalue, one_way_reference.pvalue, rtol=1e-13, atol=1e-13)


def _dimensional_conditions() -> tuple[tuple[tuple[np.ndarray, ...], tuple[int, ...]], ...]:
    rng = np.random.default_rng(942)
    base = tuple(rng.normal(loc=offset, size=10) for offset in (0.0, 0.5, 0.2))
    matrix = tuple(rng.normal(loc=offset, size=(8, 10)) for offset in (0.0, 0.5, 0.2))
    cube = tuple(rng.normal(loc=offset, size=(4, 8, 10)) for offset in (0.0, 0.5, 0.2))
    for condition_index in range(3):
        matrix[condition_index][3, :] = base[condition_index]
        cube[condition_index][1, 3, :] = base[condition_index]
    return ((base, ()), (matrix, (3,)), (cube, (1, 3)))


def _value_at(value: np.ndarray | float, index: tuple[int, ...]) -> float:
    values = np.asarray(value)
    return float(values[index] if index else values)


def _assert_active_dimension_calls(*, paired: bool, one_way: bool) -> None:
    results: list[tuple[StatcondFieldtripResult, tuple[int, ...]]] = []
    for conditions, index in _dimensional_conditions():
        data: Sequence[np.ndarray] = conditions if one_way else conditions[:2]
        result = statcondfieldtrip(data, paired=paired, method="analytic")
        assert np.asarray(result.stat).shape == conditions[0].shape[:-1]
        assert result.pvalue.shape == conditions[0].shape[:-1]
        assert result.mask.shape == conditions[0].shape[:-1]
        results.append((result, index))

    baseline, _ = results[0]
    for result, index in results[1:]:
        np.testing.assert_allclose(_value_at(result.stat, index), baseline.stat, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(_value_at(result.pvalue, index), baseline.pvalue, rtol=1e-13, atol=1e-13)
        assert result.df == baseline.df


@eeglab_test(STATCONDFIELDTRIP_WRAPPER, "test_test_statcondfieldtrip")
@eeglab_test(STATCONDFIELDTRIP_SCRIPT, "test_statcondfieldtrip")
def test_current_statcondfieldtrip_wrapper_executes_its_scientific_intent():
    # The MATLAB script returns before these calls because its `exist('kmean')`
    # guard contains a typo. Execute the active body rather than porting a no-op.
    _assert_active_vector_calls()
    _assert_active_dimension_calls(paired=True, one_way=False)
    _assert_active_dimension_calls(paired=False, one_way=False)
    _assert_active_dimension_calls(paired=False, one_way=True)


def test_statcondfieldtrip_preserves_features_with_a_nonfinal_case_axis():
    first, second, _ = _reference_conditions()
    data = [np.stack([first, first + 0.1], axis=1), np.stack([second, second - 0.1], axis=1)]

    result = statcondfieldtrip(data, paired="on", axis=0)

    assert np.asarray(result.stat).shape == (2,)
    expected = scipy_stats.ttest_rel(data[0], data[1], axis=0)
    np.testing.assert_allclose(result.stat, expected.statistic, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(result.pvalue, expected.pvalue, rtol=1e-13, atol=1e-13)


def test_statcondfieldtrip_applies_standard_pointwise_corrections():
    target_pvalues = np.array([0.2, 0.04, 0.02, 0.001])
    residual = np.arange(12, dtype=float)
    residual = (residual - np.mean(residual)) / np.std(residual, ddof=1)
    offsets = scipy_stats.t.isf(target_pvalues / 2, df=11) / np.sqrt(12)
    first = np.zeros((4, 12))
    second = residual[np.newaxis, :] + offsets[:, np.newaxis]
    raw = statcondfieldtrip([first, second], paired="on")
    np.testing.assert_allclose(raw.pvalue, target_pvalues, rtol=1e-10, atol=1e-13)

    bonferroni = statcondfieldtrip([first, second], paired="on", mcorrect="bonferoni")
    np.testing.assert_array_equal(bonferroni.pvalue, raw.pvalue)
    np.testing.assert_array_equal(bonferroni.mask, raw.pvalue <= 0.05 / 4)
    assert bonferroni.mcorrect == "bonferroni"

    order = np.argsort(raw.pvalue)
    ordered = raw.pvalue[order]
    holm_accepted = np.zeros(4, dtype=bool)
    for rank, pvalue in enumerate(ordered):
        if pvalue > 0.05 / (4 - rank):
            break
        holm_accepted[rank] = True
    holm_expected = np.zeros(4, dtype=bool)
    holm_expected[order] = holm_accepted
    holm = statcondfieldtrip([first, second], paired="on", mcorrect="holms")
    np.testing.assert_array_equal(holm.pvalue, raw.pvalue)
    np.testing.assert_array_equal(holm.mask, holm_expected)
    assert holm.mcorrect == "holm"

    accepted = ordered <= np.arange(1, 5) / 4 * 0.05
    threshold = ordered[np.flatnonzero(accepted).max()] if np.any(accepted) else 0.0
    fdr_expected = raw.pvalue <= threshold
    fdr_result = statcondfieldtrip([first, second], paired="on", mcorrect="fdr")
    np.testing.assert_array_equal(fdr_result.pvalue, raw.pvalue)
    np.testing.assert_array_equal(fdr_result.mask, fdr_expected)
    assert np.count_nonzero(raw.mask) == 3
    assert np.count_nonzero(bonferroni.mask) == 1
    assert np.count_nonzero(holm.mask) == 1
    assert np.count_nonzero(fdr_result.mask) == 2


def test_statcondfieldtrip_montecarlo_max_correction_is_seeded_and_familywise():
    rng = np.random.default_rng(944)
    first = rng.normal(size=(5, 10))
    second = first + np.linspace(0.0, 0.8, 5)[:, np.newaxis] + rng.normal(scale=0.5, size=(5, 10))

    result = statcondfieldtrip(
        [first, second],
        paired="on",
        method="montecarlo",
        naccu=128,
        mcorrect="max",
        rng=61,
    )
    repeated = statcondfieldtrip(
        [first, second],
        paired="on",
        method="permutation",
        naccu=128,
        mcorrect="max",
        rng=61,
    )

    assert result.method == "montecarlo"
    assert result.surrogate.shape == (5, 128)
    np.testing.assert_array_equal(result.pvalue, repeated.pvalue)
    null_maximum = np.max(np.abs(result.surrogate), axis=0)
    expected = (np.sum(null_maximum >= np.abs(result.stat)[:, np.newaxis], axis=-1) + 1) / 129
    np.testing.assert_array_equal(result.pvalue, expected)
    np.testing.assert_allclose(result.pvalue * 129, np.round(result.pvalue * 129), atol=1e-13)

    pointwise_expected = (np.sum(np.abs(result.surrogate) >= np.abs(result.stat)[:, np.newaxis], axis=-1) + 1) / 129
    np.testing.assert_array_equal(result.raw_pvalue, pointwise_expected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"method": "glm"}, "method"),
        ({"mcorrect": "hochberg"}, "mcorrect"),
        ({"alpha": 0.0}, "alpha"),
        ({"naccu": 0}, "naccu"),
        ({"naccu": 1.5}, "naccu"),
        ({"mcorrect": "max"}, "max correction"),
        ({"mcorrect": "cluster"}, "cluster correction"),
        ({"neighbours": [{"label": "Cz"}]}, "spatial-neighbour"),
    ],
)
def test_statcondfieldtrip_rejects_invalid_or_unavailable_inference_options(kwargs, message):
    first, second, _ = _reference_conditions()
    with pytest.raises((ValueError, NotImplementedError), match=message):
        statcondfieldtrip([first, second], paired="on", **kwargs)


def test_statcondfieldtrip_rejects_designs_disabled_by_the_maintained_wrapper():
    first, second, third = _reference_conditions()
    with pytest.raises(NotImplementedError, match="paired one-way"):
        statcondfieldtrip([first, second, third], paired="on")
    with pytest.raises(NotImplementedError, match="two-way"):
        statcondfieldtrip(((first, second), (second, third)), paired="off")
    with pytest.raises(NotImplementedError, match="equal-variance"):
        statcondfieldtrip([first, second], paired="off", variance="inhomogenous")
    with pytest.raises(ValueError, match="variance"):
        statcondfieldtrip([first, second], variance="pooled")
    with pytest.raises(ValueError, match="same number of cases"):
        statcondfieldtrip([first, second[:-1]], paired="on")


def test_statcondfieldtrip_remains_a_package_callable_after_submodule_import():
    import eegprep.functions.statistics as statistics

    module = importlib.import_module("eegprep.functions.statistics.statcondfieldtrip")

    assert statistics.statcondfieldtrip is module.statcondfieldtrip
    assert statistics.StatcondFieldtripResult is module.StatcondFieldtripResult
