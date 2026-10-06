"""Original guarded statcondfieldtrip workflow and additional Python checks."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations, product

import numpy as np
from scipy import sparse
from scipy import stats as scipy_stats
from scipy.cluster import vq

import eegprep
from eegprep.functions.statistics import StatcondFieldtripResult, statcondfieldtrip
from tests.eeglab_tests import eeglab_test
from tests.test_statcond_eeglab_tests import _legacy_assertsame, _matlab_cells

# Cluster-policy oracle: fieldtrip/fieldtrip@8e2307d7e7284c6870a5d12e244d9dc95a1faae3,
# ft_statistics_montecarlo.m and private/clusterstat.m. The exhaustive fixtures
# below independently check its label-exchangeability and maxsum principles.


def _source_pair(first, second):
    # As in the legacy statcond port, preserve the first statistic's MATLAB
    # numeric class when constructing the source's two-element comparison row.
    return np.concatenate([np.asarray(first).ravel(), np.asarray(second).ravel()]).astype(np.asarray(first).dtype)


@eeglab_test(
    "unittesting_statistics/statcondfieldtrip/statistics_statcondfieldtrip_wrapperTest.m",
    "test_test_statcondfieldtrip",
)
def test_reference_statcondfieldtrip_workflow(eeglab_backend, request):
    if request.config.getoption("--eeglab-backend") == "matlab":
        available = (
            eeglab_backend("license", "checkout", "statistics_toolbox").item()
            and eeglab_backend("exist", "kmean").item()
        )
    else:
        # No MATLAB licensing analogue; retain the original misspelled name
        # when querying the actual Python public/library namespaces.
        available = any(getattr(module, "kmean", None) is not None for module in (eegprep, vq))
    request.node.user_properties.append(("eeglab_source_body_entered", bool(available)))
    if not available:
        return

    rng = np.random.default_rng()
    rows = [[rng.random((1, 10)), rng.random((1, 10)) + 0.5]]
    for paired in ("on", "off"):
        options = {"variance": "homogenous"} if paired == "off" else {}
        first = eeglab_backend(
            "statcond", _matlab_cells(rows), mode="param", verbose="off", paired=paired, nargout=3, **options
        )
        second = eeglab_backend(
            "statcondfieldtrip",
            _matlab_cells(rows),
            mode="param",
            verbose="off",
            paired=paired,
            method="analytic",
            nargout=3,
            **options,
        )
        _legacy_assertsame(*(_source_pair(a, b) for a, b in zip(first, second, strict=True)))

    rows = [
        [rng.random((1, 10)), rng.random((1, 10)), rng.random((1, 10)) + 0.2],
        [rng.random((1, 10)), rng.random((1, 10)) + 0.2, rng.random((1, 10))],
    ]
    first = eeglab_backend("statcond", _matlab_cells(rows[:1]), mode="param", verbose="off", paired="off", nargout=3)
    second = eeglab_backend(
        "statcondfieldtrip",
        _matlab_cells(rows[:1]),
        mode="param",
        verbose="off",
        paired="off",
        method="analytic",
        nargout=3,
    )
    _legacy_assertsame(
        _source_pair(first[0], second[0]),
        _source_pair(first[1].flat[0], second[1].flat[0]),
        _source_pair(first[1].flat[1], second[1].flat[1]),
        _source_pair(first[2], second[2]),
    )
    # The source's following paired/2-way comparisons are inside literal if 0.
    indices = ((0,), (3,), (1, 3), (0, 1, 3))
    for paired in ("on", "off"):
        conditions = [
            [rng.random(shape) + offset for offset in (0, 0.5, 0)]
            for shape in ((1, 10), (10, 10), (5, 10, 10), (2, 5, 10, 10))
        ]
        for arrays, index in zip(conditions[1:], indices[1:], strict=True):
            for original, target in zip(conditions[0], arrays, strict=True):
                target[index] = original[0]
        _assert_reference_fieldtrip_dimensions(eeglab_backend, conditions, indices, paired, 2)
    # The paired 1-way dimensional section is also explicitly if 0. The final
    # unpaired ANOVA reuses the second (unpaired t-test) set of random arrays.
    _assert_reference_fieldtrip_dimensions(eeglab_backend, conditions, indices, "off", 3)


def _assert_reference_fieldtrip_dimensions(backend, conditions, indices, paired, n_conditions):
    options = {"variance": "homogenous"} if paired == "off" and n_conditions == 2 else {}
    first = [
        backend(
            "statcond",
            _matlab_cells([arrays[:n_conditions]]),
            mode="param",
            verbose="off",
            paired=paired,
            nargout=3,
            **options,
        )
        for arrays in conditions
    ]
    # FieldTrip's 4-D calls are commented out, but the original statcond 4-D
    # call above remains active even though its result is not asserted.
    second = [
        backend(
            "statcondfieldtrip",
            _matlab_cells([arrays[:n_conditions]]),
            mode="param",
            verbose="off",
            paired=paired,
            method="analytic",
            nargout=3,
        )
        for arrays in conditions[:3]
    ]
    compared = [*first[:3], *second]
    selected = [*indices[:3], *indices[:3]]
    statistics, probabilities = [], []
    for (statistic, _df, probability), index in zip(compared, selected, strict=True):
        statistics.append(np.asarray(statistic if statistic.size == 1 else statistic[index]).flat[0])
        probabilities.append(np.asarray(probability if probability.size == 1 else probability[index]).flat[0])
    _legacy_assertsame(
        np.array(statistics, dtype=np.asarray(first[0][0]).dtype),
        np.concatenate([np.asarray(result[1]).ravel(order="F") for result in compared]),
        np.array(probabilities, dtype=np.asarray(first[0][2]).dtype),
    )


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


def test_additional_statcondfieldtrip_vector_and_dimension_checks():
    # Generated fixtures and stricter assertions provide supplemental coverage.
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


def test_statcondfieldtrip_exact_paired_cluster_null_is_exhaustively_checkable():
    first = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [-2.0, -3.0, -4.0]])
    second = np.zeros_like(first)
    adjacency = sparse.csr_matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    result = statcondfieldtrip(
        [first, second],
        paired="on",
        method="montecarlo",
        naccu="all",
        mcorrect="cluster",
        neighbours=adjacency,
        clusteralpha=0.2,
        alpha=0.25,
    )

    critical = scipy_stats.t.isf(0.1, df=2)
    np.testing.assert_allclose(result.stat, np.sqrt(3) * np.array([2.0, 3.0, -3.0]))
    np.testing.assert_allclose(result.cluster_critical_value, critical)
    np.testing.assert_allclose(np.sort(result.cluster_null), [0.0] * 6 + [5 * np.sqrt(3)] * 2, atol=1e-14)
    np.testing.assert_allclose(result.raw_pvalue, [0.25, 0.25, 0.25])
    np.testing.assert_allclose(result.pvalue, [0.25, 0.25, 0.25])
    np.testing.assert_array_equal(result.mask, [True, True, True])
    np.testing.assert_array_equal(result.cluster_labels, [1, 1, 2])
    assert result.exact is True
    assert [(cluster.sign, cluster.indices) for cluster in result.clusters] == [
        (1, (0, 1)),
        (-1, (2,)),
    ]
    np.testing.assert_allclose([cluster.mass for cluster in result.clusters], [5 * np.sqrt(3), 3 * np.sqrt(3)])
    np.testing.assert_allclose([cluster.pvalue for cluster in result.clusters], [0.25, 0.25])

    # Three paired cases have exactly 2**3 label-swap assignments. Independent
    # enumeration confirms that only the unchanged and globally swapped designs
    # cross the two-sided threshold, so the exact cluster probability is 2/8.
    exhaustive = []
    for signs in product((-1.0, 1.0), repeat=3):
        statistic = scipy_stats.ttest_1samp(first * np.asarray(signs), 0.0, axis=-1).statistic
        masses = [0.0]
        if np.all(statistic[:2] >= critical):
            masses.append(np.sum(statistic[:2]))
        if np.all(statistic[:2] <= -critical):
            masses.append(-np.sum(statistic[:2]))
        if statistic[2] >= critical:
            masses.append(statistic[2])
        if statistic[2] <= -critical:
            masses.append(-statistic[2])
        exhaustive.append(max(masses))
    np.testing.assert_allclose(np.sort(result.cluster_null), np.sort(exhaustive), atol=1e-14)


def test_statcondfieldtrip_exact_unpaired_clusters_preserve_group_sizes():
    first = np.array([[5.0, 4.0], [6.0, 5.0]])
    second = np.array([[1.0, 2.0], [2.0, 3.0]])

    result = statcondfieldtrip(
        [first, second],
        paired="off",
        method="permutation",
        naccu="all",
        mcorrect="cluster",
        neighbours=np.array([[0, 1], [1, 0]], dtype=bool),
        clustercritval=1.0,
        alpha=1 / 3,
    )

    expected_mass = 6 * np.sqrt(2)
    np.testing.assert_allclose(result.stat, [3 * np.sqrt(2), 3 * np.sqrt(2)])
    np.testing.assert_allclose(np.sort(result.cluster_null), [0.0] * 4 + [expected_mass] * 2, atol=1e-14)
    np.testing.assert_allclose(result.pvalue, [1 / 3, 1 / 3])
    np.testing.assert_array_equal(result.mask, [True, True])
    assert result.surrogate.shape == (2, 6)

    pooled = np.concatenate([first, second], axis=-1)
    exhaustive = []
    for selected in combinations(range(4), 2):
        remaining = tuple(index for index in range(4) if index not in selected)
        statistic = scipy_stats.ttest_ind(
            pooled[:, selected],
            pooled[:, remaining],
            axis=-1,
            equal_var=True,
        ).statistic
        exhaustive.append(np.sum(np.abs(statistic)) if np.all(np.abs(statistic) >= 1.0) else 0.0)
    np.testing.assert_allclose(result.cluster_null, exhaustive, atol=1e-14)


def test_statcondfieldtrip_exact_one_way_clusters_use_right_tailed_f_mass():
    conditions = [
        np.array([[4.0, 5.0], [5.0, 6.0]]),
        np.array([[2.0, 3.0], [2.5, 3.5]]),
        np.array([[0.0, 1.0], [0.5, 1.5]]),
    ]
    result = statcondfieldtrip(
        conditions,
        paired="off",
        method="montecarlo",
        naccu="all",
        mcorrect="cluster",
        neighbours=np.array([[0, 1], [1, 0]]),
        clusteralpha=0.2,
    )

    pooled = np.concatenate(conditions, axis=-1)
    critical = scipy_stats.f.isf(0.2, dfn=2, dfd=3)
    exhaustive = []
    for first_group in combinations(range(6), 2):
        after_first = tuple(index for index in range(6) if index not in first_group)
        for second_group in combinations(after_first, 2):
            third_group = tuple(index for index in after_first if index not in second_group)
            statistic = scipy_stats.f_oneway(
                pooled[:, first_group],
                pooled[:, second_group],
                pooled[:, third_group],
                axis=-1,
            ).statistic
            exhaustive.append(np.sum(statistic[statistic >= critical]))

    assert result.surrogate.shape == (2, 90)
    assert all(cluster.sign == 1 for cluster in result.clusters)
    np.testing.assert_allclose(result.cluster_critical_value, critical)
    np.testing.assert_allclose(result.cluster_null, exhaustive, rtol=1e-13, atol=1e-13)


def test_statcondfieldtrip_cluster_result_is_explicit_when_no_cluster_forms():
    generator = np.random.default_rng(946)
    first = generator.normal(size=(3, 5))
    second = generator.normal(size=(3, 5))

    result = statcondfieldtrip(
        [first, second],
        paired="off",
        method="montecarlo",
        naccu=15,
        mcorrect="cluster",
        neighbours=np.zeros((3, 3)),
        clustercritval=1e6,
        rng=22,
    )

    np.testing.assert_array_equal(result.pvalue, np.ones(3))
    np.testing.assert_array_equal(result.mask, np.zeros(3, dtype=bool))
    np.testing.assert_array_equal(result.cluster_labels, np.zeros(3, dtype=int))
    np.testing.assert_array_equal(result.cluster_null, np.zeros(15))
    assert result.clusters == ()


def test_statcondfieldtrip_cluster_montecarlo_is_seeded_and_uses_plus_one():
    generator = np.random.default_rng(945)
    first = generator.normal(size=(4, 8))
    second = first + np.array([0.9, 0.7, -0.8, -0.7])[:, None] + generator.normal(scale=0.3, size=(4, 8))
    adjacency = np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ]
    )
    kwargs = {
        "paired": "on",
        "method": "montecarlo",
        "naccu": 63,
        "mcorrect": "cluster",
        "neighbours": adjacency,
        "clustercritval": 1.5,
    }

    result = statcondfieldtrip([first, second], rng=71, **kwargs)
    repeated = statcondfieldtrip([first, second], rng=71, **kwargs)
    changed_seed = statcondfieldtrip([first, second], rng=72, **kwargs)

    np.testing.assert_array_equal(result.surrogate, repeated.surrogate)
    np.testing.assert_array_equal(result.cluster_null, repeated.cluster_null)
    assert not np.array_equal(result.cluster_null, changed_seed.cluster_null)
    np.testing.assert_allclose(result.pvalue * 64, np.round(result.pvalue * 64), atol=1e-13)
    assert result.exact is False
