"""Current eeglab_tests ports for legacy decomposition and ICA helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from eegprep.functions.miscfunc.kmeans_st import kmeans_st
from eegprep.functions.miscfunc.loc_subsets import loc_subsets
from eegprep.functions.miscfunc.make_timewarp import make_timewarp
from eegprep.functions.miscfunc.misc import finite_matmul
from eegprep.functions.miscfunc.promax import promax
from eegprep.functions.miscfunc.runicalowmem import runicalowmem
from eegprep.functions.miscfunc.runpca import runpca
from eegprep.functions.miscfunc.runpca2 import runpca2
from eegprep.functions.miscfunc.varimax import varimax
from eegprep.functions.miscfunc.varsort import varsort
from eegprep.functions.miscfunc.zica import zica
from tests.eeglab_tests import eeglab_test


MISC_ROOT = "unittesting_miscfunc"


@eeglab_test(f"{MISC_ROOT}/kmeans_st/miscfunc_kmeans_st_wrapperTest.m", "test_test_kmeans_st")
def test_kmeans_st_current_suite_cluster_counts_restarts_and_sse():
    rng = np.random.default_rng(10)
    observations = np.vstack((rng.normal((-3, 0), 0.15, (100, 2)), rng.normal((3, 0), 0.15, (100, 2))))

    centers, labels, sse = kmeans_st(observations, 2)
    restarted_centers, restarted_labels, restarted_sse = kmeans_st(observations, 2, 10)
    many_centers, many_labels, many_sse = kmeans_st(observations, 20, 10)

    assert centers.shape == (2, 2)
    assert set(labels.tolist()) == {0, 1}
    assert np.mean(labels[:100] == labels[0]) == 1
    assert np.mean(labels[100:] == labels[100]) == 1
    assert labels[0] != labels[100]
    np.testing.assert_allclose(
        sse,
        np.sum((observations - centers[labels]) ** 2),
        rtol=1e-14,
        atol=1e-14,
    )
    assert restarted_centers.shape == (2, 2)
    assert restarted_labels.shape == (200,)
    assert restarted_sse <= sse + 1e-12
    assert many_centers.shape == (20, 2)
    assert len(np.unique(many_labels)) == 20
    assert np.isfinite(many_sse)


@eeglab_test(f"{MISC_ROOT}/loc_subsets/miscfunc_loc_subsets_wrapperTest.m", "test_test_loc_subsets")
def test_loc_subsets_current_suite_balances_spatial_sets_and_honors_mandatory_channels():
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    chanlocs = [
        {"labels": f"E{index + 1}", "X": np.cos(angle), "Y": np.sin(angle), "Z": 0.2 * (-1) ** index}
        for index, angle in enumerate(angles)
    ]
    before = set(plt.get_fignums())
    subsets, membership, positions = loc_subsets(
        chanlocs,
        [4, 4],
        True,
        True,
        [[0, 3], [1, 4]],
    )
    created = set(plt.get_fignums()) - before
    try:
        assert [len(subset) for subset in subsets] == [4, 4, 4]
        np.testing.assert_array_equal(np.sort(np.concatenate(subsets)), np.arange(12))
        assert {0, 3}.issubset(subsets[0])
        assert {1, 4}.issubset(subsets[1])
        assert membership.shape == (12,)
        assert positions.shape == (3, 12)
        assert len(created) == 2
    finally:
        for figure in created:
            plt.close(figure)


def _timewarp_eeg() -> dict:
    response_latencies = [300.0, 320.0, 310.0, 900.0, 305.0]
    return {
        "epoch": [
            {
                "eventtype": ["square", "noise", "rt"],
                "eventlatency": [[0.0], [100.0], [latency]],
                "eventaccuracy": [[1], [0], [1 if index != 1 else 0]],
            }
            for index, latency in enumerate(response_latencies)
        ]
    }


@eeglab_test(f"{MISC_ROOT}/make_timewarp/miscfunc_make_timewarp_wrapperTest.m", "test_test_make_timewarp")
def test_make_timewarp_current_suite_default_outlier_and_condition_calls():
    eeg = _timewarp_eeg()
    default = make_timewarp(eeg, ["square", "rt"])
    broad = make_timewarp(
        eeg,
        ["square", "rt"],
        baseline_latency=0,
        max_std_for_absolute=3,
        max_std_for_relative=2,
    )
    strict = make_timewarp(
        eeg,
        ["square", "rt"],
        baseline_latency=0,
        max_std_for_absolute=0.6,
        max_std_for_relative=0.4,
    )
    conditioned = make_timewarp(
        eeg,
        ["square", "rt"],
        baseline_latency=0,
        event_conditions=["latency < 20000", "latency > 1000 || accuracy == 1"],
        max_std_for_absolute=3,
        max_std_for_relative=2,
    )

    assert default["latencies"].shape == (5, 2)
    np.testing.assert_array_equal(default["epochs"], np.arange(5))
    np.testing.assert_array_equal(broad["epochs"], np.arange(5))
    assert 3 not in strict["epochs"]
    np.testing.assert_array_equal(conditioned["epochs"], [0, 2, 3, 4])
    assert conditioned["event_sequence"] == ["square", "rt"]


@eeglab_test(f"{MISC_ROOT}/promax/miscfunc_promax_wrapperTest.m", "test_pass_column_vector")
def test_promax_current_suite_singular_column_vector_is_finite():
    rotation, orthogonal = promax(np.asarray([[1.0], [2.0], [3.0]]))
    assert rotation.shape == orthogonal.shape == (3, 3)
    assert np.all(np.isfinite(rotation))
    assert np.all(np.isfinite(orthogonal))


@eeglab_test(f"{MISC_ROOT}/promax/miscfunc_promax_wrapperTest.m", "test_pass_general")
def test_promax_current_suite_general_rotation_contract():
    data = np.asarray([[-1, 0, 1, 2], [5, -2, 3, -4], [0, 1, -1, 0]], dtype=float)
    rotation, orthogonal = promax(data)
    assert rotation.shape == orthogonal.shape == (3, 3)
    np.testing.assert_allclose(finite_matmul(orthogonal, orthogonal.T), np.eye(3), rtol=1e-12, atol=1e-12)
    assert np.linalg.matrix_rank(rotation) == 3
    assert np.all(np.isfinite(finite_matmul(rotation, data)))


@eeglab_test(f"{MISC_ROOT}/promax/miscfunc_promax_wrapperTest.m", "test_pass_maxit")
def test_promax_current_suite_iteration_limit_is_deterministic():
    data = np.asarray([[-1, 0, 1, 2], [5, -2, 3, -4], [0, 1, -1, 0]], dtype=float)
    first = promax(data, max_iterations=2)
    second = promax(data, max_iterations=2)
    np.testing.assert_allclose(first[0], second[0], rtol=0, atol=0)
    np.testing.assert_allclose(first[1], second[1], rtol=0, atol=0)


@eeglab_test(f"{MISC_ROOT}/promax/miscfunc_promax_wrapperTest.m", "test_pass_ncomps_small")
def test_promax_current_suite_reduced_rotation_operates_on_original_channels():
    data = np.asarray([[-1, 0, 1, 2], [5, -2, 3, -4], [0, 1, -1, 0]], dtype=float)
    rotation, orthogonal = promax(data, n_components=2)
    assert rotation.shape == orthogonal.shape == (2, 3)
    assert finite_matmul(rotation, data).shape == (2, 4)
    np.testing.assert_allclose(finite_matmul(orthogonal, orthogonal.T), np.eye(2), rtol=1e-12, atol=1e-12)
    matlab_rotation, matlab_varimax = promax(data, n_components=2, max_iterations=2)
    # Golden values were generated with pinned EEGLAB 8ac485f. Component
    # signs are mathematically arbitrary, so compare magnitudes.
    np.testing.assert_allclose(
        np.abs(matlab_rotation),
        np.abs(
            [
                [0.817648255658059, -0.371658928580924, -0.520321459598312],
                [0.753283725590137, 0.251094792144295, -0.627736492228675],
            ]
        ),
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        np.abs(matlab_varimax),
        np.abs(
            [
                [0.437473594303329, -0.893099927529104, -0.104830213845220],
                [0.680046327931768, 0.404859211314217, -0.611249548777379],
            ]
        ),
        rtol=2e-12,
        atol=2e-12,
    )


@eeglab_test(f"{MISC_ROOT}/promax/miscfunc_promax_wrapperTest.m", "test_pass_row_vector")
def test_promax_current_suite_one_component_rotation_is_identity():
    rotation, orthogonal = promax(np.asarray([[-1.0, 0.0, 1.0, 92.0]]))
    np.testing.assert_array_equal(rotation, [[1.0]])
    np.testing.assert_array_equal(orthogonal, [[1.0]])


@eeglab_test(f"{MISC_ROOT}/icademo/miscfunc_icademo_wrapperTest.m", "test_pass_general")
@eeglab_test(f"{MISC_ROOT}/runicalowmem/miscfunc_runicalowmem_wrapperTest.m", "test_test_runicalowmem")
@eeglab_test(f"{MISC_ROOT}/testica/miscfunc_testica_wrapperTest.m", "test_test_testica")
def test_runicalowmem_current_suite_recovers_deterministic_independent_sources():
    # The current icademo body is entirely commented, while testica is an
    # interactive plot benchmark without accuracy assertions. This numerical
    # recovery check preserves their scientific intent without porting obsolete
    # pause-driven demos as public APIs.
    rng = np.random.default_rng(4)
    samples = 3000
    sources = np.vstack(
        (
            rng.laplace(size=samples),
            rng.uniform(-np.sqrt(3), np.sqrt(3), size=samples),
            np.sign(np.sin(np.linspace(0, 90, samples))),
        )
    )
    mixing = np.asarray([[1.0, 0.5, -0.2], [0.3, 1.0, 0.4], [-0.4, 0.2, 1.0]])
    data = finite_matmul(mixing, sources)
    weights, sphere, *_rest = runicalowmem(
        data,
        extended=1,
        seed=11,
        maxsteps=128,
        stop=1e-6,
        verbose="off",
    )
    recovered = finite_matmul(
        finite_matmul(weights, sphere),
        data - np.mean(data, axis=1, keepdims=True),
    )
    correlations = np.abs(np.corrcoef(recovered, sources)[:3, 3:])
    recovered_rows, source_rows = linear_sum_assignment(-correlations)
    assert np.min(correlations[recovered_rows, source_rows]) > 0.9

    standard_weights, standard_sphere, *_standard_rest = runicalowmem(
        data,
        seed=11,
        maxsteps=64,
        verbose="off",
    )
    assert standard_weights.shape == standard_sphere.shape == (3, 3)
    assert np.all(np.isfinite(standard_weights))
    assert np.all(np.isfinite(standard_sphere))


@eeglab_test(f"{MISC_ROOT}/runpca/miscfunc_runpca_wrapperTest.m", "test_pass_simple_pca")
def test_runpca_current_suite_rank_one_data_reconstructs_and_orders_variance():
    data = np.vstack((np.arange(8, dtype=float), np.arange(8, dtype=float)))
    centered = data - np.mean(data, axis=1, keepdims=True)
    components, mixing, singular = runpca(data)
    np.testing.assert_allclose(finite_matmul(mixing, components), centered, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(finite_matmul(components, components.T), np.eye(2), rtol=1e-13, atol=1e-13)
    assert singular[0, 0] == pytest.approx(np.sqrt(84), rel=1e-13)
    assert singular[1, 1] < 1e-14


@eeglab_test(f"{MISC_ROOT}/runpca2/miscfunc_runpca2_wrapperTest.m", "test_test_runpca2")
def test_runpca2_current_suite_full_reduced_and_large_channel_cases():
    upstream_data = np.asarray(
        [
            [2, 5, 3, 6, 7, 2, 6, 8, 1, 2],
            [6, 1, 10, 234, 3, 5, 464, 3, 2, 5],
            [1, 1, 1, 1, 3, 5, 1, 1, 4, 5],
            [4, 23456, 2, 3, 1, 1, 34, 2, 3, 5],
            [20, 30, 10, 10, 34, 10, 30, 20, 30, 10],
        ],
        dtype=float,
    )
    _upstream_components, upstream_mixing, upstream_scales = runpca2(upstream_data, 3)
    # This upstream matrix spans four orders of magnitude; the 3e-8 relative
    # bound covers LAPACK eigenvector differences while remaining far below
    # the scale of any scientifically meaningful loading change.
    np.testing.assert_allclose(
        np.abs(upstream_mixing),
        np.abs(
            [
                [-0.267285337417358, -0.887335756250489, 0.566078569671753],
                [23.9382481949261, -145.190072705975, -0.113221069510140],
                [0.433916229756323, 0.692175058147292, -0.018444428160592],
                [-7034.9729670268, -0.493135863973243, -0.00437367024143441],
                [-3.2042903605228, -1.82911199166489, 8.70677338041587],
            ]
        ),
        rtol=3e-8,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        upstream_scales,
        [7035.01444303606, 145.206792255322, 8.72591118345028, 2.25550175815324, 1.20621009228612],
        rtol=5e-9,
        atol=5e-9,
    )

    rng = np.random.default_rng(9)
    for channels, samples, retained in ((5, 10, 3), (32, 100, 26)):
        data = rng.normal(size=(channels, samples))
        full_components, full_mixing, scales = runpca2(data)
        components, mixing, reduced_scales = runpca2(data, retained)
        centered = data - np.mean(data, axis=1, keepdims=True)
        np.testing.assert_allclose(finite_matmul(full_mixing, full_components), centered, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            finite_matmul(full_components, full_components.T) / samples,
            np.eye(channels),
            rtol=1e-11,
            atol=1e-11,
        )
        assert components.shape == (retained, samples)
        assert mixing.shape == (channels, retained)
        assert scales.shape == reduced_scales.shape == (channels,)
        assert np.all(np.diff(scales) <= 0)


def _varimax_criterion(data: np.ndarray) -> float:
    return float(np.sum(np.mean(data**4, axis=1) - np.mean(data**2, axis=1) ** 2))


@eeglab_test(f"{MISC_ROOT}/varimax/miscfunc_varimax_wrapperTest.m", "test_test_varimax")
def test_varimax_current_suite_default_tolerance_and_reorder_modes():
    data = np.random.default_rng(7).normal(size=(32, 100))
    default_rotation, default_data = varimax(data)
    numeric_rotation, numeric_data = varimax(data, 1e-2, True)
    named_rotation, named_data = varimax(data, 1e-2, "reorder")

    np.testing.assert_allclose(default_data, finite_matmul(default_rotation, data), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        finite_matmul(default_rotation, default_rotation.T),
        np.eye(32),
        rtol=1e-12,
        atol=1e-12,
    )
    assert _varimax_criterion(default_data) >= _varimax_criterion(data) - 1e-11
    np.testing.assert_allclose(numeric_rotation, named_rotation, rtol=0, atol=0)
    np.testing.assert_allclose(numeric_data, named_data, rtol=0, atol=0)

    upstream_data = np.asarray([[-1, 0, 1, 2], [5, -2, 3, -4], [0, 1, -1, 0]], dtype=float)
    upstream_rotation, upstream_rotated = varimax(upstream_data, 1e-2, "reorder")
    np.testing.assert_allclose(
        upstream_rotation,
        [
            [-0.421894420114366, 0.898540791629798, 0.120952652115025],
            [0.777001824218500, 0.427079904793369, -0.462461803917697],
            [0.467197242540543, 0.101129623251460, 0.878350463006385],
        ],
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        upstream_rotated,
        [
            [4.91459837826336, -1.67612893114457, 2.15277530266000, -4.43795200674792],
            [1.35839769974834, -1.31662161350443, 2.52070334251630, -0.154315970736475],
            [0.0384508737167593, 0.676091216503464, -0.107764350711461, 0.529875992075243],
        ],
        rtol=2e-12,
        atol=2e-12,
    )


@eeglab_test(f"{MISC_ROOT}/varsort/miscfunc_varsort_wrapperTest.m", "test_test_varsort")
def test_varsort_current_suite_orders_projected_component_power_and_supports_reduction():
    activations = np.asarray([[1.0, -1.0, 1.0, -1.0], [4.0, -4.0, 4.0, -4.0]])
    weights = np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    sphere = np.eye(3)
    order, power = varsort(activations, weights, sphere)
    np.testing.assert_array_equal(order, [1, 0])
    np.testing.assert_allclose(power, [2.0, 0.5], rtol=1e-14, atol=1e-14)

    square_order, square_power = varsort(activations, np.diag([1.0, 2.0]), np.eye(2))
    np.testing.assert_array_equal(square_order, [1, 0])
    np.testing.assert_allclose(square_power, [4.0, 1.0], rtol=1e-14, atol=1e-14)


@eeglab_test(f"{MISC_ROOT}/zica/miscfunc_zica_wrapperTest.m", "test_test_zica")
def test_zica_current_suite_continuous_and_epoched_baselines_use_true_global_peaks():
    activations = np.asarray(
        [
            [1, -1, 1, -1, 1, -1, 1, -20],
            [2, -2, 2, -2, 2, -2, 2, -2],
            [1, 0, -1, 0, 1, 0, -1, 0],
        ],
        dtype=float,
    )
    continuous = zica(activations)
    epoched = zica(activations, 4, [0, 1, 2])

    assert continuous[0].shape == epoched[0].shape == activations.shape
    assert continuous[3][0] == 0
    assert continuous[4][0] == 7
    assert epoched[3][0] == 0
    assert epoched[4][0] == 7
    assert np.all(np.diff(epoched[2]) <= 0)


def test_decomposition_helpers_reject_invalid_dimensions_and_indices():
    with pytest.raises(ValueError, match="clusters"):
        kmeans_st(np.ones((2, 2)), 3)
    with pytest.raises(ValueError, match="n_components"):
        runpca(np.ones((2, 5)), 3)
    with pytest.raises(ValueError, match="divide"):
        zica(np.ones((2, 7)), frames=4)
    with pytest.raises(ValueError, match="arbitrary code|unsupported expression"):
        make_timewarp(
            {"epoch": [{"eventtype": ["x"], "eventlatency": [0]}]},
            ["x"],
            event_conditions=["__import__('os')"],
        )
