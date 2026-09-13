"""Behavioral ports of current EEGLAB miscellaneous numerical tests."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from eegprep import (
    abspeak,
    averef,
    covary,
    datlim,
    eucl,
    gabor2d,
    gauss,
    gauss2d,
    gauss3d,
    hungarian,
    laplac2d,
    mapcorr,
    matcorr,
    matperm,
    means,
    nan_std,
    pcexpand,
    pcsquash,
    perminv,
    scanfold,
    uniquef,
    vectdata,
)
from tests.eeglab_tests import eeglab_test


def _source(name: str) -> str:
    return f"unittesting_miscfunc/{name}/miscfunc_{name}_wrapperTest.m"


@eeglab_test(_source("abspeak"), "test_fail_invalid_frames")
def test_current_abspeak_rejects_an_epoch_length_that_does_not_divide_data():
    with pytest.raises(ValueError, match="divide"):
        abspeak(np.ones((2, 5)), 6)


@eeglab_test(_source("abspeak"), "test_fail_no_arg")
def test_current_abspeak_requires_data():
    with pytest.raises(TypeError):
        abspeak()  # type: ignore[call-arg]


@eeglab_test(_source("abspeak"), "test_pass_all_identical")
@eeglab_test(_source("abspeak"), "test_pass_all_negative")
@eeglab_test(_source("abspeak"), "test_pass_all_positive")
@eeglab_test(_source("abspeak"), "test_pass_inf")
@eeglab_test(_source("abspeak"), "test_pass_mixed_sign")
@eeglab_test(_source("abspeak"), "test_pass_small_diff")
def test_current_abspeak_returns_last_tied_peak_and_its_sign():
    cases = [
        (
            [[-2, -2, -2, -2], [5, 5, 5, 5]],
            [2, 5],
            [3, 3],
            [-1, 1],
        ),
        (
            [[-2, -1, -3, -5, -4], [-8, -2, -6, -7, -1]],
            [5, 8],
            [3, 0],
            [-1, -1],
        ),
        (
            [[2, 1, 3, 5, 4], [8, 2, 6, 7, 1]],
            [5, 8],
            [3, 0],
            [1, 1],
        ),
        (
            [[5, -1, 7, -3, 9], [-2, 4, -7, 3, -5]],
            [9, 7],
            [4, 2],
            [1, -1],
        ),
        (
            [[-2, -2.0000000000001, -1.9999999999999, -2], [5.0000000000002, -5.0000000000001, 5, 5]],
            [2.0000000000001, 5.0000000000002],
            [1, 0],
            [-1, 1],
        ),
    ]
    for data, expected_amplitudes, expected_frames, expected_signs in cases:
        amplitudes, frames, signs = abspeak(data)
        np.testing.assert_allclose(amplitudes[:, 0], expected_amplitudes)
        np.testing.assert_array_equal(frames[:, 0], expected_frames)
        np.testing.assert_array_equal(signs[:, 0], expected_signs)

    amplitudes, frames, signs = abspeak([[1, np.inf, 3], [-np.inf, 2, np.inf]])
    np.testing.assert_array_equal(amplitudes[:, 0], [np.inf, np.inf])
    np.testing.assert_array_equal(frames[:, 0], [1, 2])
    np.testing.assert_array_equal(signs[:, 0], [1, 1])
    amplitudes, frames, signs = abspeak([[1, -4, -7, 2]], 2)
    np.testing.assert_array_equal(amplitudes, [[4, 7]])
    np.testing.assert_array_equal(frames, [[1, 0]])
    np.testing.assert_array_equal(signs, [[-1, -1]])


@eeglab_test(_source("abspeak"), "test_pass_nan_inf")
def test_current_abspeak_ignores_nan_without_masking_infinite_peaks():
    amplitudes, frames, signs = abspeak([[np.nan, -np.inf, np.nan], [np.nan, np.nan, np.nan]])
    assert amplitudes[0, 0] == np.inf
    assert frames[0, 0] == 1
    assert signs[0, 0] == -1
    assert np.isnan(amplitudes[1, 0])
    assert frames[1, 0] == -1
    assert np.isnan(signs[1, 0])


@eeglab_test(_source("averef"), "test_fail_few_data")
def test_current_averef_rejects_single_channel_data():
    with pytest.raises(ValueError, match="two channels"):
        averef([[2, 8]])


@eeglab_test(_source("averef"), "test_fail_no_arg")
def test_current_averef_requires_data():
    with pytest.raises(TypeError):
        averef()  # type: ignore[call-arg]


@eeglab_test(_source("averef"), "test_pass_one_arg_mixed")
@eeglab_test(_source("averef"), "test_pass_one_arg_positive")
def test_current_averef_removes_each_frames_channel_mean():
    positive = np.asarray([[2, 1, 3], [8, 2, 6]], dtype=float)
    np.testing.assert_allclose(averef(positive), [[-3, -0.5, -1.5], [3, 0.5, 1.5]])
    mixed = np.asarray([[2, 1, 3, 2.4, -np.pi], [0, -3, -3, -1.8, np.pi]])
    np.testing.assert_allclose(averef(mixed), [[1, 2, 3, 2.1, -np.pi], [-1, -2, -3, -2.1, np.pi]])

    referenced, weights, sphere, mean_data = averef(positive, np.eye(2), np.eye(2), return_parameters=True)
    np.testing.assert_allclose(referenced.mean(axis=0), 0, atol=1e-15)
    np.testing.assert_allclose(sphere, np.eye(2))
    np.testing.assert_allclose(mean_data, positive.mean(axis=0))
    assert weights is not None and weights.shape == (2, 2)


@eeglab_test(_source("covary"), "test_pass_mixed_matrix")
@eeglab_test(_source("covary"), "test_pass_mixed_vector")
@eeglab_test(_source("covary"), "test_pass_positive_matrix")
@eeglab_test(_source("covary"), "test_pass_positive_vector")
@eeglab_test(_source("covary"), "test_test_covary")
def test_current_covary_preserves_grand_mean_centering_and_unbiased_scaling():
    np.testing.assert_allclose(covary([2, 1, 3, 5, 4]), 2.5)
    np.testing.assert_allclose(covary([-1, 2, 3, -4, 5]), 12.5)
    positive = np.asarray([[2, 1, 3, 5, 4], [5, 4, 3, 2, 1], [3, 3, 3, 3, 3], [1, 0, 1, 0, 1]]).T
    mixed = np.asarray([[2, -1, -4, 5, 3], [5, -4, 3, 2, -1], [-3, 3, -3, -3, 3], [1, 0, 1, 0, 1]]).T
    np.testing.assert_allclose(covary(positive), [2.95, 2.95, 0.45, 4.35])
    np.testing.assert_allclose(covary(mixed), [12.8125, 12.8125, 12.3125, 0.3125])
    assert covary([1, 4, 1]) == pytest.approx(3)


@eeglab_test(_source("datlim"), "test_fail_not_numeric")
def test_current_datlim_rejects_non_numeric_input():
    with pytest.raises(TypeError, match="numeric"):
        datlim(["one", "two"])


@eeglab_test(_source("datlim"), "test_pass_all_positive")
@eeglab_test(_source("datlim"), "test_pass_mixed")
def test_current_datlim_flattens_all_dimensions_and_retains_infinities():
    np.testing.assert_array_equal(datlim([[2, 1, 3], [8, np.inf, 0]]), [0, np.inf])
    np.testing.assert_array_equal(datlim([[-1e100, -np.inf, 12], [23, -100, 90]]), [-np.inf, 90])


@eeglab_test(_source("eucl"), "test_test_eucl")
def test_current_eucl_supports_within_between_and_reference_distances():
    first = np.asarray([[0, 0], [3, 4], [0, 8]], dtype=float)
    second = np.asarray([[0, 4], [6, 8]], dtype=float)
    np.testing.assert_allclose(eucl(first), [[0, 5, 8], [5, 0, 5], [8, 5, 0]])
    np.testing.assert_allclose(eucl(first, second), [[4, 10], [3, 5], [4, 6]])
    np.testing.assert_allclose(eucl(first, [[0, 0]]), [0, 5, 8])
    np.testing.assert_allclose(eucl([[0, 0]], second), [4, 10])
    assert eucl([[0, 0], [3, 4]]) == pytest.approx(5)
    np.testing.assert_allclose(eucl(first, second), np.asarray(eucl(second, first)).T)


@eeglab_test(_source("gabor2d"), "test_pass_cut")
@eeglab_test(_source("gabor2d"), "test_pass_general")
@eeglab_test(_source("gabor2d"), "test_pass_rotated")
@eeglab_test(_source("gabor2d"), "test_pass_ten_args")
def test_current_gabor2d_controls_orientation_frequency_phase_and_magnitude_cut():
    default = gabor2d(5, 4)
    assert default.shape == (5, 4)
    assert np.max(np.abs(default)) < 1
    rotated = gabor2d(5, 4, 72, 270)
    assert np.all(rotated[:, :2] <= 1e-15)
    assert np.all(rotated[:, 2:] >= -1e-15)
    high_frequency = gabor2d(50, 40, 20, 0, 10, 8, 25.5, 20.5, 30)
    assert np.count_nonzero(np.diff(np.signbit(high_frequency[:, 20]))) >= 4
    cut = gabor2d(50, 40, 20, 0, 10, 8, 25.5, 20.5, 0, 0.5)
    uncut = gabor2d(50, 40, 20, 0, 10, 8, 25.5, 20.5, 0, 0)
    assert np.count_nonzero(cut) < np.count_nonzero(uncut)
    assert np.any(cut < 0) and np.any(cut > 0)


@eeglab_test(_source("gauss"), "test_pass_general")
def test_current_gauss_is_symmetric_positive_and_unit_peaked():
    window = gauss(5, 1)
    np.testing.assert_allclose(window, window[::-1])
    assert window[2] == pytest.approx(1)
    assert np.all((window > 0) & (window <= 1))
    np.testing.assert_array_equal(gauss(1, 3), [1])


@eeglab_test(_source("gauss2d"), "test_pass_general")
@eeglab_test(_source("gauss2d"), "test_pass_seven_args")
def test_current_gauss2d_has_requested_peak_symmetry_and_cut():
    kernel = gauss2d(5, 5)
    np.testing.assert_allclose(kernel, kernel[::-1])
    np.testing.assert_allclose(kernel, kernel[:, ::-1])
    assert np.argmax(kernel) == np.ravel_multi_index((2, 2), kernel.shape)
    cut = gauss2d(5, 4, 1, 1, 3, 2.5, 0.5)
    uncut = gauss2d(5, 4, 1, 1, 3, 2.5, 0)
    assert np.count_nonzero(cut) < np.count_nonzero(uncut)


@eeglab_test(_source("gauss3d"), "test_test_gauss3d")
def test_current_gauss3d_honors_anisotropic_shape_peak_and_cut():
    assert gauss3d(2, 5, 7).shape == (2, 5, 7)
    kernel = gauss3d(3, 3, 3, 0.5, 0.1, 0.04, 1, 2, 3)
    assert np.argmax(kernel) == np.ravel_multi_index((0, 1, 2), kernel.shape)
    cut = gauss3d(3, 3, 3, 0.6, 0.6, 0.6, 2, 2, 2, 0.5)
    assert 0 < np.count_nonzero(cut) < cut.size


@eeglab_test(_source("hungarian"), "test_pass_equal")
@eeglab_test(_source("hungarian"), "test_pass_general")
@eeglab_test(_source("hungarian"), "test_pass_hard_one")
@eeglab_test(_source("hungarian"), "test_pass_hard_three")
@eeglab_test(_source("hungarian"), "test_pass_hard_two")
@eeglab_test(_source("hungarian"), "test_pass_ideal")
@eeglab_test(_source("hungarian"), "test_pass_negative")
@eeglab_test(_source("hungarian"), "test_pass_zeros")
def test_current_hungarian_finds_known_global_minima_with_unique_assignments():
    cases = [
        ([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]], 10),
        ([[1, 1, 1, 2], [3, 2, 4, 1], [4, 4, 2, 4], [2, 3, 3, 3]], 6),
        ([[4, 2, 4, 1], [2, 4, 1, 3], [1, 1, 3, 4], [3, 3, 2, 2]], 6),
        ([[3, 1, 1, 4], [1, 3, 2, 1], [2, 2, 3, 2], [4, 4, 4, 3]], 7),
        ([[3, 2, 3, 2], [2, 1, 2, 1], [1, 4, 1, 3], [4, 3, 4, 4]], 8),
        ([[1, 3, 2, 2], [3, 2, 4, 1], [4, 4, 1, 4], [2, 1, 3, 3]], 4),
        ([[-1, 1, 1, 2], [3, 2, 4, -1], [4, 4, -2, 4], [-2, 3, 3, 3]], -4),
        (np.zeros((4, 4)), 0),
    ]
    for matrix, expected_cost in cases:
        costs = np.asarray(matrix)
        assignment, total = hungarian(costs)
        np.testing.assert_array_equal(np.sort(assignment), np.arange(4))
        assert total == pytest.approx(expected_cost)
        brute_force = min(
            sum(costs[row, column] for column, row in enumerate(order)) for order in itertools.permutations(range(4))
        )
        assert total == pytest.approx(brute_force)


@eeglab_test(_source("laplac2d"), "test_pass_cut")
@eeglab_test(_source("laplac2d"), "test_pass_general")
@eeglab_test(_source("laplac2d"), "test_pass_mean")
@eeglab_test(_source("laplac2d"), "test_pass_sigma")
def test_current_laplac2d_honors_peak_location_scale_and_cut():
    kernel = laplac2d(5, 5)
    assert kernel[2, 2] == pytest.approx(1)
    assert kernel.min() == pytest.approx(-0.4060058497)
    shifted = laplac2d(25, 25, 5, 25, 4)
    assert np.argmax(shifted) == np.ravel_multi_index((24, 3), shifted.shape)
    assert laplac2d(25, 25, 1).max() == pytest.approx(1)
    upstream_cut = laplac2d(25, 25, 5, 12.5, 12.5, 1e-6)
    assert 0.038 < upstream_cut.max() < 0.039
    material_cut = laplac2d(25, 25, 5, 12.5, 12.5, 0.1)
    assert 0 < np.count_nonzero(material_cut) < material_cut.size


@eeglab_test(_source("mapcorr"), "test_test_mapcorr")
def test_current_mapcorr_aligns_values_by_channel_label_before_matching():
    first_channels = [{"labels": name} for name in ["Fz", "Cz", "Pz"]]
    second_channels = [{"labels": name} for name in ["Pz", "Fz", "Cz", "Oz"]]
    first = np.asarray([[1, 2, 3], [4, -2, 0]], dtype=float)
    second = np.asarray([[3, 1, 2, 99], [0, 4, -2, -12]], dtype=float)
    correlations, first_indices, second_indices, all_correlations = mapcorr(
        first, second, first_channels, second_channels, method=0
    )
    np.testing.assert_allclose(correlations, [1, 1])
    np.testing.assert_array_equal(first[first_indices], first[[0, 1]])
    np.testing.assert_array_equal(second_indices, [0, 1])
    np.testing.assert_allclose(np.diag(all_correlations), 1)
    zero_weight = mapcorr(first, second, first_channels, second_channels, method=0, weighting=0)
    np.testing.assert_allclose(zero_weight[3], all_correlations)


@eeglab_test(_source("matcorr"), "test_pass_general")
@eeglab_test(_source("matcorr"), "test_pass_mean")
@eeglab_test(_source("matcorr"), "test_pass_method_hungarian")
@eeglab_test(_source("matcorr"), "test_pass_method_vam")
@eeglab_test(_source("matcorr"), "test_pass_not_square")
@eeglab_test(_source("matcorr"), "test_pass_not_square_hungarian")
@eeglab_test(_source("matcorr"), "test_pass_num_rows")
@eeglab_test(_source("matcorr"), "test_pass_weights")
def test_current_matcorr_matches_permuted_rows_across_methods_and_rectangular_inputs():
    first = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    second = first[[1, 0, 2]]
    for method in (0, 1, 2):
        correlations, first_indices, second_indices, matrix = matcorr(first, second, method=method)
        np.testing.assert_allclose(first[first_indices], second[second_indices])
        np.testing.assert_allclose(correlations, 1)
        np.testing.assert_allclose(matrix[first_indices, second_indices], 1)
    centered = matcorr(first, second, remove_mean=True)
    np.testing.assert_allclose(centered[0], 1)
    rectangular = np.vstack([first, [1, 5, 9]])
    result = matcorr(rectangular, rectangular[[2, 3, 0, 1]], method=0)
    assert result[0].size == 4
    unequal = matcorr(rectangular, second)
    assert unequal[0].size == 3
    weights = np.asarray([[1, 0.4, 0.6], [0.5, 0.8, 1], [0.2, 1, 0.9]])
    weighted = matcorr(first, second, weighting=weights)
    np.testing.assert_allclose(weighted[3], matcorr(first, second)[3] * weights)


@eeglab_test(_source("matperm"), "test_pass_general")
def test_current_matperm_reorders_rows_and_corrects_component_polarity():
    first = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    second = np.asarray([[4, 5, 6], [-1, -2, -3], [7, 8, 9]])
    output, permutation = matperm(first, second, [1, 0, 2], [0, 1, 2], [1, -1, 1])
    np.testing.assert_array_equal(output, second)
    np.testing.assert_array_equal(permutation, [1, 0, 2])


@eeglab_test(_source("means"), "test_test_means")
def test_current_means_computes_groupwise_statistics_along_observations():
    values = np.asarray([[1, 2], [3, np.nan], [5, 6], [7, 10]], dtype=float)
    group_means, standard_errors, variances, group_ids = means(values, [2, 1, 2, 1])
    np.testing.assert_array_equal(group_ids, [1, 2])
    np.testing.assert_allclose(group_means, [[5, 10], [3, 4]])
    np.testing.assert_allclose(variances, [[8, np.nan], [8, 8]], equal_nan=True)
    np.testing.assert_allclose(standard_errors, [[2, np.nan], [2, 2]], equal_nan=True)
    assert means(np.ones((32, 100)))[0].shape == (1, 100)
    assert means(np.ones((32, 1)))[0].shape == (1, 1)
    assert means(np.ones((1, 100)))[0].shape == (1, 100)


@eeglab_test(_source("nan_std"), "test_pass_general")
@eeglab_test(_source("nan_std"), "test_pass_nan")
@eeglab_test(_source("nan_std"), "test_pass_number")
@eeglab_test(_source("nan_std"), "test_pass_row")
def test_current_nan_std_uses_sample_scaling_and_first_nonsingleton_axis():
    np.testing.assert_allclose(nan_std([[1, -5], [2, 0], [4, 6]]), [np.sqrt(7 / 3), np.sqrt(91 / 3)])
    values = [[np.nan, np.nan, 3, 0, -4], [np.nan, 1, 3, 4, 4], [np.nan, 11, 0, -7, np.nan]]
    np.testing.assert_allclose(
        nan_std(values), [np.nan, np.sqrt(50), np.sqrt(3), np.sqrt(31), np.sqrt(32)], equal_nan=True
    )
    assert np.isnan(nan_std(1))
    assert nan_std([1, 2]) == pytest.approx(np.sqrt(0.5))


@eeglab_test(_source("pcexpand"), "test_pass_general")
@eeglab_test(_source("pcexpand"), "test_pass_means_row_vector")
def test_current_pcexpand_accepts_row_or_column_mean_vectors():
    projections = np.asarray([[1, -2, 3], [-2, 0, 1]])
    vectors = np.asarray([[0.6, -0.8], [0.8, 0.6]])
    expected = np.asarray([[4.2, 0.8, 3], [0.6, -0.6, 4]])
    np.testing.assert_allclose(pcexpand(projections, vectors, [2, 1]), expected)
    np.testing.assert_allclose(pcexpand(projections, vectors, [[2], [1]]), expected)


@eeglab_test(_source("pcsquash"), "test_pass_column_vector")
@eeglab_test(_source("pcsquash"), "test_pass_general")
@eeglab_test(_source("pcsquash"), "test_pass_row_vector")
def test_current_pcsquash_orders_components_and_roundtrips_through_pcexpand():
    for data in (np.asarray([[1, 3]]).T, np.asarray([[1, 2, 3, 4, 5], [-2, 0, 2, -1, 6]]), np.asarray([1, 2, 3])):
        vectors, eigenvalues, compressed, data_mean = pcsquash(data)
        expected = np.asarray(data, dtype=float)
        if expected.ndim == 1:
            expected = expected.reshape(1, -1)
        np.testing.assert_allclose(pcexpand(compressed, vectors, data_mean), expected, atol=1e-12)
        assert np.all(np.diff(eigenvalues) <= 1e-15)
        np.testing.assert_allclose(vectors.T @ vectors, np.eye(vectors.shape[1]), atol=1e-12)


@eeglab_test(_source("perminv"), "test_pass_general")
def test_current_perminv_returns_zero_based_inverse_permutation():
    permutation = np.asarray([1, 3, 0, 4, 2])
    inverse = perminv(permutation)
    np.testing.assert_array_equal(inverse, [2, 0, 4, 1, 3])
    np.testing.assert_array_equal(permutation[inverse], np.arange(permutation.size))


@eeglab_test(_source("scanfold"), "test_test_scanfold")
def test_current_scanfold_respects_ignored_directories_and_depth(tmp_path):
    (tmp_path / "root.m").write_text("function root", encoding="utf-8")
    (tmp_path / "readme.txt").write_text("not MATLAB", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "nested.m").write_text("function nested", encoding="utf-8")
    ignored = tmp_path / "plugins"
    ignored.mkdir()
    (ignored / "plugin.m").write_text("function plugin", encoding="utf-8")

    files, command = scanfold(tmp_path)
    assert files == ["nested.m", "plugin.m", "root.m"]
    assert command == " -a nested.m -a plugin.m -a root.m"
    assert scanfold(tmp_path, {"plugins"})[0] == ["nested.m", "root.m"]
    assert scanfold(tmp_path, max_depth=1)[0] == ["root.m"]


@eeglab_test(_source("uniquef"), "test_test_uniquef")
def test_current_uniquef_returns_stable_counts_and_zero_based_first_indices():
    groups = [3, 2, 1, 2, 1, 2, 2, 2, 1, np.nan, np.inf]
    values, counts, indices = uniquef(groups)
    np.testing.assert_array_equal(values, [3, 2, 1])
    np.testing.assert_array_equal(counts, [1, 5, 3])
    np.testing.assert_array_equal(indices, [0, 1, 2])
    values, counts, indices = uniquef(groups, sort=True)
    np.testing.assert_array_equal(values, [1, 2, 3])
    np.testing.assert_array_equal(counts, [3, 5, 1])
    np.testing.assert_array_equal(indices, [2, 1, 0])


@eeglab_test(_source("vectdata"), "test_test_vectdata")
def test_current_vectdata_interpolates_and_smooths_with_explicit_v4_exclusion():
    times = np.arange(-20, 20.5, 0.5)
    values = np.sin(times)
    dense_times = np.linspace(-20, 20, 4001)
    linear, returned_times = vectdata(values, times, timesout=dense_times, method="linear")
    nearest, _ = vectdata(values, times, timesout=dense_times, method="nearest")
    cubic, _ = vectdata(values, times, timesout=dense_times, method="cubic")
    np.testing.assert_array_equal(returned_times, dense_times)
    np.testing.assert_allclose(linear[::50], values)
    np.testing.assert_allclose(nearest[::50], values)
    assert np.max(np.abs(cubic - np.sin(dense_times))) < 0.003

    impulse = np.zeros(9)
    impulse[4] = 1
    smoothed, _ = vectdata(impulse, np.arange(9), timesout=np.arange(9), average=3, avgtype="gauss")
    assert 0 < smoothed[4] < 1
    constant, _ = vectdata(np.ones(9), np.arange(9), timesout=np.arange(9), average=5, border="on")
    np.testing.assert_allclose(constant, 1)
    with pytest.raises(NotImplementedError, match="v4"):
        vectdata(values, times, timesout=dense_times, method="v4")
