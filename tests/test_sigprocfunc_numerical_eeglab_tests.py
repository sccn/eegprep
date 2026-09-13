"""Behavioral ports of current EEGLAB low-level numerical tests."""

from __future__ import annotations

import numpy as np
import pytest

from eegprep import celltomat, eyelike, fastif, matsel, mattocell, nan_mean, quantile, shuffle
from tests.eeglab_tests import eeglab_test


def _source(name: str) -> str:
    return f"unittesting_sigprocfunc/{name}/sigprocfunc_{name}_wrapperTest.m"


@eeglab_test(_source("celltomat"), "test_pass_1d_num")
@eeglab_test(_source("celltomat"), "test_pass_empty")
def test_current_celltomat_converts_rectangular_numeric_cells_and_empty_input():
    np.testing.assert_array_equal(celltomat([1, 2, 3]), [1, 2, 3])
    np.testing.assert_array_equal(celltomat([[1, 2, 3], [4, 5, 6]]), [[1, 2, 3], [4, 5, 6]])
    object_cells = np.asarray([[1, 2], [3, 4]], dtype=object)
    np.testing.assert_array_equal(celltomat(object_cells), [[1, 2], [3, 4]])
    matlab_cells = np.empty((2, 2), dtype=object)
    for index, value in enumerate([1, 2, 3, 4]):
        matlab_cells.flat[index] = np.asarray([[value]])
    np.testing.assert_array_equal(celltomat(matlab_cells), [[1, 2], [3, 4]])
    assert celltomat([]).size == 0
    with pytest.raises((TypeError, ValueError)):
        celltomat([[1, 2], [3]])
    invalid_cells = np.empty(1, dtype=object)
    invalid_cells[0] = np.asarray([1, 2])
    with pytest.raises(TypeError):
        celltomat(invalid_cells)


@eeglab_test(_source("eyelike"), "test_pass_diag_zero_1")
@eeglab_test(_source("eyelike"), "test_pass_diag_zero_2")
@eeglab_test(_source("eyelike"), "test_pass_eye")
@eeglab_test(_source("eyelike"), "test_pass_eye_scale")
@eeglab_test(_source("eyelike"), "test_pass_general")
def test_current_eyelike_produces_unit_diagonal_and_invertible_transform():
    matrices = [
        np.eye(3),
        np.diag([3, 2, 9]),
        np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float),
        np.asarray([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float),
        np.asarray([[0, -4, 1], [2, 0, 1], [1, 3, 0]], dtype=float),
        np.asarray([[1 + 2j, 3], [4, 2 - 1j]]),
    ]
    for matrix in matrices:
        normalized, scale, permutation = eyelike(matrix)
        np.testing.assert_allclose(np.diag(normalized), 1)
        np.testing.assert_allclose(np.linalg.inv(permutation) @ np.linalg.inv(scale) @ normalized, matrix)


@eeglab_test(_source("fastif"), "test_pass_complex")
@eeglab_test(_source("fastif"), "test_pass_false")
@eeglab_test(_source("fastif"), "test_pass_not_bool")
@eeglab_test(_source("fastif"), "test_pass_true")
def test_current_fastif_uses_python_truthiness_without_evaluating_strings():
    assert fastif(True, "yes", "no") == "yes"
    assert fastif(False, "yes", "no") == "no"
    assert fastif("not boolean", "yes", "no") == "yes"
    matrix = np.arange(1, 10).reshape(3, 3)
    np.testing.assert_array_equal(fastif(3 > 1, np.diag(matrix), matrix[:, 2]), [1, 5, 9])


@eeglab_test(_source("matsel"), "test_pass_general")
@eeglab_test(_source("matsel"), "test_pass_valid_frames")
def test_current_matsel_selects_zero_based_frames_from_each_flattened_epoch():
    data = np.asarray([[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(matsel(data, None, None), data)
    epoched = np.asarray([[1, 2, 3, 4], [5, 6, 7, 8]])
    np.testing.assert_array_equal(matsel(epoched, 2, [0]), [[1, 3], [5, 7]])
    np.testing.assert_array_equal(matsel(epoched, 2, [1], [1], [1, 0]), [[8, 6]])
    assert matsel(epoched, 2, [], [0], [0]).shape == (1, 0)
    with pytest.raises(IndexError):
        matsel(epoched, 2, [2])
    with pytest.raises(ValueError, match="integer"):
        matsel(epoched, 2.9, [0])
    with pytest.raises(ValueError, match="integers"):
        matsel(epoched, 2, [0.9])


@eeglab_test(_source("mattocell"), "test_pass_empty_data")
@eeglab_test(_source("mattocell"), "test_pass_general")
def test_current_mattocell_returns_nested_python_scalars_and_empty_list():
    assert mattocell([[1, 2, 3], [4, 5, 6]]) == [[1, 2, 3], [4, 5, 6]]
    assert mattocell([]) == []


@eeglab_test(_source("nan_mean"), "test_pass_general")
@eeglab_test(_source("nan_mean"), "test_pass_nan")
@eeglab_test(_source("nan_mean"), "test_pass_number")
@eeglab_test(_source("nan_mean"), "test_pass_row_vector")
def test_current_nan_mean_uses_first_nonsingleton_axis_and_preserves_all_nan_columns():
    values = np.asarray([[1, 2, 3, 0, -5], [1, 7, 3, 4, 5], [-2, 12, 0, -6, 3]], dtype=float)
    np.testing.assert_allclose(nan_mean(values), [0, 7, 2, -2 / 3, 1])
    values[[0, 1, 2], 0] = np.nan
    values[0, 1] = np.nan
    values[2, 4] = np.nan
    np.testing.assert_allclose(nan_mean(values), [np.nan, 9.5, 2, -2 / 3, 0], equal_nan=True)
    assert nan_mean(1) == 1
    assert nan_mean([1, 2, 3, 0, -5]) == pytest.approx(0.2)
    np.testing.assert_allclose(nan_mean([1 + 2j, 3 + 4j]), 2 + 3j)


@eeglab_test(_source("quantile"), "test_pass_column")
@eeglab_test(_source("quantile"), "test_pass_general")
@eeglab_test(_source("quantile"), "test_pass_matrix")
@eeglab_test(_source("quantile"), "test_pass_number")
@eeglab_test(_source("quantile"), "test_pass_unsorted")
def test_current_quantile_matches_midpoint_empirical_interpolation():
    probabilities = [0, 0.125, 0.25, 0.5, 0.75, 0.875, 1]
    expected = [1, 1, 1.5, 3.5, 7.5, 10, 10]
    np.testing.assert_allclose(quantile([1, 2, 5, 10], probabilities), expected)
    np.testing.assert_allclose(quantile([10, 2, 5, 1], probabilities), expected)
    np.testing.assert_allclose(quantile([[1], [2], [5], [10]], probabilities).reshape(-1), expected)
    matrix = np.asarray([[-1, 0, -4, -2], [1, 2, 5, 8], [3, 7, 6, 10]])
    matrix_expected = np.asarray(
        [[-1, 0, -4, -2], [-0.5, 0.5, -1.75, 0.5], [1, 2, 5, 8], [2.5, 5.75, 5.75, 9.5], [3, 7, 6, 10]]
    )
    np.testing.assert_allclose(quantile(matrix, [0, 0.25, 0.5, 0.75, 1]), matrix_expected)
    np.testing.assert_array_equal(quantile(17, [0.25, 0.5, 0.75]), [17, 17, 17])
    np.testing.assert_allclose(quantile([1, np.nan, 5], [0, 0.5, 1]), [1, 3, 5])
    with pytest.raises(ValueError, match="integer"):
        quantile(matrix, [0.5], axis=0.5)
    with pytest.raises(ValueError, match="real"):
        quantile([1 + 2j, 3 + 4j], [0.5])
    with pytest.raises(ValueError, match="real"):
        quantile([1, 3], [0.5 + 0.1j])


@eeglab_test(_source("shuffle"), "test_test_shuffle")
def test_current_shuffle_returns_zero_based_permutation_and_exact_inverse_for_any_axis():
    generator = np.random.default_rng(17)
    for shape, axis in [((32, 100), 0), ((32, 100), 1), ((4, 8, 10), 1), ((4, 8, 10), 2)]:
        data = np.arange(np.prod(shape)).reshape(shape)
        shuffled, permutation, inverse = shuffle(data, axis, rng=generator)
        np.testing.assert_array_equal(shuffled, np.take(data, permutation, axis=axis))
        np.testing.assert_array_equal(np.take(shuffled, inverse, axis=axis), data)
        np.testing.assert_array_equal(permutation[inverse], np.arange(shape[axis]))
    with pytest.raises(ValueError, match="integer"):
        shuffle(np.ones((2, 2)), axis=0.5)
