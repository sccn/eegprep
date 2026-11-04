"""Behavioral ports of current EEGLAB low-level numerical tests."""

from __future__ import annotations

import numpy as np
import pytest

from eegprep import celltomat, matsel, nan_mean, quantile, shuffle
from tests.eeglab_tests import eeglab_test
from tests.eeglab_tests import assert_matlab_near as _assert_near


def _source(name: str) -> str:
    return f"unittesting_sigprocfunc/{name}/sigprocfunc_{name}_wrapperTest.m"


def test_python_regression_celltomat_converts_rectangular_numeric_cells_and_empty_input():
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


def test_python_regression_matsel_selects_zero_based_frames_from_each_flattened_epoch():
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


def test_python_regression_nan_mean_uses_first_nonsingleton_axis_and_preserves_all_nan_columns():
    values = np.asarray([[1, 2, 3, 0, -5], [1, 7, 3, 4, 5], [-2, 12, 0, -6, 3]], dtype=float)
    np.testing.assert_allclose(nan_mean(values), [0, 7, 2, -2 / 3, 1])
    values[[0, 1, 2], 0] = np.nan
    values[0, 1] = np.nan
    values[2, 4] = np.nan
    np.testing.assert_allclose(nan_mean(values), [np.nan, 9.5, 2, -2 / 3, 0], equal_nan=True)
    assert nan_mean(1) == 1
    assert nan_mean([1, 2, 3, 0, -5]) == pytest.approx(0.2)
    np.testing.assert_allclose(nan_mean([1 + 2j, 3 + 4j]), 2 + 3j)


def test_python_regression_quantile_matches_midpoint_empirical_interpolation():
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


def test_python_regression_shuffle_returns_one_based_permutation_and_exact_inverse_for_any_dimension():
    generator = np.random.default_rng(17)
    for shape, dim in [((32, 100), 1), ((32, 100), 2), ((4, 8, 10), 2), ((4, 8, 10), 3)]:
        data = np.arange(np.prod(shape)).reshape(shape)
        shuffled, permutation, inverse = shuffle(data, dim, rng=generator)
        axis = dim - 1
        positions = permutation - 1
        inverse_positions = inverse - 1
        assert permutation.min() == 1
        assert permutation.max() == shape[axis]
        np.testing.assert_array_equal(shuffled, np.take(data, positions, axis=axis))
        np.testing.assert_array_equal(np.take(shuffled, inverse_positions, axis=axis), data)
        np.testing.assert_array_equal(positions[inverse_positions], np.arange(shape[axis]))
    row = np.arange(5).reshape(1, 5)
    shuffled, permutation, _inverse = shuffle(row, rng=np.random.default_rng(1))
    assert permutation.min() == 1 and permutation.max() == 5
    np.testing.assert_array_equal(shuffled, row[:, permutation - 1])
    column = np.arange(5).reshape(5, 1)
    shuffled, permutation, _inverse = shuffle(column, rng=np.random.default_rng(1))
    assert permutation.min() == 1 and permutation.max() == 5
    np.testing.assert_array_equal(shuffled, column[permutation - 1])
    leading = np.arange(5).reshape(1, 1, 1, 5)
    shuffled, permutation, _inverse = shuffle(leading, rng=np.random.default_rng(1))
    assert permutation.min() == 1 and permutation.max() == 5
    np.testing.assert_array_equal(shuffled, np.take(leading, permutation - 1, axis=3))
    with pytest.raises(ValueError, match="integer"):
        shuffle(np.ones((2, 2)), dim=0.5)
    with pytest.raises(ValueError, match="out of range"):
        shuffle(np.ones((2, 2)), dim=0)
    with pytest.raises(ValueError, match="out of range"):
        shuffle(np.ones((2, 2)), dim=3)


@eeglab_test(_source("celltomat"), "test_pass_1d_num")
@eeglab_test(_source("celltomat"), "test_pass_empty")
def test_reference_celltomat(eeglab_backend):
    cells = np.array([[1.0, 2.0, 3.0]], dtype=object)
    _assert_near(eeglab_backend("celltomat", cells), np.array([[1.0, 2.0, 3.0]]))
    _assert_near(eeglab_backend("celltomat", np.empty((0, 0), dtype=object)), np.empty((0, 0)))


@eeglab_test(_source("eyelike"), "test_pass_diag_zero_1")
@eeglab_test(_source("eyelike"), "test_pass_eye")
@eeglab_test(_source("eyelike"), "test_pass_eye_scale")
@eeglab_test(_source("eyelike"), "test_pass_general")
def test_reference_eyelike(eeglab_backend):
    # diag_zero_1 only calls the function; diag_zero_2 has no source statements.
    eeglab_backend("eyelike", np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float), nargout=3)
    for matrix in (np.eye(3), np.diag([3.0, 2.0, 9.0]), np.arange(1.0, 10.0).reshape(3, 3)):
        normalized, scale, permutation = eeglab_backend("eyelike", matrix, nargout=3)
        _assert_near(np.linalg.inv(permutation) @ np.linalg.inv(scale) @ normalized, matrix)
        _assert_near(np.diag(normalized)[:, None], np.ones((3, 1)))


@eeglab_test(_source("fastif"), "test_pass_complex")
@eeglab_test(_source("fastif"), "test_pass_false")
@eeglab_test(_source("fastif"), "test_pass_not_bool")
@eeglab_test(_source("fastif"), "test_pass_true")
def test_reference_fastif(eeglab_backend):
    assert eeglab_backend("fastif", True, "yes", "no") == "yes"
    assert eeglab_backend("fastif", False, "yes", "no") == "no"
    assert eeglab_backend("fastif", "not boolean!", "yes", "no") == "yes"
    matrix = np.arange(1.0, 10.0).reshape(3, 3)
    _assert_near(eeglab_backend("fastif", True, np.diag(matrix)[:, None], matrix[:, [2]]), [[1.0], [5.0], [9.0]])


@eeglab_test(_source("matsel"), "test_pass_general")
@eeglab_test(_source("matsel"), "test_pass_valid_frames")
def test_reference_matsel(eeglab_backend):
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    _assert_near(eeglab_backend("matsel", data, 0.0, 0.0), data)
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float)
    _assert_near(eeglab_backend("matsel", data, 2.0, 1.0), [[1.0, 3.0], [5.0, 7.0]])


@eeglab_test(_source("mattocell"), "test_pass_general")
def test_reference_mattocell(eeglab_backend):
    # pass_empty_data is an empty upstream function, not an empty-input check.
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    cells = eeglab_backend("mattocell", data)
    assert cells.shape == (2, 3)
    assert cells.dtype == object
    for index in np.ndindex(cells.shape):
        np.testing.assert_array_equal(cells[index], [[data[index]]])


@eeglab_test(_source("nan_mean"), "test_pass_general")
@eeglab_test(_source("nan_mean"), "test_pass_nan")
@eeglab_test(_source("nan_mean"), "test_pass_number")
@eeglab_test(_source("nan_mean"), "test_pass_row_vector")
def test_reference_nan_mean(eeglab_backend):
    data = np.array([[1, 2, 3, 0, -5], [1, 7, 3, 4, 5], [-2, 12, 0, -6, 3]], dtype=float)
    _assert_near(eeglab_backend("nan_mean", data), [[0, 7, 2, -2 / 3, 1]])
    data[:, 0] = np.nan
    data[0, 1] = np.nan
    data[2, 4] = np.nan
    _assert_near(eeglab_backend("nan_mean", data), [[np.nan, 9.5, 2, -2 / 3, 0]])
    _assert_near(eeglab_backend("nan_mean", 1.0), [[1.0]])
    _assert_near(eeglab_backend("nan_mean", np.array([[1, 2, 3, 0, -5]], dtype=float)), [[0.2]])


@eeglab_test(_source("quantile"), "test_pass_column")
@eeglab_test(_source("quantile"), "test_pass_general")
@eeglab_test(_source("quantile"), "test_pass_matrix")
@eeglab_test(_source("quantile"), "test_pass_number")
@eeglab_test(_source("quantile"), "test_pass_unsorted")
def test_reference_quantile(eeglab_backend):
    probabilities = np.array([[0, 0.125, 0.25, 0.5, 0.75, 0.875, 1]], dtype=float)
    expected = np.array([[1, 1, 1.5, 3.5, 7.5, 10, 10]], dtype=float)
    for data in (
        np.array([[1, 2, 5, 10]], dtype=float),
        np.array([[1], [2], [5], [10]], dtype=float),
        np.array([[10, 2, 5, 1]], dtype=float),
    ):
        actual = eeglab_backend("quantile", data, probabilities)
        # These source cases explicitly accept either row or column output.
        _assert_near(actual, expected if actual.shape == expected.shape else expected.T)
    actual = eeglab_backend("quantile", 17.0, np.array([[0.25, 0.5, 0.75]]))
    expected = np.array([[17.0, 17.0, 17.0]])
    _assert_near(actual, expected if actual.shape == expected.shape else expected.T)
    data = np.array([[-1, 0, -4, -2], [1, 2, 5, 8], [3, 7, 6, 10]], dtype=float)
    expected = [[-1, 0, -4, -2], [-0.5, 0.5, -1.75, 0.5], [1, 2, 5, 8], [2.5, 5.75, 5.75, 9.5], [3, 7, 6, 10]]
    _assert_near(eeglab_backend("quantile", data, np.array([[0, 0.25, 0.5, 0.75, 1]])), expected)


@eeglab_test(_source("shuffle"), "test_test_shuffle")
def test_reference_shuffle(eeglab_backend):
    rng = np.random.default_rng(17)
    data = rng.random((32, 100))
    for axis in (1, 2):
        shuffled, permutation, inverse = eeglab_backend("shuffle", data, float(axis), nargout=3)
        _assert_near(shuffled, np.take(data, permutation.astype(int).ravel() - 1, axis=axis - 1))
        _assert_near(np.take(shuffled, inverse.astype(int).ravel() - 1, axis=axis - 1), data)
    for axis in (2, 3):
        data = rng.random((32, 80, 100))
        shuffled, permutation, inverse = eeglab_backend("shuffle", data, float(axis), nargout=3)
        _assert_near(shuffled, np.take(data, permutation.astype(int).ravel() - 1, axis=axis - 1))
        _assert_near(np.take(shuffled, inverse.astype(int).ravel() - 1, axis=axis - 1), data)
