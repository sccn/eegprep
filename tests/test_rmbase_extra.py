from __future__ import annotations

import numpy as np
import pytest

from eegprep.functions.sigprocfunc.rmbase import rmbase


def _legacy_rmbase(
    data: np.ndarray,
    frames: int,
    basevector: list[int] | int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the pre-vectorization loop as a numerical reference."""
    array = np.asarray(data)
    original_shape = array.shape
    matrix = array.transpose(0, 2, 1).reshape(array.shape[0], -1) if array.ndim == 3 else array
    channels, total_frames = matrix.shape
    epochs = total_frames // frames
    baseline = None if basevector == 0 else np.asarray(basevector, dtype=int) - 1

    output = matrix.astype(np.float64, copy=True) if not np.issubdtype(matrix.dtype, np.floating) else matrix.copy()
    means = np.zeros((channels, epochs), dtype=np.result_type(matrix.dtype, np.float64))
    for epoch in range(epochs):
        start = epoch * frames
        stop = start + frames
        if baseline is None:
            mean = np.nanmean(matrix[:, start:stop], axis=1, keepdims=True, dtype=np.float64)
        else:
            mean = np.nanmean(matrix[:, start + baseline], axis=1, keepdims=True, dtype=np.float64)
        means[:, epoch : epoch + 1] = mean
        output[:, start:stop] = output[:, start:stop] - mean

    if array.ndim == 3:
        output = output.reshape(original_shape[0], original_shape[2], original_shape[1]).transpose(0, 2, 1)
    return output.reshape(original_shape), means


def test_rmbase_3d_default_frames_matches_legacy_grand_mean():
    data = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
    total_frames = data.shape[1] * data.shape[2]

    expected, expected_means = _legacy_rmbase(data, frames=total_frames)
    actual, actual_means = rmbase(data, return_mean=True)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_means, expected_means)
    assert actual_means.shape == (data.shape[0], 1)


@pytest.mark.parametrize("shape", [(3, 85), (3, 17, 5)])
@pytest.mark.parametrize("basevector", [0, [1, 4, 7, 11]])
def test_rmbase_float32_matches_legacy_rounding(shape: tuple[int, ...], basevector: list[int] | int):
    rng = np.random.default_rng(268)
    data = (rng.standard_normal(shape) * 1_000).astype(np.float32)
    original = data.copy()

    expected, expected_means = _legacy_rmbase(data, frames=17, basevector=basevector)
    actual, actual_means = rmbase(data, frames=17, basevector=basevector, return_mean=True)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_means, expected_means)
    np.testing.assert_array_equal(data, original)
    assert actual.dtype == np.float32
    assert actual_means.dtype == np.float64


@pytest.mark.parametrize("dtype", [np.float64, np.int16])
def test_rmbase_matches_legacy_for_other_dtypes(dtype: type[np.generic]):
    rng = np.random.default_rng(269)
    if np.issubdtype(dtype, np.integer):
        data = rng.integers(-2_000, 2_000, size=(4, 13, 3), dtype=dtype)
    else:
        data = (rng.standard_normal((4, 13, 3)) * 1_000).astype(dtype)

    expected, expected_means = _legacy_rmbase(data, frames=13, basevector=[2, 5, 9])
    actual, actual_means = rmbase(data, frames=13, basevector=[2, 5, 9], return_mean=True)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_means, expected_means)
    assert actual.dtype == (np.dtype(np.float64) if np.issubdtype(dtype, np.integer) else dtype)


def test_rmbase_preserves_nan_results_and_warning_behavior():
    data = np.arange(16, dtype=np.float32).reshape(2, 8)
    data[0, :2] = np.nan
    data[1, 4:6] = np.nan

    with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
        actual, actual_means = rmbase(data, frames=4, basevector=[1, 2], return_mean=True)
    with pytest.warns(RuntimeWarning, match="Mean of empty slice"):
        expected, expected_means = _legacy_rmbase(data, frames=4, basevector=[1, 2])

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_means, expected_means)
