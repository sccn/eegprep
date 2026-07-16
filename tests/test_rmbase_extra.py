import numpy as np
import pytest
from eegprep.functions.sigprocfunc.rmbase import rmbase

def legacy_rmbase_reference(data, frames=0, basevector=0):
    """Original loop-based implementation as a reference."""
    array = np.asarray(data)
    matrix = array.transpose(0, 2, 1).reshape(array.shape[0], -1) if array.ndim == 3 else array
    chans, total_frames = matrix.shape
    frames = int(frames or 0)
    if frames == 0:
        frames = total_frames
    epochs = total_frames // frames

    def _get_baseline_indices(bv, f):
        if bv is None or (isinstance(bv, (int, float)) and bv == 0):
            return None
        indices = np.asarray(bv, dtype=int)
        indices = indices[(indices >= 1) & (indices <= f)]
        return indices - 1

    baseline = _get_baseline_indices(basevector, frames)
    output = np.empty(matrix.shape, dtype=np.float64 if not np.issubdtype(matrix.dtype, np.floating) else matrix.dtype)
    means = np.zeros((chans, epochs), dtype=np.float64)

    for epoch in range(epochs):
        start = epoch * frames
        stop = start + frames
        if baseline is None:
            mean = np.nanmean(matrix[:, start:stop], axis=1, keepdims=True, dtype=np.float64)
        else:
            mean = np.nanmean(matrix[:, start + baseline], axis=1, keepdims=True, dtype=np.float64)
        means[:, epoch : epoch + 1] = mean
        output[:, start:stop] = matrix[:, start:stop] - mean

    if array.ndim == 3:
        output = output.reshape(array.shape[0], array.shape[2], array.shape[1]).transpose(0, 2, 1)
    else:
        output = output.reshape(array.shape)
    return output, means

@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
@pytest.mark.parametrize("has_nans", [False, True])
@pytest.mark.parametrize("basevector", [0, [1, 2, 3]])
def test_rmbase_comprehensive_parity(ndim, dtype, has_nans, basevector):
    chans, frames, epochs = 35, 10, 5 # 35 to test block boundary (block_size=32)
    if ndim == 2:
        shape = (chans, frames * epochs)
    else:
        shape = (chans, frames, epochs)

    if np.issubdtype(dtype, np.integer):
        data = np.random.randint(0, 100, shape).astype(dtype)
    else:
        data = np.random.randn(*shape).astype(dtype)

    if has_nans and not np.issubdtype(dtype, np.integer):
        data[0, 0] = np.nan

    out_new, means_new = rmbase(data, frames=frames, basevector=basevector, return_mean=True)
    out_ref, means_ref = legacy_rmbase_reference(data, frames=frames, basevector=basevector)

    # Verify dtypes
    expected_dtype = np.float64 if np.issubdtype(dtype, np.integer) else dtype
    assert out_new.dtype == expected_dtype
    assert means_new.dtype == np.float64

    # Verify values
    np.testing.assert_allclose(out_new, out_ref, equal_nan=True, atol=1e-7 if dtype == np.float32 else 1e-15)
    np.testing.assert_allclose(means_new, means_ref, equal_nan=True)

def test_rmbase_immutability():
    data = np.random.randn(5, 100).astype(np.float32)
    data_orig = data.copy()
    _ = rmbase(data, frames=10)
    np.testing.assert_array_equal(data, data_orig)

def test_rmbase_2d_3d_consistency():
    chans, frames, epochs = 2, 10, 3
    data_2d = np.random.randn(chans, frames * epochs)
    data_3d = data_2d.reshape(chans, epochs, frames).transpose(0, 2, 1) # (chans, frames, epochs)

    out_2d = rmbase(data_2d, frames=frames)
    out_3d = rmbase(data_3d, frames=frames)

    out_3d_flat = out_3d.transpose(0, 2, 1).reshape(chans, -1)
    np.testing.assert_allclose(out_2d, out_3d_flat)
