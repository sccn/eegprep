import numpy as np
import pytest
from eegprep.functions.sigprocfunc.rmbase import rmbase

def test_rmbase_dtype_preservation():
    chans, frames, epochs = 2, 10, 5
    # Float32 input
    data_f32 = np.random.randn(chans, frames, epochs).astype(np.float32)
    out_f32 = rmbase(data_f32, frames=frames)
    assert out_f32.dtype == np.float32

    # Float64 input
    data_f64 = np.random.randn(chans, frames, epochs).astype(np.float64)
    out_f64 = rmbase(data_f64, frames=frames)
    assert out_f64.dtype == np.float64

    # Integer input (should promote to float64)
    data_int = np.random.randint(0, 100, (chans, frames, epochs)).astype(np.int32)
    out_int = rmbase(data_int, frames=frames)
    assert out_int.dtype == np.float64

def test_rmbase_nan_equivalence():
    chans, frames, epochs = 2, 10, 5
    data = np.random.randn(chans, frames, epochs)
    data[0, 0, 0] = np.nan

    # Reference implementation (old style loop logic)
    def ref_rmbase(data, frames):
        array = np.asarray(data)
        matrix = array.transpose(0, 2, 1).reshape(array.shape[0], -1) if array.ndim == 3 else array
        chans, total_frames = matrix.shape
        epochs = total_frames // frames
        output = matrix.astype(np.float64, copy=True)
        means = np.zeros((chans, epochs))
        for epoch in range(epochs):
            start = epoch * frames
            stop = start + frames
            mean = np.nanmean(matrix[:, start:stop], axis=1, keepdims=True)
            means[:, epoch : epoch + 1] = mean
            output[:, start:stop] = matrix[:, start:stop] - mean
        return output.reshape(array.shape[0], array.shape[2], array.shape[1]).transpose(0, 2, 1)

    out_vectorized = rmbase(data, frames=frames)
    out_ref = ref_rmbase(data, frames=frames)

    np.testing.assert_allclose(out_vectorized, out_ref, equal_nan=True)

def test_rmbase_basevector_nan():
    chans, frames, epochs = 1, 10, 1
    data = np.array([[[1.0, 2.0, np.nan, 4.0, 5.0]]]).transpose(0, 2, 1) # (1, 5, 1)
    # Baseline on samples 1, 2, 3 (1-based)
    # Samples are 1.0, 2.0, nan. Mean is 1.5.
    out = rmbase(data, frames=5, basevector=[1, 2, 3])
    expected = np.array([[[1.0 - 1.5, 2.0 - 1.5, np.nan, 4.0 - 1.5, 5.0 - 1.5]]]).transpose(0, 2, 1)
    np.testing.assert_allclose(out, expected, equal_nan=True)
