"""Two-sided benchmark for rmbase: Legacy vs Optimized."""
import numpy as np
import time
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

    for epoch in range(epochs):
        start = epoch * frames
        stop = start + frames
        if baseline is None:
            mean = np.nanmean(matrix[:, start:stop], axis=1, keepdims=True, dtype=np.float64)
        else:
            mean = np.nanmean(matrix[:, start + baseline], axis=1, keepdims=True, dtype=np.float64)
        output[:, start:stop] = matrix[:, start:stop] - mean

    if array.ndim == 3:
        output = output.reshape(array.shape[0], array.shape[2], array.shape[1]).transpose(0, 2, 1)
    else:
        output = output.reshape(array.shape)
    return output

def run_benchmark(chans=128, total_pnts=1000000, frames=500, dtype=np.float32):
    epochs = total_pnts // frames
    data = np.random.randn(chans, total_pnts).astype(dtype)

    print(f"--- Benchmarking rmbase ({chans} chans, {total_pnts} pnts, {epochs} epochs, {dtype}) ---")

    # Legacy
    start = time.perf_counter()
    _ = legacy_rmbase_reference(data, frames=frames)
    legacy_time = time.perf_counter() - start
    print(f"Legacy loop implementation: {legacy_time:.4f}s")

    # Optimized
    # Warm up
    _ = rmbase(data, frames=frames)
    start = time.perf_counter()
    _ = rmbase(data, frames=frames)
    optimized_time = time.perf_counter() - start
    print(f"Optimized vectorized implementation: {optimized_time:.4f}s")

    speedup = legacy_time / optimized_time
    print(f"Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    # Test a few scenarios
    run_benchmark(chans=128, total_pnts=500000, frames=500, dtype=np.float32)
    run_benchmark(chans=32, total_pnts=1000000, frames=50, dtype=np.float64)
