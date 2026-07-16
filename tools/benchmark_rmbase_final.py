"""Benchmark for rmbase performance."""
import numpy as np
import time
from eegprep.functions.sigprocfunc.rmbase import rmbase

def benchmark_rmbase():
    # High-density, long recording scenario (e.g., 256 channels, 1 hour at 500Hz)
    # 1 hour * 500 Hz = 1,800,000 points.
    # 256 channels * 1,800,000 points * 4 bytes (float32) = 1.8 GB.
    # We'll use a slightly smaller version to fit in sandbox memory but still be substantial.
    chans = 128
    total_pnts = 500000
    frames = 500
    epochs = total_pnts // frames

    data = np.random.randn(chans, total_pnts).astype(np.float32)

    print(f"Benchmarking rmbase with {chans} channels, {total_pnts} points ({epochs} epochs of {frames} frames)")
    print(f"Data size: {data.nbytes / 1e6:.1f} MB")

    # Warm up
    _ = rmbase(data, frames=frames)

    start = time.perf_counter()
    iterations = 20
    for _ in range(iterations):
        _ = rmbase(data, frames=frames)
    end = time.perf_counter()

    avg_time = (end - start) / iterations
    print(f"Average time per call: {avg_time:.4f}s")

if __name__ == "__main__":
    benchmark_rmbase()
