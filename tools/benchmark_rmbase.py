"""Compare vectorized ``rmbase`` with its pre-vectorization loop."""

from __future__ import annotations

import argparse
import gc
import statistics
import time
import tracemalloc
from collections.abc import Callable

import numpy as np

from eegprep.functions.sigprocfunc.rmbase import rmbase


def _legacy_rmbase(data: np.ndarray, frames: int) -> np.ndarray:
    """Reproduce the implementation replaced by the optimization."""
    original_shape = data.shape
    matrix = data.transpose(0, 2, 1).reshape(data.shape[0], -1) if data.ndim == 3 else data
    channels, total_frames = matrix.shape
    epochs = total_frames // frames
    output = matrix.copy()

    for epoch in range(epochs):
        start = epoch * frames
        stop = start + frames
        mean = np.nanmean(matrix[:, start:stop], axis=1, keepdims=True, dtype=np.float64)
        output[:, start:stop] = output[:, start:stop] - mean

    if data.ndim == 3:
        return output.reshape(channels, original_shape[2], original_shape[1]).transpose(0, 2, 1)
    return output


def _median_seconds(operation: Callable[[], np.ndarray], repeats: int) -> float:
    timings: list[float] = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        operation()
        timings.append(time.perf_counter() - start)
    return statistics.median(timings)


def _peak_bytes(operation: Callable[[], np.ndarray]) -> int:
    gc.collect()
    tracemalloc.start()
    operation()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=268)
    args = parser.parse_args()
    if min(args.channels, args.frames, args.epochs, args.repeats) < 1:
        parser.error("channels, frames, epochs, and repeats must be positive")

    rng = np.random.default_rng(args.seed)
    data = rng.standard_normal((args.channels, args.frames, args.epochs), dtype=np.float32)

    def legacy() -> np.ndarray:
        return _legacy_rmbase(data, args.frames)

    def vectorized() -> np.ndarray:
        return rmbase(data, frames=args.frames)

    expected = legacy()
    actual = vectorized()
    np.testing.assert_array_equal(actual, expected)

    legacy_seconds = _median_seconds(legacy, args.repeats)
    vectorized_seconds = _median_seconds(vectorized, args.repeats)
    legacy_peak = _peak_bytes(legacy)
    vectorized_peak = _peak_bytes(vectorized)

    print(
        f"Input: {args.channels} channels x {args.frames} frames x {args.epochs} epochs "
        f"({data.nbytes / 2**20:.1f} MiB, {data.dtype})"
    )
    print(f"Median of {args.repeats} runs: legacy={legacy_seconds:.4f}s, vectorized={vectorized_seconds:.4f}s")
    print(f"Observed speed ratio: {legacy_seconds / vectorized_seconds:.2f}x")
    print(f"Tracemalloc peak: legacy={legacy_peak / 2**20:.1f} MiB, vectorized={vectorized_peak / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
