"""Benchmark the allocation and matrix kernels optimized in ``runica``.

This is a repeatable microbenchmark, not a CI performance assertion. Run it
from the repository root with ``python tools/benchmark_runica.py``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from time import perf_counter

import numpy as np


ArrayFactory = Callable[[], np.ndarray]


def _elapsed(factory: ArrayFactory, iterations: int) -> float:
    start = perf_counter()
    for _ in range(iterations):
        factory()
    return (perf_counter() - start) / iterations


def _legacy_center(data: np.ndarray) -> np.ndarray:
    output = data.copy()
    rowmeans = np.mean(output, axis=1)
    for row in range(output.shape[0]):
        output[row, :] -= rowmeans[row]
    return output


def _optimized_center(data: np.ndarray) -> np.ndarray:
    output = data.copy()
    output -= np.mean(output, axis=1, keepdims=True)
    return output


def _legacy_extended_update(
    weights: np.ndarray,
    identity: np.ndarray,
    signs: np.ndarray,
    activations: np.ndarray,
    projected: np.ndarray,
    learning_rate: float,
) -> np.ndarray:
    gradient = identity - signs @ activations @ projected.T - projected @ projected.T
    return weights + learning_rate * gradient @ weights


def _optimized_extended_update(
    weights: np.ndarray,
    identity: np.ndarray,
    signs: np.ndarray,
    activations: np.ndarray,
    projected: np.ndarray,
    learning_rate: float,
) -> np.ndarray:
    signed_activations = np.diag(signs)[:, np.newaxis] * activations
    gradient = identity - (signed_activations + projected) @ projected.T
    return weights + learning_rate * gradient @ weights


def _legacy_bias_projection(
    weights: np.ndarray, block_data: np.ndarray, bias: np.ndarray, ones: np.ndarray
) -> np.ndarray:
    return weights @ block_data + bias @ ones


def _optimized_bias_projection(weights: np.ndarray, block_data: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return weights @ block_data + bias


def _report(label: str, legacy: ArrayFactory, optimized: ArrayFactory, iterations: int) -> None:
    np.testing.assert_allclose(legacy(), optimized(), rtol=1e-12, atol=1e-12)
    legacy_seconds = _elapsed(legacy, iterations)
    optimized_seconds = _elapsed(optimized, iterations)
    print(
        f"{label}: legacy={legacy_seconds * 1e3:.3f} ms "
        f"optimized={optimized_seconds * 1e3:.3f} ms speedup={legacy_seconds / optimized_seconds:.2f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--frames", type=int, default=30_720)
    parser.add_argument("--block", type=int, default=52)
    parser.add_argument("--center-iterations", type=int, default=20)
    parser.add_argument("--training-iterations", type=int, default=2_000)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    data = rng.standard_normal((args.channels, args.frames))
    weights = rng.standard_normal((args.channels, args.channels)) * 0.01
    block_data = rng.standard_normal((args.channels, args.block))
    projected = weights @ block_data
    activations = np.tanh(projected)
    signs = np.diag(rng.choice((-1.0, 1.0), size=args.channels))
    identity = np.eye(args.channels) * args.block
    bias = rng.standard_normal((args.channels, 1)) * 0.01
    ones = np.ones((1, args.block))
    learning_rate = 0.0001

    _report(
        "channel centering",
        lambda: _legacy_center(data),
        lambda: _optimized_center(data),
        args.center_iterations,
    )
    _report(
        "extended weight update",
        lambda: _legacy_extended_update(weights, identity, signs, activations, projected, learning_rate),
        lambda: _optimized_extended_update(weights, identity, signs, activations, projected, learning_rate),
        args.training_iterations,
    )
    _report(
        "bias projection",
        lambda: _legacy_bias_projection(weights, block_data, bias, ones),
        lambda: _optimized_bias_projection(weights, block_data, bias),
        args.training_iterations,
    )


if __name__ == "__main__":
    main()
