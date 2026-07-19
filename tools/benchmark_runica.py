"""Repeatable microbenchmarks for the allocation kernels optimized in RunICA.

Run from the repository root with ``python tools/benchmark_runica.py``.
This reports timings for human inspection; it is not a CI performance test.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from time import perf_counter

import numpy as np


Benchmark = Callable[[], object]


def _elapsed(operation: Benchmark, iterations: int) -> float:
    start = perf_counter()
    for _ in range(iterations):
        operation()
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


def _legacy_block_projection(weights: np.ndarray, blocks: np.ndarray) -> float:
    checksum = 0.0
    for block_data in blocks:
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            checksum += float(np.sum(weights @ block_data))
    return checksum


def _optimized_block_projection(weights: np.ndarray, blocks: np.ndarray) -> float:
    checksum = 0.0
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for block_data in blocks:
            checksum += float(np.sum(weights @ block_data))
    return checksum


def _guarded_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return left @ right


def _legacy_extended_update(
    weights: np.ndarray,
    identity: np.ndarray,
    signs: np.ndarray,
    activations: np.ndarray,
    projected: np.ndarray,
    learning_rate: float,
) -> np.ndarray:
    gradient = (
        identity
        - _guarded_matmul(_guarded_matmul(signs, activations), projected.T)
        - _guarded_matmul(projected, projected.T)
    )
    return weights + learning_rate * _guarded_matmul(gradient, weights)


def _optimized_extended_update(
    weights: np.ndarray,
    identity: np.ndarray,
    signs: np.ndarray,
    activations: np.ndarray,
    projected: np.ndarray,
    learning_rate: float,
) -> np.ndarray:
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        signed_activations = signs[:, np.newaxis] * activations
        gradient = identity - (signed_activations + projected) @ projected.T
        return weights + learning_rate * gradient @ weights


def _legacy_bias_projection(
    weights: np.ndarray,
    block_data: np.ndarray,
    bias: np.ndarray,
    ones: np.ndarray,
) -> np.ndarray:
    return _guarded_matmul(weights, block_data) + _guarded_matmul(bias, ones)


def _optimized_bias_projection(
    weights: np.ndarray,
    block_data: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return weights @ block_data + bias


def _report(label: str, legacy: Benchmark, optimized: Benchmark, iterations: int) -> None:
    np.testing.assert_allclose(legacy(), optimized(), rtol=1e-12, atol=1e-12)
    legacy_seconds = _elapsed(legacy, iterations)
    optimized_seconds = _elapsed(optimized, iterations)
    print(
        f"{label}: legacy={legacy_seconds * 1e3:.3f} ms "
        f"optimized={optimized_seconds * 1e3:.3f} ms "
        f"speedup={legacy_seconds / optimized_seconds:.2f}x"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--frames", type=int, default=30_720)
    parser.add_argument("--block", type=int, default=52)
    parser.add_argument("--center-iterations", type=int, default=20)
    parser.add_argument("--projection-iterations", type=int, default=10)
    parser.add_argument("--training-iterations", type=int, default=2_000)
    args = parser.parse_args()

    rng = np.random.default_rng(42)
    data = rng.standard_normal((args.channels, args.frames))
    weights = rng.standard_normal((args.channels, args.channels)) * 0.01
    block_data = rng.standard_normal((args.channels, args.block))
    projected = weights @ block_data
    activations = np.tanh(projected)
    signs = rng.choice((-1.0, 1.0), size=args.channels)
    signs_matrix = np.diag(signs)
    identity = np.eye(args.channels) * args.block
    bias = rng.standard_normal((args.channels, 1)) * 0.01
    ones = np.ones((1, args.block))
    training_frames = args.frames // args.block * args.block
    blocks = data[:, :training_frames].reshape(args.channels, -1, args.block).transpose(1, 0, 2)
    learning_rate = 0.0001

    _report(
        "channel centering",
        lambda: _legacy_center(data),
        lambda: _optimized_center(data),
        args.center_iterations,
    )
    _report(
        "training block projection",
        lambda: _legacy_block_projection(weights, blocks),
        lambda: _optimized_block_projection(weights, blocks),
        args.projection_iterations,
    )
    _report(
        "extended weight update",
        lambda: _legacy_extended_update(
            weights,
            identity,
            signs_matrix,
            activations,
            projected,
            learning_rate,
        ),
        lambda: _optimized_extended_update(
            weights,
            identity,
            signs,
            activations,
            projected,
            learning_rate,
        ),
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
