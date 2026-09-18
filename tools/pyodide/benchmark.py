"""Run the deterministic Phase 2 ICA and matrix-multiplication benchmarks."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.linalg import blas as scipy_blas
from threadpoolctl import threadpool_limits

from eegprep.functions.sigprocfunc.runica import runica
from eegprep.functions.sigprocfunc.runica_matmul import BACKEND as RUNICA_MATMUL_BACKEND


SEED = 375
CHANNELS = 64
FRAMES = 15_000
MAX_ICA_ITERATIONS = 512
WARMUP_RUNS = 1
MEASURED_RUNS = 3
RUNICA_BLOCK = 49
MATMUL_REPETITIONS = 3
THREAD_COUNT = 1

# Pyodide 0.29.5 has no pthread support. Keep the native comparison on the
# same one-thread budget; browser concurrency belongs at the Web Worker/job
# level rather than inside one ICA call.

# These are the two products called out in runica's training loop for 64
# channels and the default block heuristic at 15,000 frames: the activation
# product and the final square weight-update product.
MATMUL_CASES = (
    {
        "name": "activation",
        "left_shape": (CHANNELS, CHANNELS),
        "right_shape": (CHANNELS, RUNICA_BLOCK),
    },
    {
        "name": "weight_update",
        "left_shape": (CHANNELS, CHANNELS),
        "right_shape": (CHANNELS, CHANNELS),
    },
)


def benchmark_record(
    *,
    algorithm: str,
    platform: str,
    shape: tuple[int, int],
    seed: int,
    seconds: float,
    iterations: int | None,
    converged: bool | None,
    backend: str,
    error: str | None = None,
) -> dict[str, Any]:
    """Return one JSON-serializable benchmark measurement."""
    return {
        "algorithm": algorithm,
        "platform": platform,
        "shape": list(shape),
        "seed": seed,
        "seconds": float(seconds),
        "iterations": None if iterations is None else int(iterations),
        "converged": converged,
        "backend": backend,
        "error": error,
    }


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot calculate a median from no measurements")
    return float(statistics.median(values))


def _runica_once(data: np.ndarray) -> tuple[float, int, bool]:
    result: tuple[Any, ...]
    start = time.perf_counter()
    result = runica(
        data.copy(),
        extended=1,
        maxsteps=MAX_ICA_ITERATIONS,
        verbose=False,
        rndreset="off",
    )
    seconds = time.perf_counter() - start
    iterations = len(result[5])
    return seconds, int(iterations), int(iterations) < MAX_ICA_ITERATIONS


def _run_picard_once(data: np.ndarray) -> tuple[float, int, bool]:
    # Keep this call aligned with eeg_picard.py. return_n_iter is the underlying
    # Picard API's telemetry and does not change production behavior.
    from picard import picard

    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        _weighting, _unmixing, _sources, iterations = picard(
            data.copy(),
            ortho=False,
            fun="tanh",
            verbose=False,
            m=10,
            max_iter=MAX_ICA_ITERATIONS,
            tol=1e-7,
            centering=True,
            whiten=True,
            w_init=np.eye(CHANNELS),
            return_n_iter=True,
            random_state=SEED,
        )
    seconds = time.perf_counter() - start
    did_not_converge = any("did not converge" in str(item.message).lower() for item in caught_warnings)
    return seconds, int(iterations), not did_not_converge and int(iterations) < MAX_ICA_ITERATIONS


def _ica_summary(
    *, platform: str, data: np.ndarray, name: str, operation: Callable[[np.ndarray], tuple[float, int, bool]]
) -> dict[str, Any]:
    for _ in range(WARMUP_RUNS):
        operation(data)

    records = []
    for _ in range(MEASURED_RUNS):
        try:
            seconds, iterations, converged = operation(data)
            records.append(
                benchmark_record(
                    algorithm=name,
                    platform=platform,
                    shape=(CHANNELS, FRAMES),
                    seed=SEED,
                    seconds=seconds,
                    iterations=iterations,
                    converged=converged,
                    backend=RUNICA_MATMUL_BACKEND if name == "runica" else "picard",
                )
            )
        except Exception as exc:  # Keep the raw failure visible before failing the gate.
            records.append(
                benchmark_record(
                    algorithm=name,
                    platform=platform,
                    shape=(CHANNELS, FRAMES),
                    seed=SEED,
                    seconds=0.0,
                    iterations=None,
                    converged=None,
                    backend=RUNICA_MATMUL_BACKEND if name == "runica" else "picard",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    successful = [record for record in records if record["error"] is None]
    return {
        "algorithm": name,
        "backend": RUNICA_MATMUL_BACKEND if name == "runica" else "picard",
        "warmup_runs": WARMUP_RUNS,
        "measured_runs": MEASURED_RUNS,
        "runs": records,
        "median_seconds": _median([float(record["seconds"]) for record in successful]) if successful else None,
        "median_iterations": _median([float(record["iterations"]) for record in successful]) if successful else None,
        "all_converged": bool(successful) and all(record["converged"] for record in successful),
    }


def _matmul_operation(name: str, left: np.ndarray, right: np.ndarray) -> Callable[[], np.ndarray]:
    if name == "numpy_matmul":
        return lambda: left @ right
    if name in ("dgemm", "sgemm"):
        blas_operation = getattr(scipy_blas, name)
        return lambda: blas_operation(alpha=1.0, a=left, b=right)
    raise ValueError(f"Unknown matrix multiplication operation: {name}")


def _matmul_summary(*, platform: str, rng: np.random.RandomState) -> list[dict[str, Any]]:
    summaries = []
    for case in MATMUL_CASES:
        for dtype, blas_name in ((np.float64, "dgemm"), (np.float32, "sgemm")):
            left = rng.standard_normal(case["left_shape"]).astype(dtype)
            right = rng.standard_normal(case["right_shape"]).astype(dtype)
            for operation_name in ("numpy_matmul", blas_name):
                operation = _matmul_operation(operation_name, left, right)
                measurements = []
                checksum = 0.0
                with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                    for _ in range(WARMUP_RUNS):
                        operation()
                    for _ in range(MEASURED_RUNS):
                        start = time.perf_counter()
                        for _ in range(MATMUL_REPETITIONS):
                            output = operation()
                            checksum += float(output.flat[0])
                        measurements.append(time.perf_counter() - start)
                summaries.append(
                    {
                        "case": case["name"],
                        "left_shape": list(case["left_shape"]),
                        "right_shape": list(case["right_shape"]),
                        "dtype": np.dtype(dtype).name,
                        "operation": operation_name,
                        "platform": platform,
                        "seed": SEED,
                        "warmup_runs": WARMUP_RUNS,
                        "measured_runs": MEASURED_RUNS,
                        "repetitions": MATMUL_REPETITIONS,
                        "seconds": measurements,
                        "median_seconds": _median(measurements),
                        "checksum": checksum,
                    }
                )
    return summaries


def run_benchmark(platform: str) -> dict[str, Any]:
    """Run all Phase 2 measurements on ``platform`` and return the report."""
    rng = np.random.RandomState(SEED)
    data = rng.standard_normal((CHANNELS, FRAMES)).astype(np.float64)
    with threadpool_limits(limits=THREAD_COUNT):
        ica = {
            "runica": _ica_summary(platform=platform, data=data, name="runica", operation=_runica_once),
            "picard": _ica_summary(platform=platform, data=data, name="picard", operation=_run_picard_once),
        }
        matmul = _matmul_summary(platform=platform, rng=rng)

    report = {
        "schema_version": 1,
        "platform": platform,
        "seed": SEED,
        "shape": [CHANNELS, FRAMES],
        "max_ica_iterations": MAX_ICA_ITERATIONS,
        "thread_count": THREAD_COUNT,
        "threading_mode": "single-threaded",
        "runica_block": RUNICA_BLOCK,
        "warmup_runs": WARMUP_RUNS,
        "measured_runs": MEASURED_RUNS,
        "ica": ica,
        "matmul": matmul,
    }
    errors = [record["error"] for summary in ica.values() for record in summary["runs"] if record["error"] is not None]
    if errors:
        raise RuntimeError("Benchmark operation failed: " + "; ".join(errors))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("native", "pyodide"), required=True)
    args = parser.parse_args()
    print(json.dumps(run_benchmark(args.platform), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
