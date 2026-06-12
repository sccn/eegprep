"""Surrogate resampling helper."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from eegprep.functions.statistics._shared import condition_grid, flatten_grid, normalize_method, rng_from_seed


@dataclass(frozen=True)
class SurrogateDistribution:
    """Surrogate condition grids produced by permutation or bootstrap."""

    samples: tuple[tuple[tuple[np.ndarray, ...], ...], ...]

    def __iter__(self) -> Iterator[tuple[tuple[np.ndarray, ...], ...]]:
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)


def surrogdistrib(
    data: Any,
    *,
    method: str = "perm",
    pairing: str = "on",
    naccu: int = 1,
    axis: int = -1,
    rng: np.random.Generator | int | None = None,
) -> SurrogateDistribution:
    """Build bootstrap or permutation surrogate condition grids.

    Args:
        data: One- or two-dimensional sequence of condition arrays.
        method: ``"perm"``/``"permutation"`` or ``"bootstrap"``.
        pairing: ``"on"`` to preserve case identity across conditions or
            ``"off"`` to resample from the pooled case axis.
        naccu: Number of surrogate grids to generate.
        axis: Axis in each condition array that stores cases.
        rng: Optional NumPy generator or seed for deterministic resampling.
    """

    method_name = normalize_method(method)
    if method_name == "param":
        raise ValueError("surrogdistrib only supports permutation or bootstrap methods")
    pairing_name = pairing.lower()
    if pairing_name not in {"on", "off"}:
        raise ValueError("pairing must be 'on' or 'off'")
    count = int(naccu)
    if count < 1:
        raise ValueError("naccu must be at least 1")

    generator = rng_from_seed(rng)
    grid = condition_grid(data, axis=axis, min_cases=1)
    samples = tuple(
        _resampled_grid(grid, bootstrap=method_name == "bootstrap", paired=pairing_name == "on", rng=generator)
        for _ in range(count)
    )
    return SurrogateDistribution(samples)


def _resampled_grid(
    grid: tuple[tuple[np.ndarray, ...], ...],
    *,
    bootstrap: bool,
    paired: bool,
    rng: np.random.Generator,
) -> tuple[tuple[np.ndarray, ...], ...]:
    arrays = flatten_grid(grid)
    feature_shape = arrays[0].shape[:-1]
    for index, array in enumerate(arrays):
        if array.shape[:-1] != feature_shape:
            raise ValueError(f"condition {index} has feature shape {array.shape[:-1]}, expected {feature_shape}")
    counts = [array.shape[-1] for array in arrays]

    if paired:
        if len(set(counts)) != 1:
            raise ValueError("paired surrogate resampling requires equal case counts")
        resampled = _paired_resample(arrays, bootstrap=bootstrap, rng=rng)
    else:
        resampled = _unpaired_resample(arrays, counts, bootstrap=bootstrap, rng=rng)

    iterator = iter(resampled)
    return tuple(tuple(next(iterator) for _column in row) for row in grid)


def _paired_resample(
    arrays: Sequence[np.ndarray],
    *,
    bootstrap: bool,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    n_conditions = len(arrays)
    n_cases = arrays[0].shape[-1]
    output = [np.empty_like(array) for array in arrays]
    for case_index in range(n_cases):
        source_conditions = (
            rng.integers(0, n_conditions, size=n_conditions) if bootstrap else rng.permutation(n_conditions)
        )
        for target_condition, source_condition in enumerate(source_conditions):
            output[target_condition][..., case_index] = arrays[int(source_condition)][..., case_index]
    return output


def _unpaired_resample(
    arrays: Sequence[np.ndarray],
    counts: Sequence[int],
    *,
    bootstrap: bool,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    pooled = np.concatenate(arrays, axis=-1)
    total_cases = pooled.shape[-1]
    if bootstrap:
        indices = rng.integers(0, total_cases, size=total_cases)
    else:
        indices = rng.permutation(total_cases)

    output = []
    start = 0
    for count in counts:
        stop = start + count
        output.append(np.take(pooled, indices[start:stop], axis=-1))
        start = stop
    return output


__all__ = ["SurrogateDistribution", "surrogdistrib"]
