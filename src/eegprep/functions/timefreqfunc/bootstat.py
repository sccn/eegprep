"""Bootstrap threshold helpers for time-frequency outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


Statistic = Callable[..., np.ndarray]


@dataclass(frozen=True)
class BootstrapResult:
    """Bootstrap thresholds and accumulated surrogate statistics."""

    thresholds: np.ndarray
    surrogates: np.ndarray


def bootstat(
    args: Any,
    statistic: Statistic,
    *,
    alpha: float = 0.05,
    naccu: int = 200,
    bootside: str = "both",
    boottype: str = "shuffle",
    basevect: Any = None,
    shuffledim: Any = None,
    rng: Any = None,
) -> BootstrapResult:
    """Accumulate surrogate statistics and return EEGLAB-style thresholds.

    This Python helper intentionally accepts a callable statistic instead of
    MATLAB expression strings. It covers the time-frequency bootstrap paths used
    by EEGPrep while avoiding runtime string evaluation.
    """
    arrays = _args(args)
    rng = np.random.default_rng(rng)
    selected = [_select_basevect(array, basevect) for array in arrays]
    dims = _shuffle_dims(shuffledim, selected[0].ndim, boottype=boottype)
    surrogates = []
    for _ in range(int(naccu)):
        boot_args = [_resample_array(array, rng, boottype=boottype, shuffledim=dims) for array in selected]
        surrogates.append(np.asarray(statistic(*boot_args)))
    accumulated = np.stack(surrogates, axis=0)
    thresholds = bootstrap_threshold(accumulated, alpha=alpha, bootside=bootside)
    return BootstrapResult(thresholds=thresholds, surrogates=accumulated)


def bootstrap_threshold(surrogates: Any, *, alpha: float = 0.05, bootside: str = "both") -> np.ndarray:
    """Return lower/upper or upper-only thresholds from accumulated surrogates."""
    values = np.asarray(surrogates)
    if values.ndim < 1:
        raise ValueError("surrogates must contain an accumulation axis")
    if np.iscomplexobj(values):
        values = np.abs(values)
    sorted_values = np.sort(values, axis=0)
    tail_count = max(1, int(round(sorted_values.shape[0] * float(alpha))))
    upper = np.nanmean(sorted_values[-tail_count:, ...], axis=0)
    if str(bootside).lower() == "upper":
        return np.squeeze(upper)
    lower = np.nanmean(sorted_values[:tail_count, ...], axis=0)
    return np.stack([lower, upper], axis=-1).squeeze()


def exact_p_values(observed: Any, surrogates: Any, *, center: Any = None) -> np.ndarray:
    """Return two-sided empirical p-values for observed values."""
    observed_values = np.asarray(observed)
    surrogate_values = np.asarray(surrogates)
    if center is None:
        center_values = np.nanmean(surrogate_values, axis=0)
    else:
        center_values = np.asarray(center)
    distance = np.abs(observed_values - center_values)
    surrogate_distance = np.abs(surrogate_values - center_values)
    return np.nanmean(surrogate_distance >= np.expand_dims(distance, axis=0), axis=0)


def _args(args: Any) -> list[np.ndarray]:
    if isinstance(args, (list, tuple)):
        return [np.asarray(arg) for arg in args]
    return [np.asarray(args)]


def _select_basevect(array: np.ndarray, basevect: Any) -> np.ndarray:
    indices = np.asarray([] if basevect is None else basevect, dtype=int).ravel()
    if indices.size == 0:
        return array
    if np.any(indices < 0):
        raise ValueError("basevect indices must be zero-based and non-negative")
    return np.take(array, indices, axis=1)


def _shuffle_dims(shuffledim: Any, ndim: int, *, boottype: str) -> list[int]:
    if str(boottype).lower() == "rand":
        return []
    values = np.asarray([] if shuffledim is None else shuffledim, dtype=int).ravel()
    if values.size == 0:
        values = np.asarray([2])
    dims = []
    for value in values:
        axis = int(value) - 1
        if axis < 0 or axis >= ndim:
            raise ValueError("shuffledim contains an invalid axis")
        dims.append(axis)
    return dims


def _resample_array(array: np.ndarray, rng: np.random.Generator, *, boottype: str, shuffledim: list[int]) -> np.ndarray:
    values = np.asarray(array).copy()
    mode = str(boottype).lower()
    if mode == "rand":
        if np.iscomplexobj(values):
            phases = rng.uniform(0.0, 2.0 * np.pi, size=values.shape)
            return values * np.exp(1j * phases)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=values.shape)
        return values * signs
    if mode != "shuffle":
        raise ValueError("boottype must be 'shuffle' or 'rand'")
    for axis in shuffledim:
        values = np.take(values, rng.permutation(values.shape[axis]), axis=axis)
    return values


__all__ = ["BootstrapResult", "bootstat", "bootstrap_threshold", "exact_p_values"]
