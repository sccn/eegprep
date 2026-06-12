"""Shared bootstrap helpers for EEGLAB-style time-frequency functions."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.miscfunc.value_parsing import parse_numeric_sequence
from eegprep.functions.timefreqfunc.newtimeftrialbaseln import baseline_indices


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


def thresholds_by_frequency(values: np.ndarray, *, alpha: float, bootside: str) -> np.ndarray:
    """Pool surrogate baseline/time samples per frequency before thresholding."""
    nfreq = values.shape[1]
    pooled = values.transpose(0, 2, 1).reshape(-1, nfreq)
    thresholds = np.asarray(bootstrap_threshold(pooled, alpha=alpha, bootside=bootside))
    if str(bootside).lower() == "both":
        return thresholds.reshape(nfreq, 2)
    return thresholds.reshape(nfreq)


def threshold_vector(thresholds: Any, target_shape: tuple[int, ...]) -> np.ndarray:
    """Broadcast scalar or per-frequency thresholds toward a time-frequency result."""
    values = np.asarray(thresholds, dtype=float).squeeze()
    if values.ndim == 0:
        return np.full(target_shape, float(values))
    if values.ndim == 1:
        return values[:, np.newaxis]
    return values


def bootstrap_indices(
    times: np.ndarray,
    *,
    baseline: Any = None,
    baseboot: Any = 1,
    baseln: np.ndarray | None = None,
    limit_to_baseboot: bool = False,
) -> np.ndarray:
    """Return zero-based bootstrap time indices for newtimef/newcrossf paths."""
    values = np.asarray(parse_numeric_sequence(baseboot, dtype=float), dtype=float)
    if values.size == 0:
        if baseln is not None:
            return np.asarray(baseln, dtype=int)
        return np.nonzero(np.asarray(times) <= 0)[0]
    if values.size == 1:
        if values[0] == 0:
            return np.asarray([], dtype=int)
        if not limit_to_baseboot:
            baseline_values = np.asarray(parse_numeric_sequence(baseline, dtype=float), dtype=float)
            if baseline_values.size and not np.isnan(baseline_values[0]):
                return np.asarray([] if baseln is None else baseln, dtype=int)
        indices = np.nonzero(np.asarray(times) <= values[0])[0]
        return indices if indices.size else np.arange(np.asarray(times).size, dtype=int)
    return baseline_indices(times, values)


def resample_trials(
    values: np.ndarray,
    generator: np.random.Generator,
    boottype: str,
    *,
    complex_phase: bool = False,
) -> np.ndarray:
    """Resample trial-axis time-frequency arrays for bootstrap/permutation."""
    mode = str(boottype).lower()
    sample = np.asarray(values).copy()
    if mode in {"shuffle", "shufftrials"}:
        trial_indices = generator.integers(0, sample.shape[2], size=sample.shape[2])
        return sample[:, :, trial_indices]
    if mode in {"rand", "randall"}:
        if complex_phase or np.iscomplexobj(sample):
            return sample * np.exp(1j * generator.uniform(0.0, 2.0 * np.pi, size=sample.shape))
        signs = generator.choice(np.asarray([-1.0, 1.0]), size=sample.shape)
        return sample * signs
    raise ValueError("boottype must be 'shuffle', 'shufftrials', 'rand', or 'randall'")


def resample_pair(
    first: np.ndarray, second: np.ndarray, generator: np.random.Generator, *, boottype: str
) -> tuple[np.ndarray, np.ndarray]:
    """Resample paired time-frequency arrays for cross-frequency bootstrap."""
    mode = str(boottype).lower()
    if mode in {"shuffle", "shufftrials"}:
        indices = generator.permutation(second.shape[2])
        return first, second[:, :, indices]
    if mode in {"rand", "randall"}:
        phases = np.exp(1j * generator.uniform(0.0, 2.0 * np.pi, size=first.shape))
        return first * phases, second
    raise ValueError("boottype must be 'shuffle', 'shufftrials', 'rand', or 'randall'")


def resample_array(array: np.ndarray, rng: np.random.Generator, *, boottype: str, shuffledim: list[int]) -> np.ndarray:
    """Resample a generic bootstrap array for bootstat."""
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
