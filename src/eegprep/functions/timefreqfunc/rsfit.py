"""Fit Ramberg-Schmeiser distributions to empirical values."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import optimize

from eegprep.functions.timefreqfunc.rsadjust import rsadjust
from eegprep.functions.timefreqfunc.rsget import rsget
from eegprep.functions.timefreqfunc.rspdfsolv import rspdfsolv


def rsfit(
    x: Any, value: float, plot: int = 0, *, return_details: bool = False
) -> float | tuple[float, np.ndarray, np.ndarray, float]:
    """Return a Ramberg-Schmeiser fitted p-value for ``value`` within ``x``."""
    _ = plot
    data = np.asarray(x, dtype=float).ravel()
    data = data[np.isfinite(data)]
    if data.size == 0:
        raise ValueError("x must contain finite values")

    mean = float(np.mean(data))
    variance = float(np.sum((data - mean) ** 2) / data.size)
    if variance <= 0:
        raise ValueError("x must have positive variance")
    third = float(np.sum((data - mean) ** 3) / data.size)
    fourth = float(np.sum((data - mean) ** 4) / data.size)
    skewness = third / variance**1.5
    kurtosis = fourth / variance**2
    cumulants = np.asarray([mean, variance, skewness, kurtosis], dtype=float)
    if kurtosis < 0:
        raise ValueError("cannot fit a distribution with negative kurtosis")

    sol = _fit_shape_parameters(abs(skewness), kurtosis)
    if np.isclose(sol[0] * sol[1], -1.0):
        raise RuntimeError("Ramberg-Schmeiser fit converged to an invalid sign")
    l1, l2, l3, l4 = rsadjust(float(sol[0]), float(sol[1]), mean, variance, skewness)
    lambdas = np.asarray([l1, l2, l3, l4], dtype=float)
    pvalue = rsget(lambdas, float(value))
    if return_details:
        return pvalue, cumulants, lambdas, _chi_square_fit(data, lambdas)
    return pvalue


def _fit_shape_parameters(skewness: float, kurtosis: float) -> np.ndarray:
    for initial in (np.asarray([0.1, 0.1]), np.asarray([-0.1, -0.1])):
        sol, _fopt, _iterations, _funcalls, warnflag = optimize.fmin(
            lambda current: rspdfsolv(current, skewness, kurtosis),
            initial,
            xtol=1.0e-12,
            maxfun=100000,
            disp=False,
            full_output=True,
        )
        if warnflag == 0 and np.all(np.isfinite(sol)):
            return np.asarray(sol, dtype=float)
    raise RuntimeError("Ramberg-Schmeiser shape fit did not converge")


def _chi_square_fit(data: np.ndarray, lambdas: np.ndarray) -> float:
    counts, edges = np.histogram(data, bins=25)
    counts = counts.astype(float)
    lower_edges = edges[:-1].copy()
    upper_edges = edges[1:].copy()

    index = 0
    while index < counts.size - 1:
        if counts[index] < 5:
            counts[index + 1] += counts[index]
            counts = np.delete(counts, index)
            lower_edges = np.delete(lower_edges, index + 1)
            upper_edges = np.delete(upper_edges, index)
            continue
        index += 1

    index = counts.size - 1
    while index >= 1:
        if counts[index] < 5:
            counts[index - 1] += counts[index]
            counts = np.delete(counts, index)
            lower_edges = np.delete(lower_edges, index)
            upper_edges = np.delete(upper_edges, index - 1)
        index -= 1

    expected = []
    for lower, upper in zip(lower_edges, upper_edges, strict=True):
        expected.append(data.size * (rsget(lambdas, float(upper)) - rsget(lambdas, float(lower))))
    expected_values = np.asarray(expected, dtype=float)
    if expected_values.size == 0 or np.any(expected_values <= 0):
        return float("nan")
    return float(np.sum(((expected_values - counts) ** 2) / expected_values))


__all__ = ["rsfit"]
