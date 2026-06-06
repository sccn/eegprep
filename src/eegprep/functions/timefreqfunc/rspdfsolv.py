"""Ramberg-Schmeiser moment-fit objective used by ``rsfit``."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import special


_INVALID_OBJECTIVE = 1.0e300


def rspdfsolv(lambdas: Any, skewness: float, kurtosis: float) -> float:
    """Return the Ramberg-Schmeiser moment residual for ``lambda3``/``lambda4``."""
    values = np.asarray(lambdas, dtype=float).ravel()
    if values.size != 2:
        raise ValueError("lambdas must contain [lambda3, lambda4]")
    l3, l4 = float(values[0]), float(values[1])
    try:
        estim_skew, estim_kurt = _standardized_moments(l3, l4)
    except FloatingPointError:
        return _INVALID_OBJECTIVE
    if not np.isfinite(estim_skew) or not np.isfinite(estim_kurt):
        return _INVALID_OBJECTIVE
    residual = (estim_skew - float(skewness)) ** 2 + (estim_kurt - float(kurtosis)) ** 2
    if l3 * l4 < 0:
        residual *= 2.0
    return float(residual) if np.isfinite(residual) else _INVALID_OBJECTIVE


def _standardized_moments(l3: float, l4: float) -> tuple[float, float]:
    with np.errstate(all="raise"):
        a = 1.0 / (1.0 + l3) - 1.0 / (1.0 + l4)
        b = 1.0 / (1.0 + 2.0 * l3) + 1.0 / (1.0 + 2.0 * l4) - 2.0 * special.beta(1.0 + l3, 1.0 + l4)
        c = (
            1.0 / (1.0 + 3.0 * l3)
            - 1.0 / (1.0 + 3.0 * l4)
            - 3.0 * special.beta(1.0 + 2.0 * l3, 1.0 + l4)
            + 3.0 * special.beta(1.0 + l3, 1.0 + 2.0 * l4)
        )
        d = (
            1.0 / (1.0 + 4.0 * l3)
            + 1.0 / (1.0 + 4.0 * l4)
            - 4.0 * special.beta(1.0 + 3.0 * l3, 1.0 + l4)
            - 4.0 * special.beta(1.0 + l3, 1.0 + 3.0 * l4)
            + 6.0 * special.beta(1.0 + 2.0 * l3, 1.0 + 2.0 * l4)
        )
        variance_term = b - a**2
        if variance_term <= 0:
            raise FloatingPointError
        estim_skew = (c - 3.0 * a * b + 2.0 * a**3) / variance_term**1.5
        estim_kurt = (d - 4.0 * a * c + 6.0 * a**2 * b - 3.0 * a**4) / variance_term**2
    return float(estim_skew), float(estim_kurt)


__all__ = ["rspdfsolv"]
