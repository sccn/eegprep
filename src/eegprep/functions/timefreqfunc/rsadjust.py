"""Adjust Ramberg-Schmeiser shape parameters to sample moments."""

from __future__ import annotations

import numpy as np
from scipy import special


def rsadjust(
    lambda3: float, lambda4: float, mean: float, variance: float, skewness: float
) -> tuple[float, float, float, float]:
    """Return ``lambda1`` through ``lambda4`` adjusted to mean and variance."""
    l3 = float(lambda3)
    l4 = float(lambda4)
    if float(skewness) < 0:
        l3, l4 = l4, l3

    a = 1.0 / (1.0 + l3) - 1.0 / (1.0 + l4)
    b = 1.0 / (1.0 + 2.0 * l3) + 1.0 / (1.0 + 2.0 * l4) - 2.0 * special.beta(1.0 + l3, 1.0 + l4)
    c = (
        1.0 / (1.0 + 3.0 * l3)
        - 1.0 / (1.0 + 3.0 * l4)
        - 3.0 * special.beta(1.0 + 2.0 * l3, 1.0 + l4)
        + 3.0 * special.beta(1.0 + l3, 1.0 + 2.0 * l4)
    )
    if float(variance) <= 0:
        raise ValueError("variance must be positive")
    l2 = np.sqrt((b - a**2) / float(variance))
    skew = float(skewness) if float(skewness) != 0 else -1.0e-15
    if skew * (c - 2.0 * a * b + 2.0 * a**3) < 0:
        l2 = -l2
    l1 = float(mean) - a / l2
    return float(l1), float(l2), float(l3), float(l4)


__all__ = ["rsadjust"]
