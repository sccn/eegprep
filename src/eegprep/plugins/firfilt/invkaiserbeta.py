"""Inverse Kaiser-window beta helper for the bundled firfilt plugin."""

from __future__ import annotations

from scipy.optimize import brentq


_LINEAR_THRESHOLD = 0.1102 * (50 - 8.7)


def invkaiserbeta(beta: float) -> float:
    """Estimate maximum passband deviation from a Kaiser window beta."""
    beta = float(beta)
    if beta < 0:
        raise ValueError("Kaiser beta must be non-negative.")
    if beta > _LINEAR_THRESHOLD:
        devdb = beta / 0.1102 + 8.7
    elif beta == 0:
        devdb = 21.0
    else:
        devdb = brentq(
            lambda value: 0.5842 * (value - 21) ** 0.4 + 0.07886 * (value - 21) - beta,
            21,
            51,
        )
    return float(10 ** (-devdb / 20))
