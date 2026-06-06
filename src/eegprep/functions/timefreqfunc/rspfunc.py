"""Ramberg-Schmeiser quantile residual used by ``rsget``."""

from __future__ import annotations

from typing import Any

import numpy as np


def rspfunc(pvalue: float, lambdas: Any, value: float) -> float:
    """Return the absolute quantile residual for one probability value."""
    l1, l2, l3, l4 = np.asarray(lambdas, dtype=float).ravel()
    quantile = l1 + (float(pvalue) ** l3 - (1.0 - float(pvalue)) ** l4) / l2
    return float(abs(float(value) - quantile))


__all__ = ["rspfunc"]
