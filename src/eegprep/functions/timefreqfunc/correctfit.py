"""Correct biased p-values using EEGLAB's gamma-fit convention."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def correctfit(
    pvalue: float,
    *,
    allpval: Any = None,
    gamparams: Any = None,
    zeromode: str = "on",
) -> tuple[float, float, float, float]:
    """Return a gamma-fit corrected p-value and fitted parameters."""
    params = np.asarray([] if gamparams is None else gamparams, dtype=float).ravel()
    sampled = np.asarray([] if allpval is None else allpval, dtype=float).ravel()
    if params.size:
        if params.size < 2:
            raise ValueError("gamparams must contain at least [shape, scale]")
        shape = float(params[0])
        scale = float(params[1])
        zero_frequency = float(params[2]) if params.size > 2 else 0.0
    elif sampled.size:
        nonzero = sampled[sampled != 0]
        zero_frequency = float((sampled.size - nonzero.size) / sampled.size)
        transformed = -np.log10(nonzero) + 1.0e-10
        shape, _loc, scale = stats.gamma.fit(transformed, floc=0.0)
    else:
        raise ValueError("correctfit requires allpval or gamparams")

    corrected = float(pvalue)
    if corrected == 0:
        if zeromode.lower() == "on":
            corrected = zero_frequency
    else:
        transformed_p = -np.log10(corrected) + 1.0e-10
        corrected = 1.0 - float(stats.gamma.cdf(transformed_p, shape, scale=scale))
    return corrected, float(shape), float(scale), float(zero_frequency)


__all__ = ["correctfit"]
