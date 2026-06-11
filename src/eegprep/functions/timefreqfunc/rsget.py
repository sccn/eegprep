"""Ramberg-Schmeiser cumulative probability lookup."""

from __future__ import annotations

from typing import Any

from scipy import optimize

from eegprep.functions.timefreqfunc.rspfunc import rspfunc


def rsget(lambdas: Any, value: float) -> float:
    """Return the fitted Ramberg-Schmeiser cumulative probability at ``value``."""
    result = optimize.minimize_scalar(
        lambda pvalue: rspfunc(pvalue, lambdas, value),
        bounds=(0.0, 1.0),
        method="bounded",
        options={"xatol": 1.0e-15, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError("Ramberg-Schmeiser probability lookup did not converge")
    return float(result.x)


__all__ = ["rsget"]
