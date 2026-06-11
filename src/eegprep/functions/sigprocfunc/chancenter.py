"""Recenter EEGLAB Cartesian channel coordinates."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize


def chancenter(
    x: Any, y: Any, z: Any, center: Any = (0.0, 0.0, 0.0)
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Return XYZ coordinates centered around a known or fitted sphere center."""
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    z_arr = np.asarray(z, dtype=float).reshape(-1)
    if x_arr.shape != y_arr.shape or x_arr.shape != z_arr.shape:
        raise ValueError("x, y, and z must have the same shape")
    optimize = _is_auto_center(center)
    initial = np.zeros(3, dtype=float) if optimize else np.asarray(center, dtype=float).reshape(3)
    newcenter = _fit_center(x_arr, y_arr, z_arr, initial) if optimize and x_arr.size >= 4 else initial
    return x_arr - newcenter[0], y_arr - newcenter[1], z_arr - newcenter[2], newcenter, optimize


def _fit_center(x: np.ndarray, y: np.ndarray, z: np.ndarray, initial: np.ndarray) -> np.ndarray:
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if finite.sum() < 4:
        return initial

    def objective(center: np.ndarray) -> float:
        radii = np.sqrt((x[finite] - center[0]) ** 2 + (y[finite] - center[1]) ** 2 + (z[finite] - center[2]) ** 2)
        return float(np.std(radii))

    result = minimize(objective, initial, method="Nelder-Mead", options={"maxfev": 1000})
    return np.asarray(result.x if result.success else initial, dtype=float)


def _is_auto_center(center: Any) -> bool:
    if center is None:
        return True
    if isinstance(center, str):
        return center.strip().lower() in {"", "auto", "[]"}
    if isinstance(center, np.ndarray):
        return center.size == 0
    return isinstance(center, (list, tuple)) and len(center) == 0


__all__ = ["chancenter"]
