"""Convert Cartesian channel coordinates to EEGLAB topoplot coordinates."""

from __future__ import annotations

from typing import Any

import numpy as np


def cart2topo(x: Any, *args: Any):
    """Convert XYZ Cartesian coordinates to polar ``topoplot`` coordinates.

    This follows current EEGLAB ``cart2topo.m`` behavior: Cartesian
    coordinates are converted to MATLAB-style spherical coordinates, then to
    ``topoplot`` ``theta`` and ``radius`` via ``sph2topo`` method 2.

    Args:
        x: Either an ``(n, 3)`` array of ``X, Y, Z`` coordinates or a vector of
            X coordinates when Y and Z are supplied as separate arguments.
        *args: Either ``(y, z)`` vectors. Additional EEGLAB legacy options are
            deprecated and are not supported.

    Returns:
        tuple: ``(theta, radius, x, y, z)`` arrays.
    """
    x_arr, y_arr, z_arr = _split_xyz(x, args)
    hypot_xy = np.hypot(x_arr, y_arr)
    sph_theta = np.degrees(np.arctan2(y_arr, x_arr))
    sph_phi = np.degrees(np.arctan2(z_arr, hypot_xy))

    theta = -sph_theta
    radius = 0.5 - sph_phi / 180.0
    theta = _wrap_degrees(theta)
    return theta, radius, x_arr.copy(), y_arr.copy(), z_arr.copy()


def _split_xyz(x: Any, args: tuple[Any, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(args) == 0:
        xyz = np.asarray(x, dtype=float)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("First argument must be an (n, 3) XYZ array")
        return xyz[:, 0].copy(), xyz[:, 1].copy(), xyz[:, 2].copy()

    if len(args) == 2 and not isinstance(args[0], (str, bytes)):
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        y_arr = np.asarray(args[0], dtype=float).reshape(-1)
        z_arr = np.asarray(args[1], dtype=float).reshape(-1)
        if x_arr.shape != y_arr.shape or x_arr.shape != z_arr.shape:
            raise ValueError("x, y, and z must have the same shape")
        return x_arr.copy(), y_arr.copy(), z_arr.copy()

    raise ValueError("Additional cart2topo parameters are no longer supported")


def _wrap_degrees(theta: np.ndarray) -> np.ndarray:
    wrapped = (theta + 180.0) % 360.0 - 180.0
    wrapped[np.isclose(wrapped, -180.0) & (theta > 0)] = 180.0
    return wrapped
