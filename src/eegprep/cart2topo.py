"""Convert Cartesian channel coordinates to EEGLAB topoplot coordinates."""

from __future__ import annotations

import numpy as np


def _coerce_xyz(x, y=None, z=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize supported cart2topo input forms to flat x/y/z arrays."""
    if y is None and z is None:
        xyz = np.asarray(x, dtype=np.float64)
        if xyz.size == 0:
            return (
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
            )
        if xyz.ndim == 1 and xyz.size == 3:
            xyz = xyz.reshape(1, 3)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("cart2topo: x must be an N x 3 coordinate array")
        return xyz[:, 0], xyz[:, 1], xyz[:, 2]

    if y is None or z is None:
        raise ValueError("cart2topo: x, y, and z must all be provided")

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    z_arr = np.asarray(z, dtype=np.float64)
    if x_arr.shape != y_arr.shape or x_arr.shape != z_arr.shape:
        raise ValueError("cart2topo: x, y, and z must have matching shapes")
    return x_arr.reshape(-1), y_arr.reshape(-1), z_arr.reshape(-1)


def cart2topo(x, y=None, z=None) -> tuple[np.ndarray, np.ndarray]:
    """Convert Cartesian coordinates to EEGLAB polar topoplot coordinates.

    This follows current EEGLAB ``cart2topo`` behavior, which delegates to
    ``convertlocs(..., 'cart2all')``: Cartesian coordinates are first converted
    to MATLAB spherical coordinates, then to topoplot coordinates.

    Parameters
    ----------
    x : array-like
        Either an ``N x 3`` array of ``[X, Y, Z]`` coordinates, a single
        ``[X, Y, Z]`` coordinate, or the X coordinate array when ``y`` and
        ``z`` are also supplied.
    y, z : array-like, optional
        Y and Z coordinate arrays. Must have the same shape as ``x``.

    Returns
    -------
    theta, radius : tuple[np.ndarray, np.ndarray]
        EEGLAB topoplot angle in degrees and radius by topoplot convention.
    """
    x_arr, y_arr, z_arr = _coerce_xyz(x, y, z)
    hypot_xy = np.hypot(x_arr, y_arr)
    sph_theta = np.degrees(np.arctan2(y_arr, x_arr))
    sph_phi = np.degrees(np.arctan2(z_arr, hypot_xy))

    theta = -sph_theta
    radius = 0.5 - sph_phi / 180.0
    return theta.astype(np.float64, copy=False), radius.astype(np.float64, copy=False)
