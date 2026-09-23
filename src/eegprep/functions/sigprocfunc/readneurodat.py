"""Read Neuroscan two-dimensional channel-location files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from eegprep.functions.sigprocfunc.convertlocs import convertlocs
from eegprep.functions.sigprocfunc.loadtxt import loadtxt


NEUROSCAN_RADIUS = 513.1617
NEUROSCAN_EDGE_DEGREES = 44.0


def readneurodat(filename: str | Path) -> tuple[list[dict[str, Any]], list[str], np.ndarray, np.ndarray]:
    """Read a Neuroscan ``.dat`` electrode file.

    Returns ``(chanlocs, labels, theta, phi)``. The angular arrays are degrees;
    channel locations also include EEGPrep's Cartesian, spherical, and
    topographic coordinate fields.
    """
    table = loadtxt(filename, verbose="off")
    if table.ndim != 2 or table.shape[1] < 4:
        raise ValueError("Neuroscan .dat rows must contain index, label, x, and y")
    try:
        order = np.argsort(np.asarray(table[:, 0], dtype=float), kind="stable")
        x = np.asarray(table[order, -2], dtype=float)
        y = np.asarray(table[order, -1], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Neuroscan .dat index and coordinates must be numeric") from exc
    labels = [str(value) for value in table[order, -3].tolist()]
    theta = np.hypot(x, y) / NEUROSCAN_RADIUS * NEUROSCAN_EDGE_DEGREES
    phi = np.degrees(np.arctan2(y, x))
    locs = [
        {"labels": label, "sph_theta_besa": float(theta_value), "sph_phi_besa": float(phi_value)}
        for label, theta_value, phi_value in zip(labels, theta, phi)
    ]
    return convertlocs(locs, "sphbesa2all"), labels, theta, phi


__all__ = ["readneurodat"]
