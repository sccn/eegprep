"""Inside/outside tests for spherical source compartments.

Port of the concentric-sphere branch of FieldTrip ``ft_inside_headmodel``
(fieldtrip-20250523). A dipole is inside when it is strictly inside the
innermost sphere.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.plugins.dipfit.forward.ft_headmodeltype import (
    BEM_HEADMODELS,
    SPHERICAL_HEADMODELS,
    ft_headmodeltype,
)


def ft_inside_headmodel(dippos: np.ndarray, headmodel: dict[str, Any]) -> np.ndarray:
    """Return whether each dipole lies in the source compartment.

    Args:
        dippos: Dipole positions, shape ``(n, 3)`` or ``(3,)``.
        headmodel: Volume conductor. Spherical models use the innermost radius.

    Returns:
        Boolean array of length ``n``. Points on the shell boundary are outside.
    """
    positions = np.asarray(dippos, dtype=float)
    if positions.ndim == 1:
        positions = positions.reshape(1, 3)
    positions = positions[:, :3]
    kind = ft_headmodeltype(headmodel)
    if kind not in SPHERICAL_HEADMODELS:
        if kind in BEM_HEADMODELS:
            raise NotImplementedError(
                f"{kind} inside-compartment tests are not implemented; the FieldTrip BEM forward model owns that branch"
            )
        raise ValueError(f"unsupported volume conductor model for inside detection: {kind}")
    radii = np.asarray(headmodel["r"], dtype=float).ravel()
    source = int(np.argmin(radii))
    origin = np.asarray(headmodel.get("o", np.zeros(3)), dtype=float).reshape(1, 3)
    distance = np.linalg.norm(positions - origin, axis=1) - float(radii[source])
    return distance < 0.0
