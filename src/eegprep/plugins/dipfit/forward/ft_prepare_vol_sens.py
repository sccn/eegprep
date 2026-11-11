"""Electrode preparation for spherical EEG head models.

Port of the ``singlesphere`` / ``concentricspheres`` branch of FieldTrip
``ft_prepare_vol_sens`` (fieldtrip-20250523). Electrodes are projected
radially onto the outer sphere. Boundary-element models are left for the
BEM forward-model port.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.plugins.dipfit.forward.ft_headmodeltype import (
    BEM_HEADMODELS,
    SPHERICAL_HEADMODELS,
    ft_headmodeltype,
    normalize_headmodel,
)


def ft_prepare_vol_sens(
    headmodel: dict[str, Any], sens: dict[str, Any] | np.ndarray
) -> tuple[dict[str, Any], dict[str, Any] | np.ndarray]:
    """Project EEG electrodes onto the outer sphere of a concentric head model.

    Args:
        headmodel: Spherical volume conductor. ``r`` and ``cond`` (or ``c``)
            are required. ``o`` is the sphere center, defaulting to the origin.
        sens: Electrode array ``(n, 3)`` or a mapping with ``elecpos``.

    Returns:
        The normalized head model and the sensor description with ``elecpos``
        on the outer sphere.
    """
    model = normalize_headmodel(headmodel) if "r" in headmodel else dict(headmodel)
    kind = ft_headmodeltype(model)
    positions = _electrode_positions(sens)
    if kind in SPHERICAL_HEADMODELS:
        positions = _project_to_outer_sphere(positions, model)
    elif kind in BEM_HEADMODELS:
        raise NotImplementedError(
            f"{kind} electrode projection is not implemented; the FieldTrip BEM forward model owns that branch"
        )
    else:
        raise ValueError(f"unsupported volume conductor model for EEG sensors: {kind}")
    if isinstance(sens, dict):
        updated = dict(sens)
        updated["elecpos"] = positions
        return model, updated
    return model, positions


def _project_to_outer_sphere(positions: np.ndarray, headmodel: dict[str, Any]) -> np.ndarray:
    radius = float(np.max(np.asarray(headmodel["r"], dtype=float)))
    origin = np.asarray(headmodel.get("o", np.zeros(3)), dtype=float).reshape(1, 3)
    centered = positions - origin
    distance = np.linalg.norm(centered, axis=1)
    projected = centered * radius / distance[:, np.newaxis]
    return projected + origin


def _electrode_positions(sens: dict[str, Any] | np.ndarray) -> np.ndarray:
    if isinstance(sens, dict):
        if "elecpos" in sens:
            positions = sens["elecpos"]
        elif "chanpos" in sens:
            positions = sens["chanpos"]
        else:
            raise ValueError("EEG sensors need elecpos or chanpos")
    else:
        positions = sens
    array = np.asarray(positions, dtype=float)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError("electrode positions must be an Nx3 array")
    return array[:, :3].copy()
