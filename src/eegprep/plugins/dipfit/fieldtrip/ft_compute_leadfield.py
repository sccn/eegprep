"""FieldTrip ``ft_compute_leadfield`` dispatch for EEG.

The BEM branch calls ``eeg_leadfieldb``. Concentric spheres call
``eeg_leadfield_spheres``. The post-processing matches FieldTrip 20250523:
a sensor montage when ``sens['tra']`` is present, otherwise an average
reference over electrodes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.plugins.dipfit.fieldtrip.bem import BEM_TYPES, eeg_leadfieldb
from eegprep.plugins.dipfit.fieldtrip.spheres import eeg_leadfield_spheres


_SPHERE_TYPES = frozenset({"singlesphere", "concentricspheres"})


def ft_compute_leadfield(dippos: np.ndarray, sens: Any, headmodel: dict[str, Any]) -> np.ndarray:
    """Return an EEG leadfield with one row per electrode and three columns per dipole."""
    electrodes = _electrode_positions(sens)
    kind = ft_headmodeltype(headmodel)
    if kind in BEM_TYPES:
        leadfield = eeg_leadfieldb(dippos, electrodes, headmodel)
    elif kind in _SPHERE_TYPES:
        leadfield = eeg_leadfield_spheres(dippos, electrodes, headmodel)
    else:
        raise ValueError(f"unsupported volume conductor model for EEG ({kind})")
    montage = sens.get("tra") if isinstance(sens, dict) else None
    if montage is not None:
        return np.asarray(montage, dtype=float) @ leadfield
    return leadfield - np.mean(leadfield, axis=0, keepdims=True)


def ft_headmodeltype(headmodel: dict[str, Any]) -> str:
    """Return the FieldTrip head-model type."""
    declared = headmodel.get("type")
    if isinstance(declared, str) and declared:
        return "singleshell" if declared.lower() == "nolte" else declared.lower()
    if "r" in headmodel and np.size(headmodel["r"]) == 1 and "label" not in headmodel:
        return "singlesphere"
    if "r" in headmodel and np.size(headmodel["r"]) >= 2 and "label" not in headmodel:
        return "concentricspheres"
    if "bnd" in headmodel and "mat" in headmodel:
        return "bem"
    if not headmodel or list(headmodel) == ["unit"]:
        return "infinite"
    return "unknown"


def _electrode_positions(sens: Any) -> np.ndarray:
    if isinstance(sens, dict):
        positions = np.asarray(sens.get("elecpos", sens.get("chanpos")), dtype=float)
    else:
        positions = np.asarray(sens, dtype=float)
    if positions.ndim == 1:
        positions = positions.reshape(1, 3)
    return positions
