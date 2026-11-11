"""FieldTrip ``ft_compute_leadfield`` dispatch for EEG.

The BEM branch calls ``eeg_leadfieldb``. Concentric spheres call
``eeg_leadfield_spheres``. The post-processing matches FieldTrip 20250523:
a sensor montage when ``sens['tra']`` is present, otherwise an average
reference over electrodes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.plugins.dipfit.fieldtrip.bem import BEM_TYPES, eeg_infinite_dipole, eeg_leadfieldb
from eegprep.plugins.dipfit.fieldtrip.spheres import eeg_leadfield_spheres


_SPHERE_TYPES = frozenset({"singlesphere", "concentricspheres"})
_INFINITE_TYPES = frozenset({"infinite", "infinite_currentdipole"})


def ft_compute_leadfield(dippos: np.ndarray, sens: Any, headmodel: dict[str, Any]) -> np.ndarray:
    """Return an EEG leadfield with one row per electrode and three columns per dipole."""
    electrodes = _electrode_positions(sens)
    kind = ft_headmodeltype(headmodel)
    if kind in BEM_TYPES:
        leadfield = eeg_leadfieldb(dippos, electrodes, headmodel)
    elif kind in _SPHERE_TYPES:
        leadfield = eeg_leadfield_spheres(dippos, electrodes, headmodel)
    elif kind in _INFINITE_TYPES:
        leadfield = eeg_infinite_dipole(dippos, electrodes, headmodel)
    else:
        raise ValueError(f"unsupported volume conductor model for EEG ({kind})")
    montage = sens.get("tra") if isinstance(sens, dict) else None
    if montage is not None:
        return np.asarray(montage, dtype=float) @ leadfield
    return leadfield - np.mean(leadfield, axis=0, keepdims=True)


def ft_headmodeltype(headmodel: dict[str, Any], desired: str | None = None) -> str | bool:
    """Return the FieldTrip head-model type, or whether it matches ``desired``."""
    declared = headmodel.get("type")
    if isinstance(declared, str) and declared:
        kind = "singleshell" if declared.lower() == "nolte" else declared.lower()
    elif "r" in headmodel and np.size(headmodel["r"]) == 1 and "label" not in headmodel:
        kind = "singlesphere"
    elif "r" in headmodel and np.size(headmodel["r"]) >= 2 and "label" not in headmodel:
        kind = "concentricspheres"
    elif "bnd" in headmodel and "mat" in headmodel:
        kind = "bem"
    elif not headmodel or list(headmodel) == ["unit"]:
        kind = "infinite"
    else:
        kind = "unknown"
    if desired is None:
        return kind
    if desired == "bem":
        return kind in BEM_TYPES or kind == "bem"
    return kind == desired


def _electrode_positions(sens: Any) -> np.ndarray:
    if isinstance(sens, dict):
        positions = np.asarray(sens.get("elecpos", sens.get("chanpos")), dtype=float)
    else:
        positions = np.asarray(sens, dtype=float)
    if positions.ndim == 1:
        positions = positions.reshape(1, 3)
    return positions
