"""EEG leadfield dispatch for DIPFIT head models.

Port of the concentric-sphere branch of FieldTrip ``ft_compute_leadfield``
(fieldtrip-20250523), including the origin shift, the 1/2/3-shell padding
onto ``eeg_leadfield4``, and the average reference applied when the sensor
array has no montage. BEM types raise ``NotImplementedError`` so the
boundary-element port can fill the same dispatch.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.plugins.dipfit.forward.eeg_leadfield1 import eeg_leadfield1
from eegprep.plugins.dipfit.forward.eeg_leadfield4 import eeg_leadfield4, eeg_leadfield4_prepare
from eegprep.plugins.dipfit.forward.ft_headmodeltype import (
    BEM_HEADMODELS,
    SPHERICAL_HEADMODELS,
    ft_headmodeltype,
    normalize_headmodel,
)


def ft_compute_leadfield(
    dippos: np.ndarray,
    sens: dict[str, Any] | np.ndarray,
    headmodel: dict[str, Any],
) -> np.ndarray:
    """Return the EEG leadfield at one or more dipole positions.

    Args:
        dippos: Dipole positions with shape ``(3,)``, ``(n, 3)``, or a
            length-``3n`` vector.
        sens: Electrode positions or a sensor mapping with ``elecpos``.
            Optional ``tra`` is applied instead of an average reference.
        headmodel: Volume conductor. Concentric spheres use ``r`` and ``cond``.

    Returns:
        Array of shape ``(n_electrodes, 3 * n_dipoles)``.
    """
    model = normalize_headmodel(headmodel) if "r" in headmodel else dict(headmodel)
    kind = ft_headmodeltype(model)
    if kind in BEM_HEADMODELS:
        raise NotImplementedError(f"{kind} EEG leadfields are not implemented; add that branch in ft_compute_leadfield")
    if kind not in SPHERICAL_HEADMODELS:
        raise ValueError(f"unsupported volume conductor model for EEG: {kind}")
    positions = _dipole_rows(dippos)
    electrodes = _electrode_positions(sens)
    origin = np.asarray(model.get("o", np.zeros(3)), dtype=float).reshape(1, 3)
    if np.any(origin):
        electrodes = electrodes - origin
        positions = positions - origin
        model = dict(model)
        model["o"] = np.zeros(3)
    radii = np.asarray(model["r"], dtype=float).ravel()
    cond = np.asarray(model["cond"], dtype=float).ravel()
    order = np.argsort(radii, kind="stable")
    radii = radii[order]
    cond = cond[order]
    shell_count = int(radii.size)
    if shell_count == 1:
        forward = eeg_leadfield1
        volume = {"r": radii, "cond": cond}
    elif shell_count in (2, 3, 4):
        forward = eeg_leadfield4
        padded_radii = _pad_shells(radii)
        padded_cond = _pad_shells(cond)
        volume = {"r": padded_radii, "cond": padded_cond}
        volume["t"] = model["t"] if "t" in model else eeg_leadfield4_prepare(volume)
    else:
        raise ValueError("more than 4 concentric spheres are not supported")
    columns = [forward(positions[index], electrodes, volume) for index in range(positions.shape[0])]
    leadfield = np.column_stack(columns) if columns else np.zeros((electrodes.shape[0], 0))
    montage = sens.get("tra") if isinstance(sens, dict) else None
    if montage is not None:
        return np.asarray(montage, dtype=float) @ leadfield
    return leadfield - np.mean(leadfield, axis=0, keepdims=True)


def _pad_shells(values: np.ndarray) -> np.ndarray:
    """Repeat the outer shell until four spheres are present, as FieldTrip does."""
    padded = np.array([values[min(index, values.size - 1)] for index in range(4)], dtype=float)
    return padded


def _dipole_rows(dippos: np.ndarray) -> np.ndarray:
    array = np.asarray(dippos, dtype=float)
    if array.ndim == 1:
        if array.size % 3 != 0 or array.size == 0:
            raise ValueError("dipole positions must have 3 columns")
        return array.reshape(-1, 3)
    if array.ndim == 2 and array.shape[1] == 3:
        return array.copy()
    if array.ndim == 2 and array.shape[0] == 1 and array.shape[1] % 3 == 0:
        return array.reshape(-1, 3)
    raise ValueError("dipole positions must have 3 columns")


def _electrode_positions(sens: dict[str, Any] | np.ndarray) -> np.ndarray:
    if isinstance(sens, dict):
        positions = sens["elecpos"] if "elecpos" in sens else sens["chanpos"]
    else:
        positions = sens
    array = np.asarray(positions, dtype=float)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError("electrode positions must be an Nx3 array")
    return array[:, :3].copy()
