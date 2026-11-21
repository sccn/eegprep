"""Leadfield and dipole positions for the packaged MNI BEM comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from eegprep import pop_loadset
from eegprep.plugins.dipfit.dipfit_gridsearch import fitting_config
from eegprep.plugins.dipfit.eeglab2fieldtrip import eeglab2fieldtrip
from eegprep.plugins.dipfit.ft_dipolefitting import compute_dipole_leadfield, prepare_headmodel_sens
from eegprep.plugins.dipfit.pop_dipfit_settings import pop_dipfit_settings
from eegprep.plugins.dipfit.pop_multifit import pop_multifit


PARITY_DIPOLES_MM = np.asarray([[0.0, -20.0, 40.0], [30.0, -50.0, 30.0], [-25.0, 10.0, 45.0]], dtype=float)
PARITY_COMPONENTS = (1, 2, 3)


def bem_parity_payload(dataset: str | Path) -> dict[str, object]:
    """Return leadfields and ``pop_multifit`` positions for the default MNI model."""
    eeg = pop_loadset(str(dataset))
    eeg = pop_dipfit_settings(eeg, model="standardBEM")
    comp = eeglab2fieldtrip(eeg)
    vol, sens = prepare_headmodel_sens(fitting_config(eeg), comp["elec"], eeg["chanlocs"])
    leadfield = compute_dipole_leadfield(PARITY_DIPOLES_MM, sens, vol)
    fitted = pop_multifit(eeg, list(PARITY_COMPONENTS), threshold=100)
    positions = []
    residual = []
    for component in PARITY_COMPONENTS:
        model = fitted["dipfit"]["model"][component - 1]
        raw = np.asarray(model.get("posxyz", []), dtype=float)
        if raw.size < 3:
            positions.append([float("nan"), float("nan"), float("nan")])
        else:
            positions.append(raw.reshape(-1, 3)[0].tolist())
        residual.append(float(model.get("rv", float("nan"))))
    return {
        "labels": sens["label"],
        "dipoles": PARITY_DIPOLES_MM.tolist(),
        "leadfield": np.asarray(leadfield, dtype=float).tolist(),
        "positions": positions,
        "rv": residual,
        "coord_transform": [float(value) for value in eeg["dipfit"]["coord_transform"]],
    }
