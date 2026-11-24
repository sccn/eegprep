"""Nonlinear equivalent-dipole fitting for one ICA component."""

import numpy as np

from eegprep.plugins.dipfit.dipfit_gridsearch import fitting_config, store_source
from eegprep.plugins.dipfit.eeglab2fieldtrip import eeglab2fieldtrip
from eegprep.plugins.dipfit.ft_dipolefitting import ft_dipolefitting
from eegprep.plugins.dipfit._utils import normalize_model_list
from eegprep.plugins.dipfit.eeg_compatlas import eeg_compatlas


def dipfit_nonlinear(EEG: dict, *, component: int, nonlinear="yes", symmetry=None, maxiter=None) -> dict:
    """Refine positions and moments, or fit only moments when nonlinear='no'."""
    cfg = fitting_config(EEG, [component])
    models = normalize_model_list(EEG)
    if component > len(models) or not np.size(models[component - 1].get("posxyz", [])):
        raise ValueError("An initial dipole position is required, run dipfit_gridsearch first")
    model = models[component - 1]
    positions = np.asarray(model["posxyz"], dtype=float).reshape(-1, 3)
    selected = np.asarray(model.get("select", model.get("active", np.arange(1, len(positions) + 1))), dtype=int) - 1
    if selected.size:
        positions = positions[selected]
    cfg.update(
        gridsearch="no",
        nonlinear="no" if str(nonlinear).lower() in ("no", "off", "false", "0") else "yes",
        symmetry=symmetry,
        dip={"pos": positions},
    )
    if maxiter is not None:
        cfg["dipfit"] = dict(cfg.get("dipfit", {}), maxiter=maxiter)
    fitted = store_source(EEG, cfg, ft_dipolefitting(cfg, eeglab2fieldtrip(EEG)))
    dipfit = fitted.get("dipfit") or {}
    coordformat = str(dipfit.get("coordformat", "")).lower()
    transform = np.asarray(dipfit.get("coord_transform", []), dtype=float).size
    if coordformat in {"mni", "spherical"} or transform:
        return eeg_compatlas(fitted, components=[component])
    return fitted
