"""Coarse single-dipole scanning for ICA components."""

import re

import numpy as np

from eegprep.functions.miscfunc.linspace import linspace
from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.popfunc.plot_utils import numeric_vector
from eegprep.plugins.dipfit.eeglab2fieldtrip import eeglab2fieldtrip
from eegprep.plugins.dipfit.ft_dipolefitting import ft_dipolefitting
from eegprep.plugins.dipfit._utils import (
    component_count,
    copy_eeg,
    ensure_dipfit_settings,
    normalize_model_list,
    one_based_indices,
)
from eegprep.plugins.dipfit.dipfit_reject import dipfit_reject


def default_search_grids(EEG: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return DIPFIT's 24 by 24 by 12 scan axes in mm."""
    cfg = ensure_dipfit_settings(EEG)
    radius = float(np.floor(max(cfg.get("vol", {}).get("r", [85.0]))))
    return linspace(-radius, radius, 24), linspace(-radius, radius, 24), linspace(0.0, radius, 12)


def parse_grid_values(value, *, default) -> np.ndarray:
    """Parse numeric grid coordinates or a dialog's linspace expression."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return np.asarray(default)
    if isinstance(value, str):
        match = re.fullmatch(r"linspace\(\s*([^,]+),\s*([^,]+),\s*([^)]+)\)", value.strip(), re.I)
        if match:
            return linspace(*map(float, match.groups()))
    values = numeric_vector(value, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("grid values must be finite")
    return values if values.size else np.asarray(default)


def fitting_config(EEG: dict, component=None) -> dict:
    """Build a FieldTrip configuration from the dataset's DIPFIT settings."""
    cfg = dict(ensure_dipfit_settings(EEG))
    channels = chanlocs_as_list(EEG["chanlocs"])
    selected = one_based_indices(cfg.get("chansel"), limit=len(channels), default_all=True)
    cfg["channel"] = [str(channels[index - 1]["labels"]) for index in selected]
    cfg["component"] = one_based_indices(component, limit=component_count(EEG), default_all=True)
    return cfg


def ensure_model_list(EEG: dict, count: int) -> list[dict]:
    """Ensure one model dictionary per ICA component, preserving existing fits."""
    models = normalize_model_list(EEG)
    models.extend(empty_model(index + 1) for index in range(len(models), count))
    EEG["dipfit"]["model"] = models
    return models


def empty_model(component: int) -> dict:
    """Return an empty EEGLAB dipole model."""
    return dict(posxyz=[], momxyz=[], rv=1.0, diffmap=[], sourcepot=[], datapot=[], component=component)


def store_source(EEG: dict, cfg: dict, source: dict) -> dict:
    """Store FieldTrip source results in EEG.dipfit.model without changing the input."""
    out = copy_eeg(EEG)
    models = ensure_model_list(out, component_count(out))
    for index, component in enumerate(cfg["component"]):
        dip = source["dip"][index]
        model = models[component - 1]
        model.update(
            posxyz=dip["pos"].tolist(),
            momxyz=dip["mom"].reshape(-1, 3).tolist(),
            rv=dip["rv"],
            diffmap=(source["Vmodel"][:, index] - source["Vdata"][:, index]).tolist(),
            sourcepot=source["Vmodel"][:, index].tolist(),
            datapot=source["Vdata"][:, index].tolist(),
            select=list(range(1, len(dip["pos"]) + 1)),
            active=list(range(1, len(dip["pos"]) + 1)),
        )
    return out


def dipfit_gridsearch(EEG: dict, *, component=None, xgrid=None, ygrid=None, zgrid=None, reject=None) -> dict:
    """Scan an inside-brain grid for the best single dipole per component."""
    cfg = fitting_config(EEG, component)
    for key, value, default in zip(("xgrid", "ygrid", "zgrid"), (xgrid, ygrid, zgrid), default_search_grids(EEG)):
        cfg[key] = parse_grid_values(value, default=default)
    cfg.update(gridsearch="yes", nonlinear="no")
    out = store_source(EEG, cfg, ft_dipolefitting(cfg, eeglab2fieldtrip(EEG)))
    if reject is not None:
        out["dipfit"]["model"] = dipfit_reject(out["dipfit"]["model"], float(reject) / 100)
    return out
