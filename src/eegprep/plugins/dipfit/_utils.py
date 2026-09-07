"""Shared helpers for EEGPrep's bundled DIPFIT surfaces."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.popfunc.plot_utils import numeric_vector, python_literal


FIELDTRIP_LIMITATION = (
    "This DIPFIT workflow requires MRI segmentation, BEM head-model, LORETA, "
    "or FieldTrip source-analysis routines and assets that are not bundled in "
    "standalone EEGPrep yet. EEGPrep can run deterministic spherical grid and "
    "nonlinear dipole fitting, store DIPFIT settings, compute explicit-point "
    "spherical leadfields, and plot existing dipole models."
)


class DIPFITUnavailableError(RuntimeError):
    """Raised when a DIPFIT workflow needs an unavailable FieldTrip backend."""


@dataclass(frozen=True)
class DipfitTemplate:
    """Packaged metadata for EEGLAB's standard DIPFIT templates."""

    name: str
    shortname: str
    hdmfile: str
    mrifile: str
    chanfile: str
    coordformat: str
    coord_transform: tuple[float, ...]


STANDARD_TEMPLATES: tuple[DipfitTemplate, ...] = (
    DipfitTemplate(
        name="Template Spherical Four-Shell (BESA)",
        shortname="standardBESA",
        hdmfile="standard_BESA/standard_BESA.mat",
        mrifile="standard_BESA/avg152t1.mat",
        chanfile="standard_BESA/standard-10-5-cap385.elp",
        coordformat="Spherical",
        coord_transform=(),
    ),
    DipfitTemplate(
        name="Template Boundary Element Model (MNI)",
        shortname="standardBEM",
        hdmfile="standard_BEM/standard_vol.mat",
        mrifile="standard_BEM/standard_mri.mat",
        chanfile="standard_BEM/elec/standard_1005.elc",
        coordformat="MNI",
        coord_transform=(0.0, 0.0, 0.0, 0.0, 0.0, -1.5708, 1.0, 1.0, 1.0),
    ),
    DipfitTemplate(
        name="Custom files/structures (Fieldtrip format)",
        shortname="",
        hdmfile="",
        mrifile="",
        chanfile="",
        coordformat="",
        coord_transform=(),
    ),
)


def copy_eeg(EEG: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of an EEG dictionary before mutation."""
    return copy.deepcopy(EEG)


def dipfit_history(function_name: str, *args: Any, assignment: bool = True, **kwargs: Any) -> str:
    """Build a valid Python command for GUI/console history."""
    pieces = ["EEG", *(python_literal(arg) for arg in args)]
    pieces.extend(f"{key}={python_literal(value)}" for key, value in kwargs.items() if not _empty_history_value(value))
    command = f"{function_name}({', '.join(pieces)})"
    return f"EEG = {command}" if assignment else command


def template_by_name(name: Any) -> DipfitTemplate:
    """Return a standard template by EEGLAB short name or display name."""
    requested = str(name or "standardBEM").strip().lower()
    for template in STANDARD_TEMPLATES:
        if requested in {template.shortname.lower(), template.name.lower()}:
            return template
    valid = ", ".join(template.shortname for template in STANDARD_TEMPLATES if template.shortname)
    raise ValueError(f"model must be one of {valid}")


def apply_template(dipfit: dict[str, Any], template: DipfitTemplate) -> None:
    """Apply standard-template file and coordinate defaults in-place."""
    if not template.shortname:
        return
    dipfit["hdmfile"] = template.hdmfile
    dipfit["mrifile"] = template.mrifile
    dipfit["chanfile"] = template.chanfile
    dipfit["coordformat"] = template.coordformat
    dipfit["coord_transform"] = list(template.coord_transform)


def component_count(EEG: dict[str, Any]) -> int:
    """Return the number of ICA components present in ``EEG``."""
    weights = np.asarray(EEG.get("icaweights", []))
    if weights.ndim == 2 and weights.size:
        return int(weights.shape[0])
    winv = np.asarray(EEG.get("icawinv", []))
    if winv.ndim == 2 and winv.size:
        return int(winv.shape[1])
    return 0


def ensure_ica(EEG: dict[str, Any]) -> None:
    """Raise when ``EEG`` has no ICA maps to localize."""
    if component_count(EEG) <= 0:
        raise ValueError("No ICA components to fit")


def ensure_channel_locations(EEG: dict[str, Any]) -> None:
    """Raise when no source-localization channel coordinates are available."""
    if not coordinate_channel_indices(EEG):
        raise ValueError("No channel locations with 3-D or polar coordinates are available")


def ensure_dipfit_settings(EEG: dict[str, Any]) -> dict[str, Any]:
    """Return existing DIPFIT settings or raise an EEGLAB-like error."""
    dipfit = EEG.get("dipfit")
    if not isinstance(dipfit, dict) or not dipfit:
        raise ValueError("General dipolefit settings not specified")
    if not dipfit.get("hdmfile") and not dipfit.get("vol"):
        raise ValueError("Dipolefit volume conductor model not specified")
    return dipfit


def coordinate_channel_indices(EEG: dict[str, Any]) -> list[int]:
    """Return EEGLAB 1-based channel indices with usable localization coordinates."""
    indices = []
    for index, chanloc in enumerate(chanlocs_as_list(EEG.get("chanlocs", [])), start=1):
        if _has_coordinates(chanloc):
            indices.append(index)
    return indices


def one_based_indices(value: Any, *, limit: int, default_all: bool = False) -> list[int]:
    """Normalize EEGLAB-facing 1-based indices with bounds checks."""
    if _empty_sequence(value):
        if default_all:
            return list(range(1, limit + 1))
        return []
    values = numeric_vector(value, dtype=float)
    if values.size == 0:
        return list(range(1, limit + 1)) if default_all else []
    if np.any(values != values.astype(int)):
        raise ValueError("indices must be integers")
    indices = values.astype(int).tolist()
    if any(index < 1 or index > limit for index in indices):
        raise ValueError(f"indices must be 1-based and within 1..{limit}")
    return indices


def normalize_model_list(EEG: dict[str, Any]) -> list[dict[str, Any]]:
    """Return DIPFIT model entries as a list of dictionaries."""
    dipfit = EEG.get("dipfit") or {}
    models = dipfit.get("model", []) if isinstance(dipfit, dict) else []
    if isinstance(models, dict):
        return [models]
    if isinstance(models, np.ndarray):
        models = models.tolist()
    if isinstance(models, Iterable) and not isinstance(models, (str, bytes)):
        return [dict(model) for model in models if isinstance(model, dict)]
    return []


def localized_components(EEG: dict[str, Any]) -> list[int]:
    """Return 1-based component indices that have a non-empty dipole position."""
    components = []
    for index, model in enumerate(normalize_model_list(EEG), start=1):
        if np.asarray(model.get("posxyz", [])).size:
            components.append(index)
    return components


def unavailable(action: str) -> None:
    """Raise the shared explicit missing-backend error."""
    raise DIPFITUnavailableError(f"{action}: {FIELDTRIP_LIMITATION}")


def _has_coordinates(chanloc: dict[str, Any]) -> bool:
    xyz = [chanloc.get(key) for key in ("X", "Y", "Z")]
    if all(_finite_scalar(value) for value in xyz):
        return True
    return _finite_scalar(chanloc.get("theta")) and _finite_scalar(chanloc.get("radius"))


def _finite_scalar(value: Any) -> bool:
    try:
        numeric = float(np.asarray(value).ravel()[0])
    except (IndexError, TypeError, ValueError):
        return False
    return bool(np.isfinite(numeric))


def _empty_sequence(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().strip("[]") == ""
    if isinstance(value, np.ndarray):
        return value.size == 0
    return isinstance(value, (list, tuple)) and len(value) == 0


def _empty_history_value(value: Any) -> bool:
    return _empty_sequence(value) or (isinstance(value, str) and value == "")
