"""Shared helpers for EEGPrep's bundled DIPFIT surfaces."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import re

import numpy as np

from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.popfunc.plot_utils import numeric_vector, parse_plot_options_text, python_literal

# dipfitdefs.m four-shell BESA sphere: 85 mm head, 6 mm skin, 7 mm skull, 1 mm CSF.
BESA_SHELL_RADII_MM = (71.0, 72.0, 79.0, 85.0)
BESA_SHELL_CONDUCTIVITIES = (0.33, 1.0, 0.0042, 0.33)
_QUOTED_PLOT_GAP = re.compile(r"""(?<=['"])\s+(?=['"])""")


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


def dipfit_record(EEG: dict[str, Any]) -> dict[str, Any]:
    """Return ``EEG.dipfit`` when it is a mapping.

    An empty numeric placeholder is not a settings record. Using it in
    ``value or {}`` raises ``ValueError`` because the array has no truth value.
    """
    dipfit = EEG.get("dipfit") if isinstance(EEG, dict) else None
    if isinstance(dipfit, dict):
        return dipfit
    return {}


def besa_four_shell_volume() -> dict[str, Any]:
    """Return the standard BESA four-shell sphere stored by ``dipfitdefs.m``."""
    return {
        "r": list(BESA_SHELL_RADII_MM),
        "c": list(BESA_SHELL_CONDUCTIVITIES),
        "o": [0.0, 0.0, 0.0],
    }


def template_form_state(template: DipfitTemplate) -> dict[str, Any]:
    """Return dialog field values for one head-model template."""
    coord_index = {"spherical": 1, "mni": 2}.get(template.coordformat.lower(), 4)
    transform = " ".join(f"{float(value):g}" for value in template.coord_transform)
    return {
        "hdmfile": template.hdmfile,
        "mrifile": template.mrifile,
        "chanfile": template.chanfile,
        "coord_transform": transform,
        "coordformat": coord_index,
        "no_coreg": not bool(transform),
        "browse": not bool(template.shortname),
    }


def matlab_plot_options(text: Any) -> dict[str, Any]:
    """Parse plot options, including MATLAB ``'key' 'value'`` text."""
    normalized = _QUOTED_PLOT_GAP.sub(", ", str(text or "").strip())
    return parse_plot_options_text(normalized)


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
    if template.shortname == "standardBESA":
        dipfit["vol"] = besa_four_shell_volume()
    elif template.shortname == "standardBEM":
        dipfit.pop("vol", None)


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


def _shell_radii(vol: Any) -> np.ndarray:
    if not isinstance(vol, dict):
        return np.array([])
    radii = np.asarray(vol.get("r", []), dtype=float).ravel()
    if radii.size and np.all(np.isfinite(radii)) and np.all(radii > 0):
        return radii
    return np.array([])


def _template_kind(dipfit: dict[str, Any]) -> str:
    hdm = str(dipfit.get("hdmfile") or "").lower()
    model = str(dipfit.get("model") or "").lower()
    if "standard_besa" in hdm or model == "standardbesa":
        return "besa"
    if "standard_vol" in hdm or "standard_bem" in hdm or model == "standardbem":
        return "bem"
    return ""


def fill_missing_besa_volume(EEG: dict[str, Any]) -> None:
    """Store the bundled BESA sphere when a working copy names that template and has no volume."""
    dipfit = dipfit_record(EEG)
    if _template_kind(dipfit) == "besa" and _shell_radii(dipfit.get("vol")).size == 0:
        dipfit["vol"] = besa_four_shell_volume()


def ensure_dipfit_settings(EEG: dict[str, Any], *, caller: str = "dipfit") -> dict[str, Any]:
    """Return existing DIPFIT settings or raise an actionable error.

    This check does not write ``EEG.dipfit``. Fitting copies the dataset and
    calls :func:`fill_missing_besa_volume` on that copy.
    """
    dipfit = dipfit_record(EEG)
    if not dipfit:
        raise ValueError(
            f"{caller}: dipole fit settings are missing. "
            "Run pop_dipfit_settings and choose Template Spherical Four-Shell (BESA)."
        )
    kind = _template_kind(dipfit)
    if kind == "bem" and _shell_radii(dipfit.get("vol")).size == 0:
        raise ValueError(
            f"{caller}: the MNI boundary-element volume standard_vol.mat is not bundled. "
            "Run pop_dipfit_settings and choose Template Spherical Four-Shell (BESA). "
            "That template stores the four-shell sphere from dipfitdefs.m "
            "(radii 71, 72, 79, and 85 mm)."
        )
    if _shell_radii(dipfit.get("vol")).size == 0 and not dipfit.get("hdmfile"):
        raise ValueError(
            f"{caller}: the volume conductor model is not specified. "
            "Run pop_dipfit_settings and choose Template Spherical Four-Shell (BESA)."
        )
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
    dipfit = dipfit_record(EEG)
    models = dipfit.get("model", [])
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
