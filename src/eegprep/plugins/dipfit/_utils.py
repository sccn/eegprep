"""Shared helpers for EEGPrep's bundled DIPFIT surfaces."""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.popfunc.plot_utils import numeric_vector, python_literal


FIELDTRIP_LIMITATION = (
    "This DIPFIT workflow builds a new MRI head model or runs eLORETA. EEGPrep "
    "fits dipoles with the packaged MNI boundary-element model and the spherical "
    "workflow, but it does not create bemcp or dipoli system matrices and it does "
    "not run FieldTrip eLORETA."
)


class DIPFITUnavailableError(RuntimeError):
    """Raised when a DIPFIT workflow needs an unavailable FieldTrip backend."""


@dataclass(frozen=True)
class CoordKeywordTransform:
    """One ``dipfitdefs`` channel-file rule and its coregistration vector."""

    keywords: tuple[str, ...]
    transform: tuple[float, ...]


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
    coord_transforms: tuple[CoordKeywordTransform, ...] = ()


_BESA_TRANSFORMS: tuple[CoordKeywordTransform, ...] = (
    CoordKeywordTransform(("standard-10-5-cap385",), ()),
    CoordKeywordTransform(
        ("standard_1005",),
        (13.4299, 0.746361, -0.654923, 0.000878113, -0.0818352, 0.0023747, 0.852832, 0.941595, 0.85887),
    ),
    CoordKeywordTransform(
        ("gsn", "sfp"),
        (-0.254232, 0.0, -8.4081, 0.0, 0.00272526, 0.0, 8.59463, -10.9643, 10.4963),
    ),
    CoordKeywordTransform(("egi", "elp"), (0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 85.0, 85.0, 85.0)),
    CoordKeywordTransform(("egimff",), (0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 8.5, 8.5, 8.5)),
)
_BEM_TRANSFORMS: tuple[CoordKeywordTransform, ...] = (
    CoordKeywordTransform(("standard_1005",), (0.0, 0.0, 0.0, 0.0, 0.0, -math.pi / 2.0, 1.0, 1.0, 1.0)),
    CoordKeywordTransform(
        ("standard-10-5-cap385",),
        (0.832146, -15.6287, 2.41142, 0.0812144, 0.000937391, -1.5732, 1.17419, 1.06011, 1.14846),
    ),
    CoordKeywordTransform(
        ("gsn", "sfp"),
        (0.0547605, -17.3653, -8.13178, 0.0755019, 0.00318357, -1.56963, 11.7138, 12.7933, 12.213),
    ),
    CoordKeywordTransform(("egi", "elp"), (0.0, -15.0, 0.0, 0.08, 0.0, -1.571, 102.0, 93.0, 100.0)),
    CoordKeywordTransform(("egimff",), (0.0, -15.0, 0.0, 0.08, 0.0, -1.571, 10.2, 9.3, 10.0)),
)


STANDARD_TEMPLATES: tuple[DipfitTemplate, ...] = (
    DipfitTemplate(
        name="Template Spherical Four-Shell (BESA)",
        shortname="standardBESA",
        hdmfile="standard_BESA/standard_BESA.mat",
        mrifile="standard_BESA/avg152t1.mat",
        chanfile="standard_BESA/standard-10-5-cap385.elp",
        coordformat="Spherical",
        coord_transform=(),
        coord_transforms=_BESA_TRANSFORMS,
    ),
    DipfitTemplate(
        name="Template Boundary Element Model (MNI)",
        shortname="standardBEM",
        hdmfile="standard_BEM/standard_vol.mat",
        mrifile="standard_BEM/standard_mri.mat",
        chanfile="standard_BEM/elec/standard_1005.elc",
        coordformat="MNI",
        coord_transform=(0.0, 0.0, 0.0, 0.0, 0.0, -1.5708, 1.0, 1.0, 1.0),
        coord_transforms=_BEM_TRANSFORMS,
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


def lookupchantemplate(chanfile: str, template: DipfitTemplate) -> tuple[bool, list[float]]:
    """Return the first coregistration whose keywords are all in ``chanfile``.

    This follows EEGLAB ``lookupchantemplate``: every keyword must occur in the
    lowercased channel-file name, and the first complete match wins. No match
    returns an empty transform.
    """
    filename = str(chanfile or "").lower()
    for rule in template.coord_transforms:
        if rule.keywords and all(keyword.lower() in filename for keyword in rule.keywords):
            return True, [float(value) for value in rule.transform]
    return False, []


def channel_template_filename(EEG: dict[str, Any]) -> str:
    """Return ``EEG.chaninfo.filename`` when a dataset records one."""
    chaninfo = EEG.get("chaninfo")
    if isinstance(chaninfo, dict):
        return str(chaninfo.get("filename") or "")
    return ""


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
