"""Shared helpers for EEGPrep's bundled DIPFIT surfaces."""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np

from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.popfunc.plot_utils import numeric_vector, parse_plot_options_text, python_literal

_QUOTED_PLOT_GAP = re.compile(r"""(?<=['"])\s+(?=['"])""")


class DIPFITUnavailableError(NotImplementedError):
    """DIPFIT stopped because this EEGPrep build cannot perform the request."""


_BESA_TRANSFORMS = (
    (('standard-10-5-cap385',), ()),
    (
        ('standard_1005',),
        (13.4299, 0.746361, -0.654923, 0.000878113, -0.0818352, 0.0023747, 0.852832, 0.941595, 0.85887),
    ),
    (('gsn', 'sfp'), (-0.254232, 0.0, -8.4081, 0.0, 0.00272526, 0.0, 8.59463, -10.9643, 10.4963)),
    (('egi', 'elp'), (0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 85.0, 85.0, 85.0)),
    (('egimff',), (0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 8.5, 8.5, 8.5)),
)

_BEM_TRANSFORMS = (
    (('standard_1005',), (0.0, 0.0, 0.0, 0.0, 0.0, -1.5707963267948966, 1.0, 1.0, 1.0)),
    (
        ('standard-10-5-cap385',),
        (0.832146, -15.6287, 2.41142, 0.0812144, 0.000937391, -1.5732, 1.17419, 1.06011, 1.14846),
    ),
    (('gsn', 'sfp'), (0.0547605, -17.3653, -8.13178, 0.0755019, 0.00318357, -1.56963, 11.7138, 12.7933, 12.213)),
    (('egi', 'elp'), (0.0, -15.0, 0.0, 0.08, 0.0, -1.571, 102.0, 93.0, 100.0)),
    (('egimff',), (0.0, -15.0, 0.0, 0.08, 0.0, -1.571, 10.2, 9.3, 10.0)),
)

STANDARD_TEMPLATES: tuple[dict[str, Any], ...] = (
    dict(
        name='Template Spherical Four-Shell (BESA)',
        shortname='standardBESA',
        hdmfile='standard_BESA/standard_BESA.mat',
        mrifile='standard_BESA/avg152t1.mat',
        chanfile='standard_BESA/standard-10-5-cap385.elp',
        coordformat='Spherical',
        coord_transform=(),
        coord_transforms=_BESA_TRANSFORMS,
    ),
    dict(
        name='Template Boundary Element Model (MNI)',
        shortname='standardBEM',
        hdmfile='standard_BEM/standard_vol.mat',
        mrifile='standard_BEM/standard_mri.mat',
        chanfile='standard_BEM/elec/standard_1005.elc',
        coordformat='MNI',
        coord_transform=(),
        coord_transforms=_BEM_TRANSFORMS,
    ),
    dict(
        name='Custom files/structures (Fieldtrip format)',
        shortname='',
        hdmfile='',
        mrifile='',
        chanfile='',
        coordformat='',
        coord_transform=(),
        coord_transforms=(),
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


def template_form_state(template: dict[str, Any]) -> dict[str, Any]:
    """Return dialog field values for one head-model template."""
    coord_index = {"spherical": 1, "mni": 2}.get(template["coordformat"].lower(), 4)
    transform = " ".join(f"{float(value):g}" for value in template["coord_transform"])
    return {
        "hdmfile": template["hdmfile"],
        "mrifile": template["mrifile"],
        "chanfile": template["chanfile"],
        "coord_transform": transform,
        "coordformat": coord_index,
        "no_coreg": not bool(transform),
        "browse": not bool(template["shortname"]),
    }


def matlab_plot_options(text: Any) -> dict[str, Any]:
    """Parse plot options, including MATLAB ``'key' 'value'`` text."""
    normalized = _QUOTED_PLOT_GAP.sub(", ", str(text or "").strip())
    return parse_plot_options_text(normalized)


def dipfit_history(function_name: str, *args: Any, assignment: bool = True, **kwargs: Any) -> str:
    """Build an EEGLAB history command the console can replay."""
    pieces = ["EEG", *(_history_token(arg) for arg in args)]
    for key, value in kwargs.items():
        if _empty_history_value(value):
            continue
        pieces.append(python_literal(key))
        pieces.append(_history_token(value))
    command = f"{function_name}({', '.join(pieces)});"
    return f"EEG = {command}" if assignment else command


def _history_token(value: Any) -> str:
    if isinstance(value, str) and value.strip().startswith("linspace("):
        return value.strip()
    return python_literal(value)


def template_by_name(name: Any) -> dict[str, Any]:
    """Return a standard template by EEGLAB short name or display name."""
    requested = str(name or "standardBEM").strip().lower()
    for template in STANDARD_TEMPLATES:
        if requested in {template["shortname"].lower(), template["name"].lower()}:
            return template
    valid = ", ".join(template["shortname"] for template in STANDARD_TEMPLATES if template["shortname"])
    raise ValueError(f"model must be one of {valid}")


def lookupchantemplate(chanfile: str, template: dict[str, Any]) -> tuple[bool, list[float]]:
    """Return the first coregistration whose keywords are all in ``chanfile``.

    This follows EEGLAB ``lookupchantemplate``: every keyword must occur in the
    lowercased channel-file name, and the first complete match wins. No match
    returns an empty transform.
    """
    filename = str(chanfile or "").lower()
    for keywords, transform in template["coord_transforms"]:
        if keywords and all(keyword.lower() in filename for keyword in keywords):
            return True, list(transform)
    return False, []


def packaged_dipfit_file(relative: str) -> str:
    """Return a packaged DIPFIT path, or ``relative`` when it is already a file."""
    text = str(relative or "").replace("\\", "/").strip()
    if not text:
        return ""
    path = Path(text)
    if path.is_file():
        return str(path)
    resource = files("eegprep").joinpath("resources", "dipfit", text)
    if resource.is_file():
        return str(resource)
    return text


def align_channels_to_template(
    EEG: dict[str, Any],
    chanfile: str,
    *,
    fiducials_only: bool = False,
    warpmethod: str = "traditional",
) -> list[float]:
    """Warp channel locations onto a template file the way ``coregister(..., 'warp','auto')`` does."""
    from eegprep.functions.sigprocfunc.coregister import coregister

    path = packaged_dipfit_file(chanfile)
    chaninfo = EEG.get("chaninfo") if isinstance(EEG.get("chaninfo"), dict) else None
    if fiducials_only:
        nodatchans = None if chaninfo is None else chaninfo.get("nodatchans")
        if _fiducial_count(nodatchans) < 3:
            raise ValueError(
                "pop_dipfit_settings: coord_transform 'warpfiducials' needs nasion, LPA, and RPA "
                "in EEG.chaninfo.nodatchans. Add those fiducials with pop_chanedit, or pass a 9-value transform."
            )
        source = nodatchans
    else:
        source = EEG.get("chanlocs")
    fitted = coregister(source, path, chaninfo1=chaninfo, warp="auto", warpmethod=warpmethod)
    return np.asarray(fitted.transform, dtype=float).reshape(-1).tolist()


def resolved_coord_transform(EEG: dict[str, Any], template: dict[str, Any]) -> tuple[bool, list[float]]:
    """Return ``(keyword matched, transform)``.

    A keyword match is kept even when its transform is empty (BESA cap385).
    Otherwise the montage is warped onto the template landmarks. EEGLAB leaves
    that field blank and refuses Ok; EEGPrep fills the warp when it can.
    """
    found, transform = lookupchantemplate(channel_template_filename(EEG), template)
    if found:
        return True, list(transform)
    chanfile = str(template.get("chanfile") or "")
    if not chanfile:
        return False, []
    try:
        return False, align_channels_to_template(EEG, chanfile)
    except (OSError, ValueError):
        return False, []


def _fiducial_count(nodatchans: Any) -> int:
    if nodatchans is None:
        return 0
    if isinstance(nodatchans, dict):
        return 1 if nodatchans.get("labels") else 0
    try:
        return len(nodatchans)
    except TypeError:
        return 0


def channel_template_filename(EEG: dict[str, Any]) -> str:
    """Return ``EEG.chaninfo.filename`` when a dataset records one."""
    chaninfo = EEG.get("chaninfo")
    if isinstance(chaninfo, dict):
        return str(chaninfo.get("filename") or "")
    return ""


def apply_template(dipfit: dict[str, Any], template: dict[str, Any]) -> None:
    """Apply standard-template file and coordinate defaults in-place."""
    if not template["shortname"]:
        return
    dipfit["hdmfile"] = template["hdmfile"]
    dipfit["mrifile"] = template["mrifile"]
    dipfit["chanfile"] = template["chanfile"]
    dipfit["coordformat"] = template["coordformat"]
    dipfit["coord_transform"] = list(template["coord_transform"])


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
    dipfit = dipfit_record(EEG)
    if not dipfit:
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


def unavailable(function_name: str, problem: str, fix: str) -> None:
    """Raise ``DIPFITUnavailableError`` naming the function, the problem, and the fix."""
    raise DIPFITUnavailableError(f"{function_name}: {problem} {fix}")


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
