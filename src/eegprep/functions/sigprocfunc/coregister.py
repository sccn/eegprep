"""EEGLAB-style electrode/head-model coregistration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
import re
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from eegprep.functions.popfunc._chanutils import chanlocs_as_list

DEFAULT_COREGISTER_TRANSFORM = np.asarray([0.0, -10.0, 0.0, -0.1, 0.0, -1.6, 1100.0, 1100.0, 1100.0])
_SCALE_LOWER_BOUND = 1e-9

_FIDUCIAL_LABELS = {"nz", "nasion", "nazion", "fidnz", "lpa", "left", "fidt9", "rpa", "right", "fidt10"}
_LABEL_1020 = {
    "nz",
    "lpa",
    "rpa",
    "fp1",
    "fpz",
    "fp2",
    "f7",
    "f3",
    "fz",
    "f4",
    "f8",
    "t7",
    "c3",
    "cz",
    "c4",
    "t8",
    "p7",
    "p3",
    "pz",
    "p4",
    "p8",
    "o1",
    "oz",
    "o2",
}
_LABEL_1010 = {
    "nz",
    "lpa",
    "rpa",
    "fp1",
    "fpz",
    "fp2",
    "af9",
    "af7",
    "af5",
    "af3",
    "af1",
    "afz",
    "af2",
    "af4",
    "af6",
    "af8",
    "af10",
    "f9",
    "f7",
    "f5",
    "f3",
    "f1",
    "fz",
    "f2",
    "f4",
    "f6",
    "f8",
    "f10",
    "ft9",
    "ft7",
    "fc5",
    "fc3",
    "fc1",
    "fcz",
    "fc2",
    "fc4",
    "fc6",
    "ft8",
    "ft10",
    "t9",
    "t7",
    "c5",
    "c3",
    "c1",
    "cz",
    "c2",
    "c4",
    "c6",
    "t8",
    "t10",
    "tp9",
    "tp7",
    "cp5",
    "cp3",
    "cp1",
    "cpz",
    "cp2",
    "cp4",
    "cp6",
    "tp8",
    "tp10",
    "p9",
    "p7",
    "p5",
    "p3",
    "p1",
    "pz",
    "p2",
    "p4",
    "p6",
    "p8",
    "p10",
    "po9",
    "po7",
    "po5",
    "po3",
    "po1",
    "poz",
    "po2",
    "po4",
    "po6",
    "po8",
    "po10",
    "o1",
    "oz",
    "o2",
    "i1",
    "iz",
    "i2",
}


@dataclass(frozen=True)
class ElectrodeSet:
    """Labels and Cartesian positions for one electrode montage."""

    labels: list[str]
    points: np.ndarray
    source: str = ""


@dataclass(frozen=True)
class CoregistrationResult:
    """Coregistration output returned by :func:`coregister`."""

    electrodes: ElectrodeSet
    transform: np.ndarray


def coregister(
    chanlocs1: Any,
    chanlocs2: Any = None,
    *,
    transform: Any = None,
    chaninfo1: dict[str, Any] | None = None,
    chaninfo2: dict[str, Any] | None = None,
    warp: str | list[str] | tuple[str, ...] | None = None,
    warpmethod: str = "traditional",
    alignfid: Any = None,
    autoscale: str = "on",
) -> CoregistrationResult:
    """Coregister one electrode montage to another without opening a GUI.

    This mirrors EEGLAB's noninteractive ``coregister(..., 'manual', 'off')``
    path for the transform types EEGPrep can compute standalone. ``warp='auto'``
    or ``warpmethod='traditional'`` fits the 9-parameter Talairach transform to
    common labels. ``alignfid`` and ``warpmethod='globalrescale'`` fit a shared
    scale transform.
    """
    source = load_coregistration_electrodes(chanlocs1, chaninfo=chaninfo1)
    target = load_coregistration_electrodes(chanlocs2, chaninfo=chaninfo2) if chanlocs2 is not None else None
    if target is not None and (warp is not None or alignfid is not None):
        method = "globalrescale" if alignfid is not None or warpmethod.lower() == "globalrescale" else "traditional"
        labels = (
            None
            if warp is None or (isinstance(warp, str) and warp.lower() == "auto")
            else [str(label) for label in np.asarray(warp, dtype=object).ravel()]
        )
        transform_array = estimate_coregistration_transform(
            source, target, initial=transform, method=method, labels=labels
        )
    else:
        transform_array = normalise_coregistration_transform(transform, default=[0, 0, 0, 0, 0, 0, 1, 1, 1])
        if target is not None and str(autoscale).lower() == "on" and transform is None:
            transform_array[6:9] = _radius_ratio(source.points, target.points)
    transformed = ElectrodeSet(
        source.labels, apply_coregistration_transform(source.points, transform_array), source.source
    )
    return CoregistrationResult(transformed, transform_array)


def load_coregistration_electrodes(source: Any, *, chaninfo: dict[str, Any] | None = None) -> ElectrodeSet:
    """Load electrode labels and positions from channel locations or a loc file."""
    if source is None:
        raise ValueError("coregistration requires electrode locations")
    if isinstance(source, (str, Path)):
        return read_electrode_file(source)
    return electrodes_from_chanlocs(source, chaninfo=chaninfo)


def read_electrode_file(path: str | Path) -> ElectrodeSet:
    """Read an EEGLAB-style ``.xyz``/``.sfp`` electrode file."""
    loc_path = _resolve_coregistration_file(path)
    labels: list[str] = []
    points: list[list[float]] = []
    for line in loc_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("%", "#")):
            continue
        parts = stripped.split()
        parsed = _parse_location_line(parts)
        if parsed is None:
            continue
        label, point = parsed
        labels.append(label)
        points.append(point)
    if not points:
        raise ValueError(f"no electrode positions found in {loc_path}")
    return ElectrodeSet(labels, np.asarray(points, dtype=float), str(loc_path))


def _resolve_coregistration_file(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.exists():
        return path
    resource = resources.files("eegprep").joinpath("resources").joinpath("headplot").joinpath(path.name)
    with resources.as_file(resource) as resolved:
        if resolved.exists():
            return Path(resolved)
    raise FileNotFoundError(f"electrode location file not found: {value}")


def electrodes_from_chanlocs(chanlocs: Any, *, chaninfo: dict[str, Any] | None = None) -> ElectrodeSet:
    """Convert EEG channel locations plus fiducials to a coregistration montage."""
    locs = chanlocs_as_list(chanlocs)
    chaninfo = dict(chaninfo or {})
    nodatchans = chanlocs_as_list(chaninfo.get("nodatchans")) if chaninfo.get("nodatchans") is not None else []
    labels: list[str] = []
    points: list[list[float]] = []
    for loc in [*locs, *nodatchans]:
        if not isinstance(loc, dict):
            continue
        point = _location_point(loc)
        if point is None:
            continue
        label = str(loc.get("labels") or len(labels) + 1).strip()
        labels.append(label)
        points.append(point)
    if not points:
        raise ValueError("coregistration requires channels with X/Y/Z or theta/radius coordinates")
    return ElectrodeSet(labels, np.asarray(points, dtype=float), "chanlocs")


def normalise_coregistration_transform(transform: Any, *, default: Any = None) -> np.ndarray:
    """Return a finite 9-value traditional Talairach transform."""
    if transform is None or (isinstance(transform, str) and not transform.strip()):
        values = np.asarray(DEFAULT_COREGISTER_TRANSFORM if default is None else default, dtype=float).ravel()
    else:
        values = np.asarray(_parse_numeric_sequence(transform), dtype=float).ravel()
    if values.size == 6:
        values = np.concatenate([values, np.ones(3)])
    if values.size != 9 or not np.isfinite(values).all():
        raise ValueError("coregistration transform must contain 9 finite values")
    return values.astype(float, copy=True)


def traditional_transform_matrix(transform: Any) -> np.ndarray:
    """Return EEGLAB/DIPFIT's homogeneous matrix for a 9-parameter transform."""
    values = normalise_coregistration_transform(transform)
    tx, ty, tz, pitch, roll, yaw, sx, sy, sz = values
    c_x, c_y, c_z = np.cos([pitch, roll, yaw])
    s_x, s_y, s_z = np.sin([pitch, roll, yaw])
    translation = np.eye(4)
    translation[:3, 3] = [tx, ty, tz]
    rotation = np.eye(4)
    rotation[0, 0] = c_z * c_y + s_z * s_x * s_y
    rotation[0, 1] = s_z * c_y + c_z * s_x * s_y
    rotation[0, 2] = c_x * s_y
    rotation[1, 0] = -s_z * c_x
    rotation[1, 1] = c_z * c_x
    rotation[1, 2] = s_x
    rotation[2, 0] = s_z * s_x * c_y - c_z * s_y
    rotation[2, 1] = -c_z * s_x * c_y - s_z * s_y
    rotation[2, 2] = c_x * c_y
    scale = np.diag([sx, sy, sz, 1.0])
    return translation @ rotation @ scale


def apply_coregistration_transform(points: Any, transform: Any) -> np.ndarray:
    """Apply a traditional or homogeneous transform to Cartesian points."""
    coordinates = np.asarray(points, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coregistration points must have shape (n, 3)")
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4):
        matrix = traditional_transform_matrix(transform)
    homogeneous = np.column_stack([coordinates, np.ones(coordinates.shape[0])])
    transformed = matrix @ homogeneous.T
    return transformed[:3].T


def estimate_coregistration_transform(
    source: ElectrodeSet,
    target: ElectrodeSet,
    *,
    initial: Any = None,
    method: str = "traditional",
    labels: list[str] | None = None,
) -> np.ndarray:
    """Fit a traditional transform to common source/target electrode labels."""
    source_indices, target_indices, _labels = match_electrodes(source, target, labels=labels)
    if source_indices.size < 3:
        raise ValueError("at least three common electrode labels are required for coregistration")
    source_points = source.points[source_indices]
    target_points = target.points[target_indices]
    method = method.lower()
    if method not in {"traditional", "globalrescale"}:
        raise NotImplementedError(
            "EEGPrep coregistration supports standalone 'traditional' and 'globalrescale' alignment"
        )
    initial_transform = _initial_transform(source_points, target_points)
    if initial is not None:
        user_initial = _canonicalise_fit_transform(
            normalise_coregistration_transform(initial, default=initial_transform)
        )
        if (
            _fit_rms(source_points, target_points, user_initial)
            <= _fit_rms(source_points, target_points, initial_transform) * 5
        ):
            initial_transform = user_initial
    initial_transform = _canonicalise_fit_transform(initial_transform)
    if method == "globalrescale":
        x0 = np.concatenate([initial_transform[:6], [max(float(np.mean(initial_transform[6:9])), _SCALE_LOWER_BOUND)]])
        result = least_squares(
            lambda values: _fit_residual(source_points, target_points, _expand_globalrescale(values)),
            x0,
            bounds=([-np.inf] * 6 + [_SCALE_LOWER_BOUND], [np.inf] * 7),
            max_nfev=5000,
        )
        return _canonicalise_fit_transform(_expand_globalrescale(result.x))
    result = least_squares(
        lambda values: _fit_residual(source_points, target_points, values),
        initial_transform,
        bounds=([-np.inf] * 6 + [_SCALE_LOWER_BOUND] * 3, [np.inf] * 9),
        max_nfev=5000,
    )
    return _canonicalise_fit_transform(result.x)


def match_electrodes(
    source: ElectrodeSet, target: ElectrodeSet, *, labels: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return paired source/target indices for common labels."""
    wanted = {_normalise_label(label) for label in labels} if labels is not None else None
    target_lookup: dict[str, int] = {}
    for index, label in enumerate(target.labels):
        key = _normalise_label(label)
        if key and key not in target_lookup:
            target_lookup[key] = index
    source_indices: list[int] = []
    target_indices: list[int] = []
    matched_labels: list[str] = []
    for index, label in enumerate(source.labels):
        key = _normalise_label(label)
        if not key or (wanted is not None and key not in wanted) or key not in target_lookup:
            continue
        source_indices.append(index)
        target_indices.append(target_lookup[key])
        matched_labels.append(label)
    return np.asarray(source_indices, dtype=int), np.asarray(target_indices, dtype=int), matched_labels


def electrode_subset_indices(
    electrodes: ElectrodeSet, subset: str | list[int] | tuple[int, ...] | np.ndarray
) -> np.ndarray:
    """Decode EEGLAB coregister reference-electrode subset names."""
    if not isinstance(subset, str):
        values = np.asarray(subset, dtype=int).ravel()
        if values.size == 0:
            return np.arange(len(electrodes.labels), dtype=int)
        if np.min(values) >= 1:
            values = values - 1
        return values[(values >= 0) & (values < len(electrodes.labels))]
    key = subset.lower().strip()
    if key.startswith("21"):
        wanted = _LABEL_1020
    elif key.startswith("86"):
        wanted = _LABEL_1010
    elif key.startswith("all"):
        return np.arange(len(electrodes.labels), dtype=int)
    else:
        raise ValueError(f"unknown electrode subset: {subset}")
    return np.asarray(
        [index for index, label in enumerate(electrodes.labels) if _normalise_label(label) in wanted],
        dtype=int,
    )


def is_fiducial_label(label: str) -> bool:
    """Return true for common fiducial names used by EEGLAB coregister."""
    return _normalise_label(label) in _FIDUCIAL_LABELS


def format_transform(transform: Any) -> str:
    """Format a transform like EEGLAB's ``num2str`` field."""
    values = normalise_coregistration_transform(transform)
    return " ".join(f"{value:.6g}" for value in values)


def _parse_location_line(parts: list[str]) -> tuple[str, list[float]] | None:
    if len(parts) >= 5 and _is_number(parts[0]) and all(_is_number(value) for value in parts[1:4]):
        return parts[4], [float(parts[1]), float(parts[2]), float(parts[3])]
    if len(parts) >= 4 and not _is_number(parts[0]) and all(_is_number(value) for value in parts[1:4]):
        return parts[0], [float(parts[1]), float(parts[2]), float(parts[3])]
    if len(parts) >= 4 and all(_is_number(value) for value in parts[:3]) and not _is_number(parts[3]):
        return parts[3], [float(parts[0]), float(parts[1]), float(parts[2])]
    return None


def _location_point(loc: dict[str, Any]) -> list[float] | None:
    if all(_has_coordinate(loc, key) for key in ("X", "Y", "Z")):
        return [_coordinate_value(loc["X"]), _coordinate_value(loc["Y"]), _coordinate_value(loc["Z"])]
    if all(_has_coordinate(loc, key) for key in ("theta", "radius")):
        theta = np.deg2rad(_coordinate_value(loc["theta"]))
        radius = _coordinate_value(loc["radius"])
        return [float(radius * np.sin(theta)), float(radius * np.cos(theta)), 0.0]
    return None


def _has_coordinate(loc: dict[str, Any], key: str) -> bool:
    if key not in loc:
        return False
    value = loc[key]
    if value is None or (isinstance(value, str) and not value.strip()):
        return False
    return np.asarray(value).size > 0


def _coordinate_value(value: Any) -> float:
    return float(np.asarray(value, dtype=float).ravel()[0])


def _parse_numeric_sequence(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.astype(float).ravel().tolist()
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=float).ravel().tolist()
    if isinstance(value, (int, float, np.integer, np.floating)):
        return [float(value)]
    text = str(value).strip().strip("[]")
    if not text:
        return []
    return [float(item) for item in re.split(r"[\s,]+", text) if item]


def _normalise_label(label: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", str(label).strip().lower())
    return {
        "nasion": "nz",
        "nazion": "nz",
        "fidnz": "nz",
        "left": "lpa",
        "fidt9": "lpa",
        "right": "rpa",
        "fidt10": "rpa",
    }.get(
        key,
        key,
    )


def _initial_transform(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    ratio = _radius_ratio(source_points, target_points)
    translation = np.nanmean(target_points, axis=0) - np.nanmean(source_points * ratio, axis=0)
    return np.asarray([translation[0], translation[1], translation[2], 0, 0, 0, ratio, ratio, ratio], dtype=float)


def _radius_ratio(source_points: np.ndarray, target_points: np.ndarray) -> float:
    source_radius = np.linalg.norm(np.asarray(source_points, dtype=float), axis=1)
    target_radius = np.linalg.norm(np.asarray(target_points, dtype=float), axis=1)
    source_mean = float(np.nanmean(source_radius[source_radius > 0]))
    target_mean = float(np.nanmean(target_radius[target_radius > 0]))
    if not np.isfinite(source_mean) or source_mean == 0 or not np.isfinite(target_mean):
        return 1.0
    return target_mean / source_mean


def _fit_residual(source_points: np.ndarray, target_points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    transformed = apply_coregistration_transform(source_points, transform)
    scale = max(float(np.nanmedian(np.linalg.norm(target_points, axis=1))), 1.0)
    return ((transformed - target_points) / scale).ravel()


def _fit_rms(source_points: np.ndarray, target_points: np.ndarray, transform: np.ndarray) -> float:
    residual = _fit_residual(source_points, target_points, transform)
    return float(np.sqrt(np.nanmean(residual**2)))


def _expand_globalrescale(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[6], values[6]]
    )


def _canonicalise_fit_transform(transform: Any) -> np.ndarray:
    values = normalise_coregistration_transform(transform)
    values[3:6] = (values[3:6] + np.pi) % (2 * np.pi) - np.pi
    values[6:9] = np.maximum(np.abs(values[6:9]), _SCALE_LOWER_BOUND)
    return values


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


__all__ = [
    "CoregistrationResult",
    "DEFAULT_COREGISTER_TRANSFORM",
    "ElectrodeSet",
    "apply_coregistration_transform",
    "coregister",
    "electrode_subset_indices",
    "electrodes_from_chanlocs",
    "estimate_coregistration_transform",
    "format_transform",
    "is_fiducial_label",
    "load_coregistration_electrodes",
    "match_electrodes",
    "normalise_coregistration_transform",
    "read_electrode_file",
    "traditional_transform_matrix",
]
