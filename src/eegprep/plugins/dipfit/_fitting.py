"""EEGPrep-native DIPFIT fitting primitives.

``standardBESA`` uses the four-shell concentric-sphere EEG leadfield from
FieldTrip (``eeg_leadfield4`` via ``ft_compute_leadfield``), with the radii
and conductivities stored in ``standard_BESA.mat``. Grid search and nonlinear
refinement follow ``ft_dipolefitting``: a 24 by 24 by 12 scan inside the brain
shell, then Nelder-Mead with FieldTrip's ``fminsearch`` limits. MNI and BEM
settings keep the infinite-medium kernel until the boundary-element model is
ported through the same dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np
from scipy.optimize import minimize

from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.popfunc.plot_utils import component_channel_indices, component_maps, numeric_vector
from eegprep.plugins.dipfit._coordinates import apply_transform, traditionaldipfit
from eegprep.plugins.dipfit.forward.ft_compute_leadfield import ft_compute_leadfield
from eegprep.plugins.dipfit.forward.ft_headmodeltype import SPHERICAL_HEADMODELS, ft_headmodeltype, normalize_headmodel
from eegprep.plugins.dipfit.forward.ft_inside_headmodel import ft_inside_headmodel
from eegprep.plugins.dipfit.forward.ft_prepare_vol_sens import ft_prepare_vol_sens
from eegprep.plugins.dipfit.forward.standard_besa import standard_besa_headmodel
from eegprep.plugins.dipfit._utils import (
    component_count,
    coordinate_channel_indices,
    copy_eeg,
    ensure_channel_locations,
    ensure_dipfit_settings,
    ensure_ica,
    normalize_model_list,
    one_based_indices,
)


DEFAULT_HEAD_RADIUS_MM = 85.0
MIN_DIPOLE_ELECTRODE_DISTANCE_MM = 1e-3
DEFAULT_GRID_X_COUNT = 24
DEFAULT_GRID_Y_COUNT = 24
DEFAULT_GRID_Z_COUNT = 12
_FMINSEARCH_USUAL_DELTA = 0.05
_FMINSEARCH_ZERO_DELTA = 0.00025
_LINSPACE_PATTERN = re.compile(
    r"^linspace\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DipfitForwardData:
    """Prepared channels and ICA maps for EEGPrep-native dipole fitting."""

    positions: np.ndarray
    maps: np.ndarray
    components: list[int]
    chansel: list[int]
    labels: list[str]
    head_radius: float
    coordformat: str
    headmodel: dict[str, Any] | None = None


def dipfit_gridsearch(
    EEG: dict[str, Any],
    *,
    component: Any = None,
    xgrid: Any = None,
    ygrid: Any = None,
    zgrid: Any = None,
    reject: float | None = None,
) -> dict[str, Any]:
    """Fit initial single-dipole locations by scanning a regular grid."""
    forward = prepare_forward_data(EEG, component)
    x_default, y_default, z_default = default_scan_axes(forward.head_radius)
    x_values = parse_grid_values(xgrid, default=x_default)
    y_values = parse_grid_values(ygrid, default=y_default)
    z_values = parse_grid_values(zgrid, default=z_default)
    candidates = _candidate_grid(x_values, y_values, z_values, forward.head_radius, headmodel=forward.headmodel)
    if candidates.size == 0:
        raise ValueError("grid does not contain any points inside the spherical head")
    out = copy_eeg(EEG)
    models = ensure_model_list(out, component_count(out))
    for component_index in forward.components:
        fit = fit_component_grid(forward, component_index, candidates)
        models[component_index - 1].update(fit)
    if reject is not None:
        models = dipfit_reject(models, reject)
    out["dipfit"]["model"] = models
    return out


def dipfit_nonlinear(
    EEG: dict[str, Any],
    *,
    component: int,
    nonlinear: str | bool = "yes",
    symmetry: str | None = None,
    maxiter: int = 3000,
) -> dict[str, Any]:
    """Refine one component's dipole position and moment."""
    forward = prepare_forward_data(EEG, [component])
    out = copy_eeg(EEG)
    models = ensure_model_list(out, component_count(out))
    model = models[component - 1]
    positions = _model_positions(model, fallback=np.asarray([[0.0, 0.0, forward.head_radius * 0.5]]))
    selected = _selected_dipoles(model, positions.shape[0])
    positions = positions[selected]
    optimize_position = _is_nonlinear_enabled(nonlinear)
    if optimize_position:
        positions = _optimize_positions(forward, component, positions, symmetry=symmetry, maxiter=maxiter)
    fit = fit_component_at_positions(forward, component, positions)
    fit["select"] = [index + 1 for index in selected]
    models[component - 1].update(fit)
    out["dipfit"]["model"] = models
    return out


def dipfit_reject(models: Any, reject: float) -> list[dict[str, Any]]:
    """Return models with high-RV entries emptied like EEGLAB DIPFIT."""
    threshold = _threshold_fraction(reject)
    out = []
    for model in normalize_model_value(models):
        rv = model.get("rv", np.nan)
        if _finite_scalar(rv) and float(rv) <= threshold:
            out.append(dict(model))
            continue
        rejected = {key: [] for key in model}
        rejected["rv"] = 1.0
        if "component" in model:
            rejected["component"] = model["component"]
        out.append(rejected)
    return out


def remove_outside_head(
    models: Any, head_radius: float = DEFAULT_HEAD_RADIUS_MM
) -> tuple[list[dict[str, Any]], list[int]]:
    """Empty spherical dipoles outside the head and return removed component indices."""
    out = normalize_model_value(models)
    removed = []
    for index, model in enumerate(out, start=1):
        positions = np.asarray(model.get("posxyz", []), dtype=float)
        if positions.size == 0:
            continue
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        if np.any(np.linalg.norm(positions[:, :3], axis=1) > float(head_radius)):
            for key in list(model):
                if key != "component":
                    model[key] = []
            model["rv"] = 1.0
            removed.append(index)
    return out, removed


def prepare_forward_data(EEG: dict[str, Any], components: Any = None) -> DipfitForwardData:
    """Prepare selected channels, coordinates, and ICA maps for fitting."""
    ensure_channel_locations(EEG)
    ensure_ica(EEG)
    dipfit = ensure_dipfit_settings(EEG)
    count = component_count(EEG)
    selected_components = one_based_indices(components, limit=count, default_all=True)
    maps = np.asarray(component_maps(EEG), dtype=float)
    chanlocs = chanlocs_as_list(EEG.get("chanlocs", []))
    chansel = one_based_indices(dipfit.get("chansel", []), limit=len(chanlocs), default_all=True)
    usable = set(coordinate_channel_indices(EEG))
    chansel = [index for index in chansel if index in usable]
    if not chansel:
        raise ValueError("No channel locations with coordinates remain for DIPFIT")
    map_rows, chansel = _map_rows_for_channels(EEG, maps, chansel, len(chanlocs))
    positions = _channel_positions([chanlocs[index - 1] for index in chansel])
    headmodel = resolve_spherical_headmodel(dipfit)
    if headmodel is not None:
        head_radius = float(np.max(np.asarray(headmodel["r"], dtype=float)))
        positions = _scale_subcentimeter_positions(positions, head_radius)
        if dipfit.get("coord_transform"):
            positions = apply_transform(traditionaldipfit(dipfit["coord_transform"]), positions)
        _prepared, sensors = ft_prepare_vol_sens(headmodel, {"elecpos": positions})
        positions = np.asarray(sensors["elecpos"], dtype=float)
    else:
        head_radius = _head_radius(dipfit, positions)
        positions = _scale_or_project_to_head(positions, head_radius)
        if dipfit.get("coord_transform"):
            positions = apply_transform(traditionaldipfit(dipfit["coord_transform"]), positions)
    selected_maps = maps[map_rows, :][:, [component - 1 for component in selected_components]]
    labels = [str(chanlocs[index - 1].get("labels") or index) for index in chansel]
    return DipfitForwardData(
        positions=positions,
        maps=selected_maps,
        components=selected_components,
        chansel=chansel,
        labels=labels,
        head_radius=head_radius,
        coordformat=str(dipfit.get("coordformat", "")),
        headmodel=headmodel,
    )


def fit_component_grid(forward: DipfitForwardData, component: int, candidates: np.ndarray) -> dict[str, Any]:
    """Return the best single-dipole fit for one component over candidates."""
    topography = _average_reference(_component_topography(forward, component))
    fields = leadfield_matrix(forward.positions, candidates, forward.headmodel)
    best_index: int | None = None
    best_error = np.inf
    for index, field in enumerate(fields):
        if not np.isfinite(field).all():
            continue
        _moment, *_ = np.linalg.lstsq(field, topography, rcond=None)
        residual = topography - field @ _moment
        error = float(np.sum(residual**2))
        if error < best_error:
            best_error = error
            best_index = index
    if best_index is None:
        return empty_model(component)
    fit = fit_component_at_positions(forward, component, candidates[best_index])
    fit["select"] = [1]
    fit["active"] = [1]
    return fit


def fit_component_at_positions(
    forward: DipfitForwardData, component: int, positions: np.ndarray | list[list[float]]
) -> dict[str, Any]:
    """Fit dipole moments at fixed positions for one component."""
    fit = _fit_positions_to_topography(
        forward.positions,
        _component_topography(forward, component),
        np.asarray(positions),
        forward.headmodel,
    )
    fit["component"] = component
    fit["active"] = list(range(1, np.asarray(fit["posxyz"]).reshape(-1, 3).shape[0] + 1))
    return fit


def leadfield_matrix(
    electrodes: np.ndarray, sources: np.ndarray, headmodel: dict[str, Any] | None = None
) -> list[np.ndarray]:
    """Return one average-referenced 3-column leadfield per source point.

    A spherical ``headmodel`` uses ``ft_compute_leadfield``. Without one, the
    columns are the infinite-medium kernel kept for MNI and BEM settings.
    """
    source_points, _ = _as_positions(sources)
    if headmodel is None:
        return [_average_reference(_unit_moment_leadfield(electrodes, point)) for point in source_points]
    stacked = ft_compute_leadfield(source_points, {"elecpos": np.asarray(electrodes, dtype=float)}, headmodel)
    return [stacked[:, 3 * index : 3 * index + 3] for index in range(source_points.shape[0])]


def source_model_from_points(EEG: dict[str, Any], points: Any, *, labels: list[str] | None = None) -> dict[str, Any]:
    """Build an EEGLAB-shaped source model with leadfields for explicit points."""
    forward = prepare_forward_data(EEG)
    source_points, _ = _as_positions(points)
    fields = leadfield_matrix(forward.positions, source_points, forward.headmodel)
    if forward.headmodel is None:
        inside = [True] * source_points.shape[0]
    else:
        inside = [bool(value) for value in ft_inside_headmodel(source_points, forward.headmodel)]
    return {
        "pos": source_points.tolist(),
        "inside": inside,
        "leadfield": [field.tolist() for field in fields],
        "label": labels or forward.labels,
        "unit": "mm",
        "coordformat": forward.coordformat,
    }


def default_scan_axes(head_radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return DIPFIT's default x, y, and z scan axes for a head radius in mm.

    ``dipfitdefs.m`` builds ``linspace`` grids of 24, 24, and 12 samples out to
    ``floor`` of the outer BESA radius.
    """
    radius = float(np.floor(float(head_radius)))
    return (
        np.linspace(-radius, radius, DEFAULT_GRID_X_COUNT),
        np.linspace(-radius, radius, DEFAULT_GRID_Y_COUNT),
        np.linspace(0.0, radius, DEFAULT_GRID_Z_COUNT),
    )


def parse_grid_values(value: Any, *, default: np.ndarray) -> np.ndarray:
    """Parse grid values, including EEGLAB dialog ``linspace`` strings."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return np.asarray(default, dtype=float)
    if isinstance(value, str):
        match = _LINSPACE_PATTERN.match(value.strip())
        if match:
            start = float(match.group(1))
            stop = float(match.group(2))
            count = int(float(match.group(3)))
            return np.linspace(start, stop, count)
    values = numeric_vector(value, dtype=float)
    if values.size == 0:
        return np.asarray(default, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("grid values must be finite")
    return values


def ensure_model_list(EEG: dict[str, Any], count: int) -> list[dict[str, Any]]:
    """Ensure ``EEG.dipfit.model`` has one dictionary per ICA component."""
    dipfit = ensure_dipfit_settings(EEG)
    existing = normalize_model_list(EEG)
    models = [dict(model) for model in existing]
    while len(models) < count:
        models.append(empty_model(len(models) + 1))
    dipfit["model"] = models
    return models


def normalize_model_value(models: Any) -> list[dict[str, Any]]:
    """Normalize a raw model value into mutable dictionaries."""
    if isinstance(models, dict):
        return [dict(models)]
    if isinstance(models, np.ndarray):
        models = models.tolist()
    if isinstance(models, list):
        return [dict(model) if isinstance(model, dict) else {} for model in models]
    return []


def empty_model(component: int) -> dict[str, Any]:
    """Return an empty EEGLAB-shaped dipfit model entry."""
    return {
        "posxyz": [],
        "momxyz": [],
        "rv": 1.0,
        "diffmap": [],
        "sourcepot": [],
        "datapot": [],
        "component": component,
    }


def _fit_positions_to_topography(
    electrodes: np.ndarray,
    topography: np.ndarray,
    positions: np.ndarray,
    headmodel: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_positions, _ = _as_positions(positions)
    leadfield = np.column_stack(leadfield_matrix(electrodes, source_positions, headmodel))
    data = _average_reference(np.asarray(topography, dtype=float).ravel())
    if not np.any(np.isfinite(data)) or np.linalg.norm(data) == 0:
        return {
            "posxyz": source_positions.tolist(),
            "momxyz": np.zeros((source_positions.shape[0], 3)).tolist(),
            "rv": 1.0,
            "diffmap": (-data).tolist(),
            "sourcepot": np.zeros_like(data).tolist(),
            "datapot": data.tolist(),
        }
    moment, *_ = np.linalg.lstsq(leadfield, data, rcond=None)
    model = leadfield @ moment
    # dipfit_nonlinear stores Vmodel - Vdata.
    difference = model - data
    denominator = float(np.sum(data**2))
    rv = float(np.sum(difference**2) / denominator) if denominator > 0 else 1.0
    moments = moment.reshape(source_positions.shape[0], 3)
    return {
        "posxyz": source_positions.tolist(),
        "momxyz": moments.tolist(),
        "rv": min(max(rv, 0.0), 1.0),
        "diffmap": difference.tolist(),
        "sourcepot": model.tolist(),
        "datapot": data.tolist(),
    }


def _optimize_positions(
    forward: DipfitForwardData,
    component: int,
    initial: np.ndarray,
    *,
    symmetry: str | None,
    maxiter: int,
) -> np.ndarray:
    initial_positions, _ = _as_positions(initial)
    topography = _component_topography(forward, component)
    symmetry_axis = _symmetry_axis(symmetry)
    if symmetry_axis is not None and initial_positions.shape[0] == 2:
        initial_params = initial_positions[0].copy()

        def unpack(values: np.ndarray) -> np.ndarray:
            mirrored = values.copy()
            mirrored[symmetry_axis] = -mirrored[symmetry_axis]
            return np.vstack([values, mirrored])

    else:
        initial_params = initial_positions.ravel()

        def unpack(values: np.ndarray) -> np.ndarray:
            return values.reshape(initial_positions.shape)

    def objective(values: np.ndarray) -> float:
        fit = _fit_positions_to_topography(forward.positions, topography, unpack(values), forward.headmodel)
        rv = float(fit["rv"])
        if not np.isfinite(rv):
            return 1.0
        return rv

    start = np.asarray(initial_params, dtype=float).ravel()
    result = minimize(
        objective,
        start,
        method="Nelder-Mead",
        options={
            "maxiter": int(maxiter),
            "maxfev": 2 * int(maxiter) * int(start.size),
            "xatol": 1e-4,
            "fatol": 1e-4,
            "initial_simplex": _fminsearch_simplex(start),
            "adaptive": False,
            "disp": False,
        },
    )
    fitted = result.x if np.isfinite(result.x).all() else start
    return unpack(fitted)


def _candidate_grid(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    head_radius: float,
    headmodel: dict[str, Any] | None = None,
) -> np.ndarray:
    x_grid, y_grid, z_grid = np.meshgrid(x_values, y_values, z_values, indexing="ij")
    points = np.column_stack((x_grid.ravel(order="F"), y_grid.ravel(order="F"), z_grid.ravel(order="F")))
    if headmodel is None:
        inside = np.linalg.norm(points, axis=1) < float(head_radius) * 0.98
    else:
        inside = ft_inside_headmodel(points, headmodel)
    return points[inside]


def _fminsearch_simplex(start: np.ndarray) -> np.ndarray:
    """Return MATLAB ``fminsearch``'s initial simplex around ``start``."""
    origin = np.asarray(start, dtype=float).ravel()
    simplex = np.tile(origin, (origin.size + 1, 1))
    for index, value in enumerate(origin):
        if value != 0.0:
            simplex[index + 1, index] = (1.0 + _FMINSEARCH_USUAL_DELTA) * value
        else:
            simplex[index + 1, index] = _FMINSEARCH_ZERO_DELTA
    return simplex


def _component_topography(forward: DipfitForwardData, component: int) -> np.ndarray:
    local_index = forward.components.index(component)
    return forward.maps[:, local_index]


def _unit_moment_leadfield(electrodes: np.ndarray, position: np.ndarray) -> np.ndarray:
    delta = electrodes - position.reshape(1, 3)
    distances = np.linalg.norm(delta, axis=1)
    distances = np.maximum(distances, MIN_DIPOLE_ELECTRODE_DISTANCE_MM)
    return delta / distances[:, np.newaxis] ** 3


def _average_reference(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array - np.mean(array)
    return array - np.mean(array, axis=0, keepdims=True)


def _map_rows_for_channels(
    EEG: dict[str, Any], maps: np.ndarray, chansel: list[int], chanloc_count: int
) -> tuple[list[int], list[int]]:
    zero_based = [index - 1 for index in chansel]
    if maps.shape[0] == chanloc_count:
        return zero_based, chansel
    ica_channels = component_channel_indices(EEG, chanloc_count).tolist()
    rows = []
    kept_chansel = []
    for channel_index, channel in zip(chansel, zero_based, strict=True):
        if channel in ica_channels:
            rows.append(ica_channels.index(channel))
            kept_chansel.append(channel_index)
    if not rows:
        raise ValueError("No DIPFIT channels overlap with ICA channel indices")
    return rows, kept_chansel


def _channel_positions(chanlocs: list[dict[str, Any]]) -> np.ndarray:
    positions = []
    for chanloc in chanlocs:
        xyz = [chanloc.get(key) for key in ("X", "Y", "Z")]
        if all(_finite_scalar(value) for value in xyz):
            positions.append([float(value) for value in xyz])
        else:
            positions.append(_polar_to_cartesian(chanloc))
    return np.asarray(positions, dtype=float)


def _polar_to_cartesian(chanloc: dict[str, Any]) -> list[float]:
    theta = np.deg2rad(float(chanloc.get("theta", 0.0) or 0.0))
    radius = min(abs(float(chanloc.get("radius", 0.5) or 0.5)) / 0.5, 1.0)
    z = np.sqrt(max(0.0, 1.0 - radius**2))
    return [radius * np.cos(theta), radius * np.sin(theta), z]


def _head_radius(dipfit: dict[str, Any], positions: np.ndarray) -> float:
    vol = dipfit.get("vol")
    if isinstance(vol, dict) and "r" in vol:
        radius = float(np.max(np.asarray(vol["r"], dtype=float)))
        if np.isfinite(radius) and radius > 0:
            return radius
    norms = np.linalg.norm(np.asarray(positions, dtype=float), axis=1)
    finite = norms[np.isfinite(norms) & (norms > 0)]
    if finite.size and np.median(finite) > 20:
        return float(np.median(finite))
    return DEFAULT_HEAD_RADIUS_MM


def resolve_spherical_headmodel(dipfit: dict[str, Any]) -> dict[str, Any] | None:
    """Return a concentric-sphere volume, or ``None`` for MNI and BEM settings."""
    vol = dipfit.get("vol")
    if isinstance(vol, dict) and "r" in vol and np.size(vol.get("r")):
        model = normalize_headmodel(vol)
        if ft_headmodeltype(model) in SPHERICAL_HEADMODELS:
            return model
        return None
    hdmfile = str(dipfit.get("hdmfile") or "").lower()
    coordformat = str(dipfit.get("coordformat") or "").lower()
    if "standard_besa" in hdmfile or coordformat == "spherical":
        return standard_besa_headmodel()
    return None


def _scale_subcentimeter_positions(positions: np.ndarray, head_radius: float) -> np.ndarray:
    norms = np.linalg.norm(positions, axis=1)
    finite = norms[np.isfinite(norms) & (norms > 0)]
    if finite.size == 0:
        raise ValueError("channel positions are degenerate")
    scaled = np.asarray(positions, dtype=float).copy()
    median = float(np.median(finite))
    if median < 20:
        scaled *= float(head_radius) / median
    return scaled


def _scale_or_project_to_head(positions: np.ndarray, head_radius: float) -> np.ndarray:
    norms = np.linalg.norm(positions, axis=1)
    finite = norms[np.isfinite(norms) & (norms > 0)]
    if finite.size == 0:
        raise ValueError("channel positions are degenerate")
    scaled = np.asarray(positions, dtype=float).copy()
    median = float(np.median(finite))
    if median < 20:
        scaled *= float(head_radius) / median
        norms = np.linalg.norm(scaled, axis=1)
    mask = norms > 0
    scaled[mask] = scaled[mask] / norms[mask, np.newaxis] * float(head_radius)
    return scaled


def _model_positions(model: dict[str, Any], *, fallback: np.ndarray) -> np.ndarray:
    raw = np.asarray(model.get("posxyz", []), dtype=float)
    if raw.size == 0:
        return fallback
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if raw.shape[1] < 3:
        return fallback
    return raw[:, :3]


def _selected_dipoles(model: dict[str, Any], count: int) -> list[int]:
    raw = model.get("select", model.get("active", []))
    if raw is None or raw == []:
        return list(range(count))
    values = np.asarray(raw, dtype=int).ravel()
    selected = [int(value) - 1 for value in values if 1 <= int(value) <= count]
    return selected or list(range(count))


def _symmetry_axis(value: str | None) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    axis = str(value).strip().lower()
    if axis == "x":
        return 0
    if axis == "y":
        return 1
    if axis == "z":
        return 2
    raise ValueError("symmetry must be 'x', 'y', 'z', or empty")


def _is_nonlinear_enabled(value: str | bool) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"no", "off", "false", "0"}
    return bool(value)


def _threshold_fraction(value: float | int | str) -> float:
    threshold = float(np.asarray(value, dtype=float).ravel()[0])
    if threshold > 1.0:
        threshold /= 100.0
    if threshold < 0:
        raise ValueError("reject threshold must be non-negative")
    return threshold


def _as_positions(value: Any) -> tuple[np.ndarray, bool]:
    array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        if array.size != 3:
            raise ValueError("positions must have 3 columns")
        return array.reshape(1, 3), False
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError("positions must be an Nx3 array")
    return array[:, :3], False


def _finite_scalar(value: Any) -> bool:
    try:
        numeric = float(np.asarray(value).ravel()[0])
    except (IndexError, TypeError, ValueError):
        return False
    return bool(np.isfinite(numeric))


__all__ = [
    "DEFAULT_HEAD_RADIUS_MM",
    "DipfitForwardData",
    "dipfit_gridsearch",
    "dipfit_nonlinear",
    "dipfit_reject",
    "empty_model",
    "ensure_model_list",
    "fit_component_at_positions",
    "fit_component_grid",
    "leadfield_matrix",
    "default_scan_axes",
    "parse_grid_values",
    "resolve_spherical_headmodel",
    "prepare_forward_data",
    "remove_outside_head",
    "source_model_from_points",
]
