"""DIPFIT fitting primitives.

Datasets that use the MNI boundary-element template are fitted with the
FieldTrip dipoli leadfield (``ft_prepare_vol_sens`` and ``ft_compute_leadfield``).
Other head models keep the single-sphere primary-current leadfield. eLORETA and
creation of a new BEM system matrix are not part of this path.
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
from eegprep.plugins.dipfit.fieldtrip.bem import ft_prepare_vol_sens, is_bem_hdmfile, load_bem_headmodel
from eegprep.plugins.dipfit.fieldtrip.ft_compute_leadfield import ft_compute_leadfield
from eegprep.plugins.dipfit.fieldtrip.geometry import ft_inside_headmodel
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
_BEM_TYPES = frozenset({"bem", "dipoli", "asa", "bemcp"})
_BEM_XGRID = np.linspace(-DEFAULT_HEAD_RADIUS_MM, DEFAULT_HEAD_RADIUS_MM, 24)
_BEM_YGRID = np.linspace(-DEFAULT_HEAD_RADIUS_MM, DEFAULT_HEAD_RADIUS_MM, 24)
_BEM_ZGRID = np.linspace(0.0, DEFAULT_HEAD_RADIUS_MM, 12)
_BEM_OPTIMIZER_MAXITER = 1000
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
    sens: dict[str, Any] | None = None


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
    if forward.headmodel is not None:
        x_default, y_default, z_default = _BEM_XGRID, _BEM_YGRID, _BEM_ZGRID
    else:
        x_default = np.linspace(-forward.head_radius, forward.head_radius, 11)
        y_default = np.linspace(-forward.head_radius, forward.head_radius, 11)
        z_default = np.linspace(0.0, forward.head_radius, 6)
    x_values = parse_grid_values(xgrid, default=x_default)
    y_values = parse_grid_values(ygrid, default=y_default)
    z_values = parse_grid_values(zgrid, default=z_default)
    candidates = _candidate_grid(x_values, y_values, z_values, forward.head_radius, headmodel=forward.headmodel)
    if candidates.size == 0:
        message = (
            "grid does not contain any points inside the brain compartment"
            if forward.headmodel is not None
            else "grid does not contain any points inside the spherical head"
        )
        raise ValueError(message)
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
    maxiter: int = 200,
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


def default_search_grids(EEG: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the grid ``pop_dipfit_gridsearch`` uses when the caller omits one.

    The MNI boundary-element model uses EEGLAB's ``linspace(-85,85,24)`` and
    ``linspace(0,85,12)`` search. Spherical fits keep the 11 by 11 by 6 grid.
    """
    dipfit = EEG.get("dipfit") if isinstance(EEG.get("dipfit"), dict) else {}
    if _uses_bem(dipfit or {}):
        return _BEM_XGRID.copy(), _BEM_YGRID.copy(), _BEM_ZGRID.copy()
    radius = DEFAULT_HEAD_RADIUS_MM
    return (
        np.linspace(-radius, radius, 11),
        np.linspace(-radius, radius, 11),
        np.linspace(0.0, radius, 6),
    )


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
    head_radius = _head_radius(dipfit, positions)
    labels = [str(chanlocs[index - 1].get("labels") or index) for index in chansel]
    headmodel = None
    sens = None
    if _uses_bem(dipfit):
        if dipfit.get("coord_transform"):
            positions = apply_transform(traditionaldipfit(dipfit["coord_transform"]), positions)
        sens = {"elecpos": positions, "label": labels, "unit": "mm"}
        headmodel, sens = ft_prepare_vol_sens(_load_bem_volume(dipfit), sens)
        positions = np.asarray(sens["elecpos"], dtype=float)
    else:
        positions = _scale_or_project_to_head(positions, head_radius)
        if dipfit.get("coord_transform"):
            positions = apply_transform(traditionaldipfit(dipfit["coord_transform"]), positions)
    selected_maps = maps[map_rows, :][:, [component - 1 for component in selected_components]]
    return DipfitForwardData(
        positions=positions,
        maps=selected_maps,
        components=selected_components,
        chansel=chansel,
        labels=labels,
        head_radius=head_radius,
        coordformat=str(dipfit.get("coordformat", "")),
        headmodel=headmodel,
        sens=sens,
    )


def fit_component_grid(forward: DipfitForwardData, component: int, candidates: np.ndarray) -> dict[str, Any]:
    """Return the best single-dipole fit for one component over candidates."""
    if forward.headmodel is not None:
        return _fit_bem_component_grid(forward, component, candidates)
    topography = _component_topography(forward, component)
    best: dict[str, Any] | None = None
    best_rv = np.inf
    for position in candidates:
        fit = _fit_positions_to_topography(forward.positions, topography, position.reshape(1, 3))
        if fit["rv"] < best_rv:
            best = fit
            best_rv = fit["rv"]
    if best is None:
        return empty_model(component)
    best["component"] = component
    best["select"] = [1]
    best["active"] = [1]
    return best


def fit_component_at_positions(
    forward: DipfitForwardData, component: int, positions: np.ndarray | list[list[float]]
) -> dict[str, Any]:
    """Fit dipole moments at fixed positions for one component."""
    fit = _fit_positions_to_topography(
        forward.positions,
        _component_topography(forward, component),
        np.asarray(positions),
        forward=forward,
    )
    fit["component"] = component
    fit["active"] = list(range(1, np.asarray(fit["posxyz"]).reshape(-1, 3).shape[0] + 1))
    return fit


def leadfield_matrix(electrodes: np.ndarray, sources: np.ndarray) -> list[np.ndarray]:
    """Return one average-referenced 3-column leadfield per source point."""
    source_points, _ = _as_positions(sources)
    return [_average_reference(_unit_moment_leadfield(electrodes, point)) for point in source_points]


def source_model_from_points(EEG: dict[str, Any], points: Any, *, labels: list[str] | None = None) -> dict[str, Any]:
    """Build an EEGLAB-shaped source model with leadfields for explicit points."""
    forward = prepare_forward_data(EEG)
    source_points, _ = _as_positions(points)
    if forward.headmodel is not None:
        combined = ft_compute_leadfield(source_points, forward.sens, forward.headmodel)
        fields = [combined[:, 3 * index : 3 * index + 3] for index in range(source_points.shape[0])]
        inside = ft_inside_headmodel(source_points, forward.headmodel).tolist()
    else:
        fields = leadfield_matrix(forward.positions, source_points)
        inside = [True] * source_points.shape[0]
    return {
        "pos": source_points.tolist(),
        "inside": inside,
        "leadfield": [field.tolist() for field in fields],
        "label": labels or forward.labels,
        "unit": "mm",
        "coordformat": forward.coordformat,
    }


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
    *,
    forward: DipfitForwardData | None = None,
) -> dict[str, Any]:
    source_positions, _ = _as_positions(positions)
    bem = forward is not None and forward.headmodel is not None
    if bem:
        leadfield = ft_compute_leadfield(source_positions, forward.sens, forward.headmodel)
    else:
        leadfield = np.column_stack(leadfield_matrix(electrodes, source_positions))
    data = _average_reference(np.asarray(topography, dtype=float).ravel())
    if not np.any(np.isfinite(data)) or np.linalg.norm(data) == 0:
        return {
            "posxyz": source_positions.tolist(),
            "momxyz": np.zeros((source_positions.shape[0], 3)).tolist(),
            "rv": 1.0,
            "diffmap": data.tolist(),
            "sourcepot": np.zeros_like(data).tolist(),
            "datapot": data.tolist(),
        }
    if bem:
        moment = np.linalg.pinv(leadfield) @ data
    else:
        moment, *_ = np.linalg.lstsq(leadfield, data, rcond=None)
    model = leadfield @ moment
    residual = data - model
    denominator = float(np.sum(data**2))
    rv = float(np.sum(residual**2) / denominator) if denominator > 0 else 1.0
    moments = moment.reshape(source_positions.shape[0], 3)
    return {
        "posxyz": source_positions.tolist(),
        "momxyz": moments.tolist(),
        "rv": min(max(rv, 0.0), 1.0),
        "diffmap": residual.tolist(),
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
        first = initial_positions[0].copy()
        first[symmetry_axis] = abs(first[symmetry_axis]) or forward.head_radius * 0.45
        initial_params = first

        def unpack(values: np.ndarray) -> np.ndarray:
            mirrored = values.copy()
            mirrored[symmetry_axis] = -mirrored[symmetry_axis]
            return np.vstack([values, mirrored])

    else:
        initial_params = initial_positions.ravel()

        def unpack(values: np.ndarray) -> np.ndarray:
            return values.reshape(initial_positions.shape)

    def objective(values: np.ndarray) -> float:
        positions = unpack(values)
        if forward.headmodel is None and np.any(np.linalg.norm(positions, axis=1) >= forward.head_radius * 0.995):
            return 1e3 + float(np.max(np.linalg.norm(positions, axis=1)))
        return float(_fit_positions_to_topography(forward.positions, topography, positions, forward=forward)["rv"])

    if forward.headmodel is None:
        result = minimize(
            objective,
            initial_params,
            method="Nelder-Mead",
            options={"maxiter": int(maxiter), "xatol": 1e-4, "fatol": 1e-7, "disp": False},
        )
    else:
        result = minimize(
            objective,
            initial_params,
            method="BFGS",
            options={"maxiter": _BEM_OPTIMIZER_MAXITER, "gtol": 1e-9, "eps": 1e-3, "disp": False},
        )
    return unpack(result.x if result.success or np.isfinite(result.fun) else initial_params)


def _candidate_grid(
    x_values: np.ndarray,
    y_values: np.ndarray,
    z_values: np.ndarray,
    head_radius: float,
    *,
    headmodel: dict[str, Any] | None = None,
) -> np.ndarray:
    if headmodel is not None:
        xx, yy, zz = np.meshgrid(x_values, y_values, z_values, indexing="ij")
        points = np.column_stack([xx.ravel(order="F"), yy.ravel(order="F"), zz.ravel(order="F")])
        return points[ft_inside_headmodel(points, headmodel)]
    points = np.asarray(np.meshgrid(x_values, y_values, z_values, indexing="ij"), dtype=float).reshape(3, -1).T
    inside = np.linalg.norm(points, axis=1) < float(head_radius) * 0.98
    return points[inside]


def _fit_bem_component_grid(forward: DipfitForwardData, component: int, candidates: np.ndarray) -> dict[str, Any]:
    data = _average_reference(_component_topography(forward, component).ravel())
    if not np.any(np.isfinite(data)) or float(np.sum(data**2)) == 0.0:
        return empty_model(component)
    leadfield = ft_compute_leadfield(candidates, forward.sens, forward.headmodel)
    denominator = float(np.sum(data**2))
    best_index = 0
    best_rv = np.inf
    best_moment = np.zeros(3)
    best_model = np.zeros_like(data)
    for index in range(candidates.shape[0]):
        block = leadfield[:, 3 * index : 3 * index + 3]
        moment = np.linalg.pinv(block) @ data
        model = block @ moment
        rv = float(np.sum((data - model) ** 2) / denominator)
        if rv < best_rv:
            best_rv = rv
            best_index = index
            best_moment = moment
            best_model = model
    residual = data - best_model
    return {
        "posxyz": [candidates[best_index].tolist()],
        "momxyz": [best_moment.reshape(3).tolist()],
        "rv": min(max(float(best_rv), 0.0), 1.0),
        "diffmap": residual.tolist(),
        "sourcepot": best_model.tolist(),
        "datapot": data.tolist(),
        "component": component,
        "select": [1],
        "active": [1],
    }


def _uses_bem(dipfit: dict[str, Any]) -> bool:
    volume = dipfit.get("vol")
    if isinstance(volume, dict) and "bnd" in volume and "mat" in volume:
        kind = str(volume.get("type", "bem")).lower()
        if kind in _BEM_TYPES:
            return True
    return is_bem_hdmfile(dipfit.get("hdmfile"))


def _load_bem_volume(dipfit: dict[str, Any]) -> dict[str, Any]:
    volume = dipfit.get("vol")
    if isinstance(volume, dict) and "bnd" in volume and "mat" in volume:
        return volume
    return load_bem_headmodel(str(dipfit.get("hdmfile") or ""))


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
    "default_search_grids",
    "dipfit_gridsearch",
    "dipfit_nonlinear",
    "dipfit_reject",
    "empty_model",
    "ensure_model_list",
    "fit_component_at_positions",
    "fit_component_grid",
    "leadfield_matrix",
    "parse_grid_values",
    "prepare_forward_data",
    "remove_outside_head",
    "source_model_from_points",
]
