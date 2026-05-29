"""EEGLAB ``headplot`` spline setup and 3-D scalp rendering."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.io import loadmat, savemat
from scipy.special import eval_legendre

from eegprep.functions.miscfunc.misc import finite_matmul, finite_pinv
from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.popfunc._pop_utils import is_on as _is_on_value
from eegprep.functions.popfunc._pop_utils import parse_numeric_sequence as _parse_numeric_sequence_value
from eegprep.functions.sigprocfunc.coregister import (
    DEFAULT_COREGISTER_TRANSFORM,
    apply_coregistration_transform,
    normalise_coregistration_transform,
    traditional_transform_matrix,
)

DEFAULT_MESH = "mheadnew.mat"
DEFAULT_TRANSFORM = DEFAULT_COREGISTER_TRANSFORM
COLIN27_TRANSFORM = np.asarray([0.0, -15.0, -15.0, 0.05, 0.0, -1.57, 100.0, 88.0, 110.0])
HEAD_CENTER = np.asarray([0.0, 0.0, 30.0], dtype=float)
HEADPLOT_FACE_COLOR = np.asarray([0.88, 0.605, 0.385, 1.0], dtype=float)
DEFAULT_HEADPLOT_LIGHTS = np.asarray(
    [[-125.0, 125.0, 80.0], [125.0, 125.0, 80.0], [125.0, -125.0, 125.0], [-125.0, -125.0, 125.0]],
    dtype=float,
)
HEADPLOT_VIEW_SCALE = 0.78
ELECTRODE_DISPLAY_FACTOR = 1.06
MAPLIMIT_PADDING = 1.1
# EEGLAB's headplot.m adds this scalar to G before solving the constrained
# spline system. This is intentionally not a diagonal ridge; keep it for
# numerical parity with EEGLAB history/setup replays.
EEGLAB_SPLINE_LAMBDA = 0.1


@dataclass(frozen=True)
class HeadplotMesh:
    """Head mesh arrays used by ``headplot``."""

    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray | None
    scalp_indices: np.ndarray
    center: np.ndarray
    source: str


@dataclass(frozen=True)
class HeadplotSpline:
    """Loaded headplot spline metadata."""

    path: Path
    xe: np.ndarray
    ye: np.ndarray
    ze: np.ndarray
    g: np.ndarray
    gx: np.ndarray
    new_electrodes: np.ndarray
    electrode_names: list[str]
    indices: np.ndarray
    transform: np.ndarray
    meshfile: str
    vertex_indices: np.ndarray
    headplot_version: int


def headplot(values: Any, arg1: Any, **kwargs: Any):
    """Plot values on a spline-interpolated 3-D head mesh.

    Args:
        values: One data value per electrode, or one value per original channel
            when the spline stores channel indices.
        arg1: Path to a ``.spl`` file created by :func:`headplot_setup`.
        **kwargs: EEGLAB-style options including ``meshfile``, ``title``,
            ``maplimits``, ``electrodes``, ``labels``, ``view`` and ``cbar``.
    """
    if isinstance(values, str) and values.lower() == "setup":
        return headplot_setup(arg1, kwargs.pop("splinefile"), **kwargs)
    spline = load_headplot_spline(arg1)
    data = _values_for_spline(values, spline)
    mesh = load_headplot_mesh(kwargs.get("meshfile") or spline.meshfile or DEFAULT_MESH)
    interpolated = _interpolate_values(data, spline)
    maplimits = _map_limits(interpolated, kwargs.get("maplimits", "absmax"))

    ax = kwargs.get("ax")
    if ax is None:
        fig = plt.figure(figsize=(5.6, 5.2))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure
    _plot_head_mesh(ax, mesh, spline, interpolated, maplimits, kwargs)
    _set_view(ax, kwargs.get("view", [143, 18]))
    if str(kwargs.get("electrodes", "on")).lower() == "on":
        _plot_electrodes(ax, spline, labels=int(kwargs.get("labels", 0) or 0))
    title = str(kwargs.get("title", "") or "")
    if title:
        ax.set_title(title)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    if kwargs.get("cbar", None) is not None:
        mappable = cm.ScalarMappable(norm=colors.Normalize(*maplimits), cmap=plt.get_cmap("turbo"))
        fig.colorbar(mappable, ax=ax, shrink=0.7)
    if kwargs.get("tight_layout", True):
        fig.tight_layout()
    return fig


def headplot_setup(
    chanlocs: Any,
    splinefile: str | Path,
    *,
    meshfile: str | Path | None = None,
    transform: Any = None,
    chaninfo: dict[str, Any] | None = None,
    ica: str | bool = "off",
    plotchans: Any = None,
    comment: str = "",
    orilocs: str = "off",
    plotmeshonly: str = "off",
) -> Path:
    """Create an EEGLAB-compatible ``.spl`` spline setup file.

    The generated file is a MATLAB ``.mat`` file with the conventional
    ``.spl`` extension. It contains the spline matrices and transformed
    electrode positions needed for later :func:`headplot` calls.
    """
    if str(plotmeshonly).lower() != "off":
        raise NotImplementedError("headplot plotmeshonly preview is not yet available in EEGPrep")
    if str(orilocs).lower() != "off":
        raise NotImplementedError("headplot setup with original unprojected locations is not yet available in EEGPrep")

    output = _normalise_spline_path(splinefile)
    output.parent.mkdir(parents=True, exist_ok=True)
    locs = chanlocs_as_list(chanlocs)
    if not locs:
        raise ValueError("headplot setup requires channel locations")
    chaninfo = dict(chaninfo or {})
    selected, value_indices = _setup_channel_selection(plotchans, chaninfo, len(locs), _is_on(ica), locs)
    labels = _channel_labels(locs, selected)
    coordinates = _channel_xyz(locs, selected)
    coordinates = _rotate_for_nosedir(coordinates, str(chaninfo.get("nosedir", "+x")))
    transform_array = _normalise_transform(transform)
    transformed = _apply_transform(coordinates, transform_array)
    unit_electrodes = _normalise_to_head_sphere(transformed)

    mesh = load_headplot_mesh(meshfile or DEFAULT_MESH)
    sphere_vertices = _mesh_unit_sphere(mesh)
    g_matrix = _spherical_spline_matrix(unit_electrodes)
    gx = _spherical_spline_between(sphere_vertices[mesh.scalp_indices], unit_electrodes)
    new_electrodes = _project_electrodes_to_mesh(
        unit_electrodes, mesh.vertices[mesh.scalp_indices], sphere_vertices[mesh.scalp_indices]
    )

    savemat(
        output,
        {
            "Xe": unit_electrodes[:, 0],
            "Ye": unit_electrodes[:, 1],
            "Ze": unit_electrodes[:, 2],
            "G": g_matrix,
            "gx": gx,
            "newElect": new_electrodes,
            "ElectrodeNames": np.asarray(labels, dtype=object),
            "indices": value_indices + 1,
            "comment": comment,
            "headplot_version": 2,
            "transform": transform_array,
            "meshfile": str(mesh.source),
            "index1": mesh.scalp_indices + 1,
        },
        do_compression=True,
    )
    return output


def load_headplot_spline(path: str | Path) -> HeadplotSpline:
    """Load an EEGPrep/EEGLAB headplot spline file."""
    spline_path = _normalise_spline_path(path)
    if not spline_path.exists():
        raise FileNotFoundError(f"headplot spline file not found: {spline_path}")
    mat = loadmat(spline_path, squeeze_me=True, struct_as_record=False)
    required = ("Xe", "Ye", "Ze", "G", "gx", "newElect", "ElectrodeNames")
    missing = [name for name in required if name not in mat]
    if missing:
        raise ValueError(f"headplot spline file is missing required fields: {', '.join(missing)}")
    names = np.asarray(mat["ElectrodeNames"], dtype=object).ravel().tolist()
    names = [str(name).strip() for name in names]
    indices = np.asarray(mat.get("indices", np.arange(len(names)) + 1), dtype=int).ravel()
    if indices.size and indices.min() >= 1:
        indices = indices - 1
    transform = np.asarray(mat.get("transform", DEFAULT_TRANSFORM), dtype=float).ravel()
    meshfile = str(np.asarray(mat.get("meshfile", DEFAULT_MESH)).item()) if "meshfile" in mat else DEFAULT_MESH
    vertex_indices = np.asarray(mat.get("index1", []), dtype=int).ravel()
    if vertex_indices.size and np.min(vertex_indices) >= 1:
        vertex_indices = vertex_indices - 1
    return HeadplotSpline(
        path=spline_path,
        xe=np.asarray(mat["Xe"], dtype=float).ravel(),
        ye=np.asarray(mat["Ye"], dtype=float).ravel(),
        ze=np.asarray(mat["Ze"], dtype=float).ravel(),
        g=np.asarray(mat["G"], dtype=float),
        gx=np.asarray(mat["gx"], dtype=float),
        new_electrodes=np.asarray(mat["newElect"], dtype=float),
        electrode_names=names,
        indices=indices,
        transform=transform,
        meshfile=meshfile,
        vertex_indices=vertex_indices,
        headplot_version=int(np.asarray(mat.get("headplot_version", 1)).item()),
    )


def load_headplot_mesh(meshfile: str | Path | None = None) -> HeadplotMesh:
    """Load a packaged or user-provided EEGLAB head mesh."""
    mesh_path = _resolve_headplot_file(meshfile or DEFAULT_MESH)
    mat = loadmat(mesh_path, squeeze_me=True, struct_as_record=False)
    if "POS" in mat and "TRI1" in mat:
        vertices = np.asarray(mat["POS"], dtype=float)
        faces = np.asarray(mat["TRI1"], dtype=int)
        normals = np.asarray(mat["NORM"], dtype=float) if "NORM" in mat and np.asarray(mat["NORM"]).size else None
        center = np.asarray(mat.get("center", [0, 0, 0]), dtype=float).ravel()
        scalp_indices = np.asarray(mat.get("index1", []), dtype=int).ravel()
    elif "vertices" in mat and "faces" in mat:
        vertices = np.asarray(mat["vertices"], dtype=float)
        faces = np.asarray(mat["faces"], dtype=int)
        normals = None
        center = np.zeros(3)
        scalp_indices = np.asarray(mat.get("index1", []), dtype=int).ravel()
    else:
        raise ValueError(f"Unknown headplot mesh file format: {mesh_path}")
    faces = _zero_based_faces(faces)
    if scalp_indices.size and np.min(scalp_indices) >= 1:
        scalp_indices = scalp_indices - 1
    if scalp_indices.size == 0:
        scalp_indices = np.unique(faces.ravel())
    return HeadplotMesh(vertices, faces, normals, scalp_indices, center, str(mesh_path))


def default_headplot_transform(chaninfo: dict[str, Any] | None = None) -> np.ndarray:
    """Return the EEGLAB pop_headplot default transform for known templates."""
    filename = str((chaninfo or {}).get("filename", "")).lower()
    if "standard-10-5-cap385" in filename:
        return np.asarray(
            [-0.355789, -6.33688, 12.3705, 0.0533239, 0.0187461, -1.55264, 1.06367, 0.987721, 0.932694],
            dtype=float,
        )
    if "standard_1005" in filename:
        return np.asarray(
            [-1.13598, 7.75226, 11.4527, -0.0271167, 0.0155306, -1.54547, 0.912338, 0.931611, 0.806978],
            dtype=float,
        )
    if "gsn" in filename or "sfp" in filename:
        return np.asarray(
            [0.664455, -3.39403, -14.2521, -0.00241453, 0.015519, -1.55584, 11, 10.1455, 12],
            dtype=float,
        )
    if "egi" in filename or "elp" in filename:
        return np.asarray([0.0773, -5.3235, -14.72, -0.1187, -0.0023, -1.5940, 92.4, 92.5, 110.9], dtype=float)
    return np.asarray([], dtype=float)


def default_headplot_mesh_transform(
    meshfile: str | Path | None = None, chaninfo: dict[str, Any] | None = None
) -> np.ndarray:
    """Return the default transform for a ``pop_headplot`` mesh selection."""
    filename = Path(str(meshfile or DEFAULT_MESH)).name.lower()
    if filename == "colin27headmesh.mat":
        return COLIN27_TRANSFORM.copy()
    template_transform = default_headplot_transform(chaninfo)
    if template_transform.size:
        return template_transform
    return DEFAULT_TRANSFORM.copy()


def packaged_headplot_path(name: str) -> Path:
    """Return a packaged headplot support-file path."""
    return _resolve_headplot_file(name)


def _normalise_spline_path(path: str | Path) -> Path:
    spline_path = Path(path).expanduser()
    if not spline_path.suffix:
        spline_path = spline_path.with_suffix(".spl")
    return spline_path


def _resolve_headplot_file(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.exists():
        return path
    resource = resources.files("eegprep").joinpath("resources").joinpath("headplot").joinpath(path.name)
    with resources.as_file(resource) as resolved:
        if resolved.exists():
            return Path(resolved)
    raise FileNotFoundError(f"headplot support file not found: {value}")


def _normalise_transform(transform: Any) -> np.ndarray:
    try:
        return normalise_coregistration_transform(transform, default=DEFAULT_TRANSFORM)
    except ValueError as exc:
        raise ValueError("headplot transform must contain 9 finite values") from exc


def _parse_numeric_sequence(value: Any) -> list[float]:
    return _parse_numeric_sequence_value(value, dtype=float)


def _setup_channel_selection(
    plotchans: Any,
    chaninfo: dict[str, Any],
    count: int,
    is_ica: bool,
    locs: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    selected = _setup_channel_indices(plotchans, chaninfo, count, is_ica)
    if not is_ica:
        located = _filter_located_channels(selected, locs)
        return located, located

    icachans = _setup_channel_indices(None, chaninfo, count, True)
    located_positions = []
    located_channels = set(_filter_located_channels(selected, locs).tolist())
    for position, channel_index in enumerate(icachans):
        if int(channel_index) in located_channels:
            located_positions.append(position)
    if not located_positions:
        raise ValueError("headplot setup requires at least one channel with X/Y/Z or theta/radius coordinates")
    value_indices = np.asarray(located_positions, dtype=int)
    return icachans[value_indices], value_indices


def _setup_channel_indices(plotchans: Any, chaninfo: dict[str, Any], count: int, is_ica: bool) -> np.ndarray:
    if plotchans is not None and _parse_numeric_sequence(plotchans):
        values = np.asarray(_parse_numeric_sequence(plotchans), dtype=int)
        if values.min() < 1 or values.max() > count:
            raise ValueError(f"plotchans must be 1-based and within 1..{count}")
        return values - 1
    if not is_ica:
        return np.arange(count, dtype=int)
    raw = chaninfo.get("icachansind", [])
    values = np.asarray(raw, dtype=float).ravel()
    if values.size == 0:
        return np.arange(count, dtype=int)
    if np.any(values != values.astype(int)):
        raise ValueError("icachansind must contain integer channel indices")
    indices = values.astype(int)
    if np.all(indices >= 1) and np.max(indices) <= count and 0 not in indices:
        indices = indices - 1
    if np.any(indices < 0) or np.any(indices >= count):
        raise ValueError(f"icachansind values must be within 0..{count - 1}")
    return indices


def _filter_located_channels(indices: np.ndarray, locs: list[dict[str, Any]]) -> np.ndarray:
    located = [int(index) for index in indices if _has_usable_coordinates(locs[int(index)])]
    if not located:
        raise ValueError("headplot setup requires at least one channel with X/Y/Z or theta/radius coordinates")
    return np.asarray(located, dtype=int)


def _channel_labels(locs: list[dict[str, Any]], indices: np.ndarray) -> list[str]:
    labels = []
    for index in indices:
        label = str(locs[int(index)].get("labels") or int(index) + 1).strip()
        labels.append(label)
    return labels


def _channel_xyz(locs: list[dict[str, Any]], indices: np.ndarray) -> np.ndarray:
    coords = []
    for index in indices:
        loc = locs[int(index)]
        if all(_has_coordinate(loc, key) for key in ("X", "Y", "Z")):
            coords.append([_coordinate_value(loc["X"]), _coordinate_value(loc["Y"]), _coordinate_value(loc["Z"])])
            continue
        if "theta" in loc and "radius" in loc:
            theta = np.deg2rad(_coordinate_value(loc["theta"]))
            radius = _coordinate_value(loc["radius"])
            coords.append([radius * np.sin(theta), radius * np.cos(theta), 0.0])
            continue
        raise ValueError("headplot setup requires channel locations with X/Y/Z or theta/radius coordinates")
    coordinates = np.asarray(coords, dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError("headplot setup channel coordinates must be finite")
    return coordinates


def _has_coordinate(loc: dict[str, Any], key: str) -> bool:
    if key not in loc:
        return False
    value = loc[key]
    if value is None or (isinstance(value, str) and value == ""):
        return False
    return np.asarray(value).size > 0


def _has_usable_coordinates(loc: dict[str, Any]) -> bool:
    return all(_has_coordinate(loc, key) for key in ("X", "Y", "Z")) or all(
        _has_coordinate(loc, key) for key in ("theta", "radius")
    )


def _coordinate_value(value: Any) -> float:
    return float(np.asarray(value, dtype=float).ravel()[0])


def _rotate_for_nosedir(coordinates: np.ndarray, nosedir: str) -> np.ndarray:
    direction = nosedir.lower()
    if direction == "+x":
        return coordinates.copy()
    if direction == "+y":
        angle = 3 * np.pi / 2
    elif direction == "-x":
        angle = np.pi
    else:
        angle = np.pi / 2
    complex_xy = (coordinates[:, 1] + 1j * coordinates[:, 0]) * np.exp(1j * angle)
    rotated = coordinates.copy()
    rotated[:, 0] = np.imag(complex_xy)
    rotated[:, 1] = np.real(complex_xy)
    return rotated


def _traditional_transform_matrix(transform: np.ndarray) -> np.ndarray:
    return traditional_transform_matrix(transform)


def _apply_transform(coordinates: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return apply_coregistration_transform(coordinates, transform)


def _normalise_to_head_sphere(coordinates: np.ndarray) -> np.ndarray:
    centered = coordinates - HEAD_CENTER
    norm = np.linalg.norm(centered, axis=1)
    if np.any(norm == 0):
        raise ValueError("headplot setup encountered a channel at the head center")
    return centered / norm[:, np.newaxis]


def _mesh_unit_sphere(mesh: HeadplotMesh) -> np.ndarray:
    centered = mesh.vertices - HEAD_CENTER
    norm = np.linalg.norm(centered, axis=1)
    norm[norm == 0] = 1.0
    return centered / norm[:, np.newaxis]


def _spherical_spline_matrix(electrodes: np.ndarray) -> np.ndarray:
    distances = _one_minus_distance(electrodes, electrodes)
    return _calc_gx(distances)


def _spherical_spline_between(points: np.ndarray, electrodes: np.ndarray) -> np.ndarray:
    distances = _one_minus_distance(points, electrodes)
    return _calc_gx(distances)


def _one_minus_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    diff = left[:, np.newaxis, :] - right[np.newaxis, :, :]
    return 1.0 - np.linalg.norm(diff, axis=2)


def _calc_gx(values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=float)
    for degree in range(1, 8):
        out += ((2 * degree + 1) / (degree**4 * (degree + 1) ** 4)) * eval_legendre(degree, values)
    return out / (4 * np.pi)


def _project_electrodes_to_mesh(
    unit_electrodes: np.ndarray, mesh_vertices: np.ndarray, sphere_vertices: np.ndarray
) -> np.ndarray:
    projected = []
    for electrode in unit_electrodes:
        distances = np.linalg.norm(sphere_vertices - electrode, axis=1)
        nearest = np.argsort(distances)[:3]
        delta = np.mean(mesh_vertices[nearest] - sphere_vertices[nearest], axis=0)
        projected.append(electrode + delta * ELECTRODE_DISPLAY_FACTOR)
    return np.asarray(projected, dtype=float)


def _values_for_spline(values: Any, spline: HeadplotSpline) -> np.ndarray:
    data = np.asarray(values, dtype=float).ravel()
    if data.size == spline.xe.size:
        return data
    if spline.indices.size and data.size > np.max(spline.indices):
        return data[spline.indices]
    raise ValueError("headplot values must match the spline electrode count or original channel count")


def _interpolate_values(values: np.ndarray, spline: HeadplotSpline) -> np.ndarray:
    centered = values - np.nanmean(values)
    enum = values.size
    system = np.vstack([spline.g + EEGLAB_SPLINE_LAMBDA, np.ones((1, enum))])
    target = np.concatenate([centered, [0.0]])
    coefficients = finite_matmul(finite_pinv(system), target)
    return finite_matmul(spline.gx, coefficients) + np.nanmean(values)


def _map_limits(values: np.ndarray, setting: Any) -> tuple[float, float]:
    if isinstance(setting, str):
        lower = setting.lower()
        if lower in {"maxmin", "minmax"}:
            return float(np.nanmin(values) * MAPLIMIT_PADDING), float(np.nanmax(values) * MAPLIMIT_PADDING)
        if lower == "absmax":
            limit = float(np.nanmax(np.abs(values)) * MAPLIMIT_PADDING)
            return -limit, limit
    numeric = np.asarray(setting, dtype=float).ravel()
    if numeric.size == 2:
        return float(numeric[0]), float(numeric[1])
    raise ValueError("headplot maplimits must be 'absmax', 'maxmin', or [min max]")


def _plot_head_mesh(
    ax: Any,
    mesh: HeadplotMesh,
    spline: HeadplotSpline,
    interpolated: np.ndarray,
    maplimits: tuple[float, float],
    kwargs: dict[str, Any],
) -> None:
    vertex_values = np.full(mesh.vertices.shape[0], np.nan)
    indices = spline.vertex_indices if spline.vertex_indices.size else mesh.scalp_indices
    if indices.size != interpolated.size:
        indices = mesh.scalp_indices[: interpolated.size]
    vertex_values[indices] = interpolated
    with np.errstate(invalid="ignore"):
        face_values = np.nanmean(vertex_values[mesh.faces], axis=1)
    cmap = plt.get_cmap("turbo")
    norm = colors.Normalize(*maplimits)
    facecolors = cmap(norm(face_values))
    facecolors[~np.isfinite(face_values)] = HEADPLOT_FACE_COLOR
    if str(kwargs.get("lighting", "on")).lower() == "off":
        collection = Poly3DCollection(mesh.vertices[mesh.faces], facecolors=facecolors, linewidths=0.15)
        collection.set_edgecolor("0.45")
    else:
        facecolors = _lit_facecolors(mesh.vertices, mesh.faces, facecolors, kwargs.get("lights"))
        collection = Poly3DCollection(mesh.vertices[mesh.faces], facecolors=facecolors, linewidths=0.0)
        collection.set_edgecolor("none")
        collection.set_antialiased(False)
    ax.add_collection3d(collection)
    _autoscale_3d(ax, mesh.vertices)


def _lit_facecolors(
    vertices: np.ndarray,
    faces: np.ndarray,
    facecolors: np.ndarray,
    lights: Any,
) -> np.ndarray:
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normal_lengths = np.linalg.norm(normals, axis=1)
    valid = normal_lengths > 0
    normals[valid] = normals[valid] / normal_lengths[valid, None]
    centroids = np.nanmean(triangles, axis=1)
    outward = centroids - HEAD_CENTER
    flipped = np.sum(normals * outward, axis=1) < 0
    normals[flipped] *= -1

    light_positions = _normalise_lights(lights)
    light_dirs = light_positions - HEAD_CENTER
    light_lengths = np.linalg.norm(light_dirs, axis=1)
    light_dirs = light_dirs[light_lengths > 0] / light_lengths[light_lengths > 0, None]
    if light_dirs.size == 0:
        light_dirs = DEFAULT_HEADPLOT_LIGHTS - HEAD_CENTER
        light_dirs = light_dirs / np.linalg.norm(light_dirs, axis=1)[:, None]

    diffuse = np.maximum(0.0, finite_matmul(normals, light_dirs.T))
    intensity = 0.78 + 0.28 * np.nanmax(diffuse, axis=1)
    intensity = np.clip(intensity, 0.75, 1.06)
    shaded = np.asarray(facecolors, dtype=float).copy()
    finite = np.isfinite(shaded[:, :3]).all(axis=1)
    shaded[finite, :3] = np.clip(shaded[finite, :3] * intensity[finite, None], 0.0, 1.0)
    return shaded


def _normalise_lights(value: Any) -> np.ndarray:
    if value is None:
        return DEFAULT_HEADPLOT_LIGHTS
    try:
        lights = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return DEFAULT_HEADPLOT_LIGHTS
    if lights.ndim == 1:
        if lights.size != 3:
            return DEFAULT_HEADPLOT_LIGHTS
        lights = lights.reshape(1, 3)
    if lights.ndim != 2 or lights.shape[1] != 3:
        return DEFAULT_HEADPLOT_LIGHTS
    return lights


def _plot_electrodes(ax: Any, spline: HeadplotSpline, *, labels: int = 0) -> None:
    electrodes = np.asarray(spline.new_electrodes, dtype=float)
    ax.scatter(electrodes[:, 0], electrodes[:, 1], electrodes[:, 2], color="black", s=14, depthshade=False)
    for index, point in enumerate(electrodes, start=1):
        ax.plot(
            [point[0], HEAD_CENTER[0]],
            [point[1], HEAD_CENTER[1]],
            [point[2], HEAD_CENTER[2]],
            color="black",
            linewidth=0.4,
        )
        if labels == 1:
            ax.text(point[0] * 1.04, point[1] * 1.04, point[2] * 1.04, str(index), fontsize=8)
        elif labels == 2 and index - 1 < len(spline.electrode_names):
            ax.text(point[0] * 1.04, point[1] * 1.04, point[2] * 1.04, spline.electrode_names[index - 1], fontsize=8)


def _set_view(ax: Any, value: Any) -> None:
    if isinstance(value, str):
        views = {
            "front": (-180, 30),
            "f": (-180, 30),
            "back": (0, 30),
            "b": (0, 30),
            "left": (-90, 30),
            "l": (-90, 30),
            "right": (90, 30),
            "r": (90, 30),
            "frontright": (135, 30),
            "fr": (135, 30),
            "backright": (45, 30),
            "br": (45, 30),
            "frontleft": (-135, 30),
            "fl": (-135, 30),
            "backleft": (-45, 30),
            "bl": (-45, 30),
            "top": (0, 90),
            "bottom": (0, -90),
        }
        try:
            azimuth, elevation = views[value.lower()]
        except KeyError as exc:
            raise ValueError(f"Invalid headplot view: {value}") from exc
    else:
        numeric = np.asarray(value, dtype=float).ravel()
        if numeric.size != 2:
            raise ValueError("headplot view must be a string or [azimuth elevation]")
        azimuth, elevation = float(numeric[0]), float(numeric[1])
    # MATLAB and Matplotlib use opposite azimuth conventions for the same
    # camera direction; keep the public option in EEGLAB/MATLAB coordinates.
    ax.view_init(elev=elevation, azim=180 - azimuth)


def _autoscale_3d(ax: Any, vertices: np.ndarray) -> None:
    center = np.nanmean(vertices, axis=0)
    span = np.nanmax(np.ptp(vertices, axis=0)) / 2 * HEADPLOT_VIEW_SCALE
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)


def _zero_based_faces(faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, dtype=int)
    if faces.size and np.min(faces) >= 1:
        faces = faces - 1
    return faces


def _is_on(value: str | bool) -> bool:
    return _is_on_value(value)


__all__ = [
    "HeadplotMesh",
    "HeadplotSpline",
    "MAPLIMIT_PADDING",
    "default_headplot_transform",
    "default_headplot_mesh_transform",
    "headplot",
    "headplot_setup",
    "load_headplot_mesh",
    "load_headplot_spline",
    "packaged_headplot_path",
]
