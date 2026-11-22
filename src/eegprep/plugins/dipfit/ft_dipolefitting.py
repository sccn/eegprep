"""Procedural FieldTrip EEG dipole localization in millimeters.

The four-shell series, scalp projection, solid-angle brain test, and inverse
fit follow FieldTrip as called by EEGLAB DIPFIT. Mesh indices are zero based.
"""

from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.io import loadmat
from scipy.optimize import minimize
from scipy.special import lpmv


_EPS = np.finfo(float).eps
_SOLID_ANGLE_TOLERANCE = 1000 * _EPS
_FUNCTION_TOLERANCE = 1e-9
_STEP_TOLERANCE = 1e-6
_TYPICAL_X_MM = 10.0
_ARMIJO = 0.01
_WOLFE_CURVATURE = 0.9
_LINE_SEARCH_ITERATIONS = 40


def standard_besa_headmodel() -> dict:
    """Return DIPFIT's standard BESA four-shell volume in mm."""
    return {
        "r": np.array([71.0, 72.0, 79.0, 85.0]),
        "cond": np.array([0.33, 1.0, 0.0042, 0.33]),
        "o": np.zeros(3),
        "unit": "mm",
        "type": "concentricspheres",
    }


@lru_cache(maxsize=4)
def load_bem_headmodel(hdmfile=None) -> dict:
    """Read an existing FieldTrip volume or the packaged standard MNI dipoli model."""
    path = Path(hdmfile or "standard_vol.npz")
    if not path.is_file() and path.name.lower() in ("standard_vol.mat", "standard_vol.npz"):
        path = Path(str(files("eegprep") / "resources/dipfit/standard_BEM/standard_vol.npz"))
    if not path.is_file():
        raise FileNotFoundError(f"DIPFIT head model {str(path)!r} not found")
    if path.suffix.lower() == ".npz":
        with np.load(path) as data:
            if "mat_data" not in data or "bnd0_pos" not in data:
                raise ValueError(f"{path.name} is not a boundary-element head model")
            vol = {
                "type": str(data["type"]),
                "cond": data["cond"],
                "unit": "mm",
                "mat": sparse.csr_matrix(
                    (data["mat_data"], data["mat_indices"], data["mat_indptr"]), shape=tuple(data["mat_shape"])
                ),
            }
            vol["bnd"] = [
                {"pos": data[f"bnd{i}_pos"].astype(float), "tri": data[f"bnd{i}_tri"].astype(int) - 1}
                for i in range(sum(key.endswith("_pos") for key in data.files))
            ]
    else:
        data = loadmat(path, simplify_cells=True)
        raw = data.get("vol", data.get("headmodel", {}))
        if not all(key in raw for key in ("bnd", "mat", "cond")):
            raise ValueError(f"{path.name} is not a boundary-element head model")
        boundaries = raw["bnd"] if isinstance(raw["bnd"], list) else [raw["bnd"]]
        vol = dict(raw)
        vol["bnd"] = [
            {"pos": np.asarray(b.get("pos", b.get("pnt")), dtype=float), "tri": np.asarray(b["tri"], dtype=int) - 1}
            for b in boundaries
        ]
    if vol["type"] == "openmeeg":
        raise ValueError("OpenMEEG head models are not implemented")
    if vol["type"] not in ("dipoli", "bem", "bemcp", "asa"):
        raise ValueError(f"{path.name} is not a boundary-element head model")
    vol["cond"] = np.atleast_1d(vol["cond"]).astype(float)
    return vol


def prepare_headmodel_sens(dipfit_cfg_dict: dict, comp_elec_dict: dict, chanlocs_list) -> tuple[dict, dict]:
    """Select and project sensors onto a sphere or scalp, returning mm dictionaries.

    BEM interpolation and average reference are folded into vol['mat'], as in
    ft_prepare_vol_sens. Its remaining sensor transfer is empty (identity).
    """
    cfg = dipfit_cfg_dict
    sens = dict(comp_elec_dict)
    labels = list(sens.get("label", []))
    if "channel" in cfg:
        selected = [labels.index(label) for label in cfg["channel"] if label in labels]
    elif "chansel" in cfg:
        selected = [labels.index(str(chanlocs_list[i - 1]["labels"])) for i in cfg["chansel"]]
    else:
        selected = list(range(len(sens["elecpos"])))
    sens["elecpos"] = np.asarray(sens["elecpos"], dtype=float)[selected].copy()
    sens["label"] = [labels[i] for i in selected] if labels else []
    supplied = cfg.get("vol", cfg.get("headmodel"))
    if isinstance(supplied, dict):
        vol = dict(supplied)
    elif Path(str(cfg.get("hdmfile", ""))).name.lower() == "standard_besa.mat":
        vol = standard_besa_headmodel()
    else:
        vol = dict(load_bem_headmodel(cfg.get("hdmfile")))
    if "r" in vol:
        vol["r"] = np.atleast_1d(vol["r"]).astype(float)
        vol["cond"] = np.atleast_1d(vol.get("cond", vol.get("c"))).astype(float)
        vol["o"] = np.asarray(vol.get("o", np.zeros(3)), dtype=float)
        vol["type"] = "concentricspheres" if len(vol["r"]) > 1 else "singlesphere"
        center = sens["elecpos"] - vol["o"]
        sens["elecpos"] = center * max(vol["r"]) / np.linalg.norm(center, axis=1)[:, None] + vol["o"]
        vol["source"] = int(np.argmin(vol["r"]))
    else:
        if vol.get("type") == "openmeeg":
            raise ValueError("OpenMEEG head models are not implemented")
        boundaries = vol["bnd"]
        nesting = [
            sum(surface_inside(b["pos"][:1], other["pos"], other["tri"])[0] for other in boundaries if other is not b)
            for b in boundaries
        ]
        vol["source"] = int(np.argmax(nesting))
        vol["skin_surface"] = int(np.argmin(nesting))
        skin = boundaries[vol["skin_surface"]]
        weights, sens["elecpos"] = project_elec(sens["elecpos"], skin["pos"], skin["tri"])
        transfer = transfer_elec(skin["pos"], skin["tri"], weights)
        matrix = vol["mat"]
        if matrix.shape[0] != len(skin["pos"]):
            transfer = np.hstack(
                [
                    transfer if i == vol["skin_surface"] else np.zeros((len(selected), len(b["pos"])))
                    for i, b in enumerate(boundaries)
                ]
            )
        matrix = transfer @ matrix
        vol["mat"] = matrix - np.mean(matrix, axis=0, keepdims=True)
        sens["tra"] = np.empty((0, 0))
    vol["unit"] = sens["unit"] = "mm"
    sens["chanpos"] = sens["elecpos"].copy()
    return vol, sens


def inside_headmodel(dippos, vol: dict) -> np.ndarray:
    """Test the strict inner sphere or the BEM brain mesh."""
    positions = np.asarray(dippos, dtype=float).reshape(-1, 3)
    if "r" in vol:
        return np.linalg.norm(positions - vol.get("o", np.zeros(3)), axis=1) < min(vol["r"])
    boundary = vol["bnd"][vol["source"]]
    return surface_inside(positions, boundary["pos"], boundary["tri"])


def inside_brain_grid(xgrid, ygrid, zgrid, vol: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return FieldTrip's tight ndgrid in column-major order and its brain mask."""
    axes = [np.asarray(values, dtype=float).ravel() for values in (xgrid, ygrid, zgrid)]
    mesh = np.meshgrid(*axes, indexing="ij")
    pos = np.column_stack([axis.ravel(order="F") for axis in mesh])
    inside = inside_headmodel(pos, vol)
    if not inside.any():
        return pos, inside
    mask = inside.reshape(tuple(len(axis) for axis in axes), order="F")
    keep = [np.any(mask, axis=tuple(j for j in range(3) if j != i)) for i in range(3)]
    selected = (keep[0][:, None, None] & keep[1][None, :, None] & keep[2][None, None, :]).ravel(order="F")
    return pos[selected], inside[selected]


def solid_angle(pnt: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Return the solid angle of each triangle as seen from the origin."""
    vertices = np.asarray(pnt, dtype=float)
    triangles = np.asarray(tri, dtype=int)
    r1 = vertices[triangles[:, 0]]
    r2 = vertices[triangles[:, 1]]
    r3 = vertices[triangles[:, 2]]
    normal = np.cross(r2, r3)
    numerator = np.einsum("ij,ij->i", normal, r1)
    norm1 = np.linalg.norm(r1, axis=1)
    norm2 = np.linalg.norm(r2, axis=1)
    norm3 = np.linalg.norm(r3, axis=1)
    ip12 = np.einsum("ij,ij->i", r1, r2)
    ip23 = np.einsum("ij,ij->i", r2, r3)
    ip13 = np.einsum("ij,ij->i", r1, r3)
    denominator = norm1 * norm2 * norm3 + ip12 * norm3 + ip23 * norm1 + ip13 * norm2
    angle = 2.0 * np.arctan2(numerator, denominator)
    angle[(numerator == 0.0) & (denominator <= 0.0)] = np.nan
    return angle


def _pinv_n_by_2(matrix: np.ndarray) -> np.ndarray:
    """Return the Moore-Penrose inverse of each ``M x N x 2`` slice as ``M x 2 x N``."""
    gram = np.einsum("mni,mnj->mij", matrix, matrix)
    determinant = gram[:, 0, 0] * gram[:, 1, 1] - gram[:, 0, 1] * gram[:, 1, 0]
    inverse = np.empty_like(gram)
    inverse[:, 0, 0] = gram[:, 1, 1] / determinant
    inverse[:, 0, 1] = -gram[:, 0, 1] / determinant
    inverse[:, 1, 0] = -gram[:, 1, 0] / determinant
    inverse[:, 1, 1] = gram[:, 0, 0] / determinant
    transposed = np.transpose(matrix, (0, 2, 1))
    return np.einsum("mij,mjk->mik", inverse, transposed)


DEFAULT_SERIES_ORDER = 60


def compute_dipole_leadfield(dippos_Mx3, sens: dict, vol: dict) -> np.ndarray:
    """Return an average-referenced N-channel by (3 * M)-dipole EEG leadfield."""
    positions = np.asarray(dippos_Mx3, dtype=float).reshape(-1, 3)
    electrodes = np.asarray(sens["elecpos"], dtype=float).reshape(-1, 3)
    if "r" in vol:
        order = np.argsort(vol["r"], kind="stable")
        radii = np.asarray(vol["r"])[order]
        cond = np.asarray(vol.get("cond", vol.get("c")))[order]
        origin = np.asarray(vol.get("o", np.zeros(3)))
        electrodes = electrodes - origin
        positions = positions - origin
        if len(radii) == 1:
            model = {"r": radii, "cond": cond}
            forward = eeg_leadfield1
        else:
            model = {
                "r": radii[np.minimum(np.arange(4), len(radii) - 1)],
                "cond": cond[np.minimum(np.arange(4), len(cond) - 1)],
            }
            model["t"] = vol["t"] if "t" in vol else eeg_leadfield4_prepare(model)
            forward = eeg_leadfield4
        fields = [forward(position, electrodes, model) for position in positions]
        lf = np.column_stack(fields) if fields else np.zeros((len(electrodes), 0))
    else:
        if not vol.get("bnd"):
            raise ValueError("there are no compartment boundaries present")
        lf = np.empty((vol["mat"].shape[0], 3 * len(positions)))
        for start in range(0, len(positions), 128):
            lf[:, 3 * start : 3 * (start + 128)] = eeg_leadfieldb(positions[start : start + 128], electrodes, vol)
    if np.size(sens.get("tra", [])):
        return np.asarray(sens["tra"]) @ lf
    return lf - np.mean(lf, axis=0, keepdims=True)


def eeg_leadfield4_prepare(vol: dict[str, Any], order: int | None = None) -> np.ndarray:
    """Return the shell coefficients ``t`` for the four-sphere series.

    Args:
        vol: Mapping with ``r`` (4 radii) and ``cond`` (4 conductivities),
            smallest sphere first after sorting.
        order: Number of series terms. FieldTrip's default is 60.

    Returns:
        Length-``order`` array of Lutkenhoner coefficients.
    """
    radii, cond = _sorted_shells(vol)
    r1, r2, r3, r4 = (float(value) for value in radii)
    if order is None:
        order = DEFAULT_SERIES_ORDER
    if order < 1:
        raise ValueError("series order must be positive")
    k1 = float(cond[0] / cond[1])
    k2 = float(cond[1] / cond[2])
    k3 = float(cond[2] / cond[3])
    terms = np.empty(order, dtype=float)
    for index, degree in enumerate(range(1, order + 1)):
        n = float(degree)
        two_n = 2.0 * n + 1.0
        r12 = (r1 / r2) ** two_n
        r13 = (r1 / r3) ** two_n
        r23 = (r2 / r3) ** two_n
        r34 = (r3 / r4) ** two_n
        terms[index] = ((n * k1 + n + 1.0) * (n * k2 + n + 1.0) + n * (n + 1.0) * (k1 - 1.0) * (k2 - 1.0) * r12) * (
            (n * k3 + n + 1.0) + (n + 1.0) * (k3 - 1.0) * r34
        ) + ((k1 - 1.0) * ((n + 1.0) * k2 + n) * r13 + (n * k1 + n + 1.0) * (k2 - 1.0) * r23) * (n + 1.0) * (
            n * (k3 - 1.0) + ((n + 1.0) * k3 + n) * r34
        )
    return terms


def eeg_leadfield4(
    dipole: np.ndarray,
    electrodes: np.ndarray,
    vol: dict[str, Any],
    order: int | None = None,
) -> np.ndarray:
    """Return the electric leadfield of one dipole in four concentric spheres.

    Args:
        dipole: Dipole position, shape ``(3,)``. The sphere center is the origin.
        electrodes: Electrode positions, shape ``(n_electrodes, 3)``.
        vol: Radii ``r``, conductivities ``cond``, and optional series
            coefficients ``t`` from :func:`eeg_leadfield4_prepare`.
        order: Series length used when ``vol`` has no ``t``. Default 60.

    Returns:
        Leadfield of shape ``(n_electrodes, 3)`` for the x, y, and z moments.
        Potentials are not average-referenced; ``ft_compute_leadfield`` does that.
    """
    position = np.asarray(dipole, dtype=float).reshape(3)
    sensors = np.asarray(electrodes, dtype=float).reshape(-1, 3).copy()
    radii, cond = _sorted_shells(vol)
    r4 = float(radii[3])
    c4 = float(cond[3])
    distance = np.linalg.norm(sensors, axis=1)
    sensors = r4 * sensors / distance[:, np.newaxis]
    sensors, rotation, reflect_z = _rotate_dipole_to_positive_z(position, sensors)
    coefficients = vol.get("t")
    if coefficients is None:
        coefficients = eeg_leadfield4_prepare({"r": radii, "cond": cond}, order)
    else:
        coefficients = np.asarray(coefficients, dtype=float).ravel()
    degree = np.arange(1, coefficients.size + 1, dtype=float)
    eccentricity = float(np.linalg.norm(position) / r4)
    const = (2.0 * degree + 1.0) ** 4 * eccentricity ** (degree - 1.0)
    const = const / (coefficients * 4.0 * np.pi * c4 * r4**2)
    azimuth, elevation = _cart2sph(sensors)
    cos_theta = np.clip(np.sin(elevation), -1.0, 1.0)
    tangential = np.zeros(sensors.shape[0], dtype=float)
    radial = np.zeros(sensors.shape[0], dtype=float)
    for index, n_value in enumerate(degree.astype(int)):
        tangential += const[index] * lpmv(1, n_value, cos_theta) / n_value
        radial += const[index] * lpmv(0, n_value, cos_theta)
    leadfield = np.column_stack(
        (
            -np.cos(azimuth) * tangential,
            -np.sin(azimuth) * tangential,
            radial,
        )
    )
    if rotation is not None:
        leadfield = leadfield @ rotation
    elif reflect_z:
        leadfield[:, 2] *= -1.0
    return leadfield


def _sorted_shells(vol: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    radii = np.asarray(vol["r"], dtype=float).ravel()
    cond = np.asarray(vol.get("cond", vol.get("c")), dtype=float).ravel()
    if radii.size != 4 or cond.size != 4:
        raise ValueError("eeg_leadfield4 requires 4 radii and 4 conductivities")
    order = np.argsort(radii, kind="stable")
    return radii[order], cond[order]


def _rotate_dipole_to_positive_z(
    dipole: np.ndarray, electrodes: np.ndarray
) -> tuple[np.ndarray, np.ndarray | None, bool]:
    x_coord, y_coord, z_coord = (float(value) for value in dipole)
    if x_coord != 0.0 or y_coord != 0.0:
        radius = float(np.linalg.norm(dipole))
        axial = float(np.linalg.norm(dipole[:2]))
        rotation = np.array(
            [
                [
                    x_coord * z_coord / (radius * axial),
                    y_coord * z_coord / (radius * axial),
                    -axial / radius,
                ],
                [-y_coord / axial, x_coord / axial, 0.0],
                dipole / radius,
            ],
            dtype=float,
        )
        return electrodes @ rotation.T, rotation, False
    if z_coord < 0.0:
        reflected = electrodes.copy()
        reflected[:, 2] *= -1.0
        return reflected, None, True
    return electrodes, None, False


def _cart2sph(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    elevation = np.arctan2(points[:, 2], np.hypot(points[:, 0], points[:, 1]))
    return azimuth, elevation


def eeg_leadfield1(dipole: np.ndarray, electrodes: np.ndarray, vol: dict[str, Any]) -> np.ndarray:
    """Return the electric leadfield of one dipole in a homogeneous sphere.

    Args:
        dipole: Dipole position, shape ``(3,)``.
        electrodes: Electrode positions, shape ``(n_electrodes, 3)``.
        vol: Sphere radius ``r`` and conductivity ``cond``. When several
            shells are present, the outermost radius and its conductivity
            are used, matching FieldTrip.

    Returns:
        Leadfield of shape ``(n_electrodes, 3)``.
    """
    position = np.asarray(dipole, dtype=float).reshape(3)
    sensors = np.asarray(electrodes, dtype=float).reshape(-1, 3).copy()
    radii = np.asarray(vol["r"], dtype=float).ravel()
    cond = np.asarray(vol.get("cond", vol.get("c")), dtype=float).ravel()
    outer = int(np.argmax(radii))
    radius = float(radii[outer])
    conductivity = float(cond[outer])
    distance = np.linalg.norm(sensors, axis=1)
    sensors = radius * sensors / distance[:, np.newaxis]
    dipole_radius = float(np.linalg.norm(position))
    scale = 4.0 * np.pi * dipole_radius**2 * conductivity
    if dipole_radius == 0.0:
        azimuth = np.arctan2(sensors[:, 1], sensors[:, 0])
        elevation = np.arctan2(sensors[:, 2], np.hypot(sensors[:, 0], sensors[:, 1]))
        theta = np.pi / 2.0 - elevation
        leadfield = np.column_stack(
            (
                np.sin(theta) * np.cos(azimuth),
                np.sin(theta) * np.sin(azimuth),
                np.cos(theta),
            )
        )
        return 3.0 / (radius**2 * 4.0 * np.pi * conductivity) * leadfield

    leadfield = np.zeros((sensors.shape[0], 3), dtype=float)
    for index, sensor in enumerate(sensors):
        separation = sensor - position
        separation_norm = float(np.linalg.norm(separation))
        tangential_scale = radius**2 * dipole_radius**2 - float(np.dot(sensor, position)) ** 2
        tangential_vector = dipole_radius**2 * sensor - float(np.dot(sensor, position)) * position
        radial_gain = (radius**2 - dipole_radius**2) / separation_norm**3 - 1.0 / radius
        if tangential_scale < 1000.0 * _EPS:
            numerator = 2.0 / separation_norm**3 * tangential_vector + radial_gain * position
        else:
            direction = sensor / radius - separation / separation_norm
            numerator = (float(np.dot(position, direction)) / tangential_scale + 2.0 / separation_norm**3) * (
                tangential_vector
            ) + radial_gain * position
        leadfield[index] = numerator / scale
    return leadfield


def eeg_infinite_dipole(dippos, elc, vol) -> np.ndarray:
    """Return the infinite-medium EEG leadfield for a conductivity or volume."""
    conductivity = float(np.asarray(vol["cond"] if isinstance(vol, dict) else vol).ravel()[0])
    positions = np.asarray(dippos, dtype=float).reshape(-1, 3)
    electrodes = np.asarray(elc, dtype=float).reshape(-1, 3)
    relative = electrodes[None, :, :] - positions[:, None, :]
    scale = (4 * np.pi * conductivity) * np.sum(relative**2, axis=2) ** 1.5
    return (relative / scale[:, :, None]).transpose(1, 0, 2).reshape(len(electrodes), -1)


def eeg_leadfieldb(dippos, elc, vol: dict) -> np.ndarray:
    """Apply a prepared dipoli, ASA, or bemcp system matrix."""
    boundaries = vol.get("bnd", [])
    if not boundaries:
        raise ValueError("there are no compartment boundaries present")
    kind = vol.get("type", "dipoli")
    vertices = np.vstack([boundary["pos"] for boundary in boundaries])
    if kind in ("dipoli", "bem"):
        field = eeg_infinite_dipole(dippos, vertices, 1.0)
    elif kind == "asa":
        field = eeg_infinite_dipole(dippos, vertices, vol["cond"][vol["source"]])
    elif kind == "bemcp":
        cond = np.r_[vol["cond"], 0.0]
        field = np.vstack(
            [eeg_infinite_dipole(dippos, b["pos"], (cond[i] + cond[i + 1]) / 2) for i, b in enumerate(boundaries)]
        )
    else:
        raise ValueError(f"Unsupported BEM type {kind!r}")
    return vol["mat"] @ field


def ft_dipolefitting(cfg: dict, comp: dict) -> dict:
    """Fit moving ICA topographies using FieldTrip configuration dictionaries.

    Components are 1-based. cfg selects channels, a headmodel, grid axes or
    an initial dip['pos'], and optional bilateral symmetry ('x', 'y', 'z').
    """
    labels = comp["topolabel"]
    channels = [label for label in cfg.get("channel", labels) if label in labels]
    if not channels:
        raise ValueError("No DIPFIT channels overlap with ICA channel indices")
    prepare_cfg = dict(cfg, channel=channels)
    vol, sens = prepare_headmodel_sens(prepare_cfg, comp["elec"], [])
    components = np.asarray(cfg.get("component", np.arange(1, comp["topo"].shape[1] + 1)), dtype=int).ravel()
    # Reference all maps before selection so batching cannot change rounding.
    data = comp["topo"][[labels.index(label) for label in channels], :]
    data = data - np.mean(data, axis=0, keepdims=True)
    data = data[:, components - 1]
    if cfg.get("gridsearch", "yes") == "yes":
        pos, inside = inside_brain_grid(cfg["xgrid"], cfg["ygrid"], cfg["zgrid"], vol)
        candidates = pos[inside]
        if not len(candidates):
            raise ValueError("grid does not contain any points inside the brain compartment")
        leadfield = compute_dipole_leadfield(candidates, sens, vol)
        fields = leadfield.reshape(len(channels), -1, 3).transpose(1, 0, 2)
        inverse = np.linalg.pinv(fields)
        residual = data[None, :, :] - fields @ (inverse @ data)
        best = np.argmin(np.sum(residual**2, axis=1), axis=0)
        positions = [candidates[index : index + 1] for index in best]
    else:
        initial = np.asarray(cfg["dip"]["pos"], dtype=float).reshape(-1, 3)
        positions = [initial.copy() for _ in components]
    dips = []
    modeled = np.empty_like(data)
    for index, position in enumerate(positions):
        if cfg.get("nonlinear", "yes") == "yes":
            position = _refine_positions(position, data[:, index], sens, vol, cfg)
        field = compute_dipole_leadfield(position, sens, vol)
        moment = np.linalg.pinv(field) @ data[:, index]
        modeled[:, index] = field @ moment
        power = np.sum(data[:, index] ** 2)
        rv = np.sum((data[:, index] - modeled[:, index]) ** 2) / power if power else 1.0
        dips.append({"pos": position, "mom": moment, "rv": float(rv)})
    return {"dip": dips, "Vdata": data, "Vmodel": modeled, "label": channels}


def _cubic(a, fa, ga, b, fb, gb):
    if a == b:
        return a
    slope = (fa - fb) / (a - b)
    z = ga + gb - 3 * slope
    rad = z * z - ga * gb
    if rad < 0:
        return (a + b) / 2
    w = np.sign(b - a) * np.sqrt(rad)
    denominator = gb - ga + 2 * w
    if denominator == 0:
        return (a + b) / 2
    return b - (b - a) * (gb + w - z) / denominator


def _section_trial(trial, anchor, other):
    """Clip a sectioning trial to 10% from ``anchor`` and 50% from ``other``."""
    low, high = sorted((anchor + 0.1 * (other - anchor), other - 0.5 * (other - anchor)))
    return float(min(max(trial, low), high))


def _line_search(fun, grad, x, p, g, f, first_step=1.0):
    """Strong Wolfe search. ``a`` is the last step that met sufficient decrease."""
    derivative = g @ p
    a, fa, ga = 0.0, f, derivative
    b = first_step
    last = (0.0, f, g)
    positive_slope = False
    for i in range(_LINE_SEARCH_ITERATIONS):
        fb = fun(x + b * p)
        gbvec = grad(x + b * p, fb)
        gb = gbvec @ p
        last = (b, fb, gbvec)
        if fb > f + _ARMIJO * b * derivative or (i and fb >= fa):
            break
        if abs(gb) <= -_WOLFE_CURVATURE * derivative:
            return b, fb, gbvec
        if gb >= 0:
            positive_slope = True
            break
        # MATLAB extrapolates between one and nine previous interval lengths.
        new = _cubic(a, fa, ga, b, fb, gb)
        new = np.clip(new, b + (b - a), b + 9 * (b - a)) if new > b else b + 9 * (b - a)
        a, fa, ga = b, fb, gb
        b = new
    if positive_slope:
        a, fa, ga, b, fb, gb = b, fb, gb, a, fa, ga
    roundoff = np.spacing(max(1.0, abs(f)))
    for _ in range(_LINE_SEARCH_ITERATIONS):
        c = _section_trial(_cubic(a, fa, ga, b, fb, gb), a, b)
        if (a - c) * ga <= roundoff:
            return c, last[1], last[2]
        fc = fun(x + c * p)
        gcvec = grad(x + c * p, fc)
        gc = gcvec @ p
        last = (c, fc, gcvec)
        if fc > f + _ARMIJO * c * derivative or fc >= fa:
            b, fb, gb = c, fc, gc
        else:
            if abs(gc) <= -_WOLFE_CURVATURE * derivative:
                return c, fc, gcvec
            if gc * (b - a) >= 0:
                b, fb, gb = a, fa, ga
            a, fa, ga = c, fc, gc
    return last


def _fminunc(fun, start, maxiter=1000):
    """FieldTrip dipole ``fminunc``: TypicalX of 10 mm, scaled BFGS, Wolfe search.

    The forward difference is ``sqrt(eps) * sign'(x) * max(|x|, 10)``, with
    ``sign'(0) = 1``. The inverse Hessian starts as the identity, is scaled by
    ``(s'y)/(y'y)`` on the first accepted step, and takes the BFGS update only
    when ``s'y`` clears the curvature guard. Iteration stops when
    ``||g||_inf < TolFun * (1 + ||g0||_inf)`` or
    ``||dx / (1 + |x|)||_inf < TolX``.
    """

    def grad(point, known):
        step = np.sqrt(_EPS) * np.where(point < 0, -1.0, 1.0) * np.maximum(np.abs(point), _TYPICAL_X_MM)
        return np.array([(fun(point + np.eye(len(point))[i] * step[i]) - known) / step[i] for i in range(len(point))])

    values = np.array(start, dtype=float).ravel()
    value = fun(values)
    gradient = grad(values, value)
    initial_gradient = np.linalg.norm(gradient, np.inf)
    inverse = np.eye(len(values))
    for iteration in range(maxiter):
        if np.linalg.norm(gradient, np.inf) < _FUNCTION_TOLERANCE * (1 + initial_gradient):
            break
        direction = -inverse @ gradient
        first_step = min(1 / initial_gradient, 1.0) if iteration == 0 else 1.0
        alpha, new_value, new_gradient = _line_search(fun, grad, values, direction, gradient, value, first_step)
        if alpha is None or new_value >= value:
            break
        delta = alpha * direction
        values = values + delta
        delta_gradient = new_gradient - gradient
        value, gradient = new_value, new_gradient
        if np.linalg.norm(delta / (1 + np.abs(values)), np.inf) < _STEP_TOLERANCE:
            break
        curvature = delta @ delta_gradient
        if curvature >= np.sqrt(_EPS) * max(_EPS, np.linalg.norm(delta) * np.linalg.norm(delta_gradient)):
            if iteration == 0:
                inverse = curvature / (delta_gradient @ delta_gradient) * np.eye(len(values))
            projected = inverse @ delta_gradient
            inverse += (1 + delta_gradient @ projected / curvature) * np.outer(delta, delta) / curvature - (
                np.outer(delta, projected) + np.outer(projected, delta)
            ) / curvature
    return values


class _EvaluationLimit(Exception):
    pass


def _fminunc_coregister(
    fun,
    start,
    maxiter=1000,
    tol_fun=_FUNCTION_TOLERANCE,
    tol_x=_STEP_TOLERANCE,
    max_fun_evals=None,
):
    """Quasi-Newton BFGS used by the traditional electrode warp.

    ``ft_warp_optim`` leaves ``TypicalX`` unset. This forward difference is
    ``sqrt(eps) * sign'(x)``, with ``sign'(0) = 1``, which is the step that
    matches EEGLAB's traditional warp. TolFun stops on the relative
    infinity-norm of the gradient. TolX stops on the relative step. The
    default evaluation budget is ``100 * n``.
    """
    x = np.asarray(start, dtype=float).ravel().astype(float)
    if max_fun_evals is None:
        max_fun_evals = 100 * x.size
    evaluations = 0

    def counted(values):
        nonlocal evaluations
        evaluations += 1
        if evaluations > max_fun_evals:
            raise _EvaluationLimit
        return float(fun(np.asarray(values, dtype=float)))

    def grad(point):
        value = counted(point)
        step = np.sqrt(_EPS) * np.where(point < 0, -1.0, 1.0)
        slope = np.empty(point.size)
        for index in range(point.size):
            trial = point.copy()
            trial[index] += step[index]
            slope[index] = (counted(trial) - value) / step[index]
        return value, slope

    try:
        f, g = grad(x)
        g0 = np.linalg.norm(g, np.inf)
        inverse = np.eye(x.size)
        for iteration in range(maxiter):
            if np.linalg.norm(g, np.inf) <= tol_fun * max(1.0, g0):
                break
            direction = -inverse @ g
            if g @ direction >= 0:
                inverse = np.eye(x.size)
                direction = -g
            first_step = min(1.0 / g0, 1.0) if iteration == 0 and g0 > 0 else 1.0
            alpha, nf, ng = _line_search(counted, lambda point, known: grad(point)[1], x, direction, g, f, first_step)
            if alpha is None or nf >= f:
                break
            dx = alpha * direction
            x = x + dx
            dg = ng - g
            f, g = nf, ng
            if np.linalg.norm(dx / (1.0 + np.abs(x)), np.inf) < tol_x:
                break
            curvature = float(dx @ dg)
            if curvature >= np.sqrt(_EPS) * max(_EPS, np.linalg.norm(dx) * np.linalg.norm(dg)):
                if iteration == 0:
                    inverse = curvature / float(dg @ dg) * np.eye(x.size)
                vector = inverse @ dg
                inverse = (
                    inverse
                    + (1.0 + float(dg @ vector) / curvature) * np.outer(dx, dx) / curvature
                    - (np.outer(dx, vector) + np.outer(vector, dx)) / curvature
                )
    except _EvaluationLimit:
        pass
    return x


def _refine_positions(initial, data, sens, vol, cfg):
    axis = cfg.get("symmetry")
    if axis not in (None, "", "x", "y", "z"):
        raise ValueError("symmetry must be 'x', 'y', or 'z'")
    if axis and len(initial) != 2:
        raise ValueError("symmetry requires two dipoles")

    def unpack(values):
        if not axis:
            return values.reshape(-1, 3)
        mirror = values.copy()
        mirror["xyz".index(axis)] *= -1
        return np.vstack([values, mirror])

    start = initial[0].copy() if axis else initial.ravel().copy()
    power = data @ data
    if power == 0:
        return unpack(start)
    options = cfg.get("dipfit", {})

    def objective(values):
        pos = unpack(values)
        if options.get("checkinside", False) and not inside_headmodel(pos, vol).all():
            return 1.0
        field = compute_dipole_leadfield(pos, sens, vol)
        if not np.isfinite(field).all():
            return np.inf
        residual = data - field @ (np.linalg.pinv(field) @ data)
        return float((residual @ residual) / power)

    method = options.get("optimfun", "fminunc")
    maxiter = int(options.get("maxiter", 3000 if method == "fminsearch" else 1000))
    if maxiter < 1:
        raise ValueError("maxiter must be positive")
    if method == "fminsearch":
        result = minimize(
            objective,
            start,
            method="Nelder-Mead",
            options={
                "maxiter": maxiter,
                "maxfev": 2 * maxiter * start.size,
                "xatol": 1e-4,
                "fatol": 1e-4,
                "adaptive": False,
            },
        )
        return unpack(result.x)
    if method != "fminunc":
        raise ValueError("optimfun must be 'fminunc' or 'fminsearch'")
    return unpack(_fminunc(objective, start, maxiter))


def surface_inside(dippos, pos, tri) -> np.ndarray:
    """Test a closed brain mesh using triangle solid angles, with its boundary outside."""
    points = np.asarray(dippos, dtype=float).reshape(-1, 3)
    vertices = np.asarray(pos, dtype=float)
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    inside = np.zeros(len(points), dtype=bool)
    for index, point in enumerate(points):
        if np.all(point >= lower) and np.all(point <= upper):
            total = np.sum(solid_angle(vertices - point, tri))
            inside[index] = abs(abs(total) - 4 * np.pi) < _SOLID_ANGLE_TOLERANCE
    return inside


def project_elec(elc, pnt, tri) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest scalp projections and [triangle, lambda, mu, distance] weights."""
    electrodes = np.asarray(elc, dtype=float).reshape(-1, 3)
    vertices = np.asarray(pnt, dtype=float)
    triangles = np.asarray(tri, dtype=int)
    a, b, c = vertices[triangles].transpose(1, 0, 2)
    u, v = b - a, c - a
    pseudo = _pinv_n_by_2(np.stack([u, v], axis=2))
    weights = np.empty((len(electrodes), 4))
    projected = np.empty_like(electrodes)
    for index, electrode in enumerate(electrodes):
        la = np.einsum("ij,ij->i", electrode - a, pseudo[:, 0, :])
        mu = np.einsum("ij,ij->i", electrode - a, pseudo[:, 1, :])
        closest = a + la[:, None] * u + mu[:, None] * v
        for mask, start, stop in (
            (la < 0, a, c),
            ((mu < 0) & (la >= 0), a, b),
            ((la + mu > 1) & (mu >= 0) & (la >= 0), b, c),
        ):
            segment = stop[mask] - start[mask]
            parameter = np.sum((electrode - start[mask]) * segment, axis=1) / np.sum(segment**2, axis=1)
            closest[mask] = start[mask] + np.clip(parameter, 0.0, 1.0)[:, None] * segment
        distance = np.linalg.norm(electrode - closest, axis=1)
        nearest = int(np.argmin(distance))
        la, mu = np.linalg.lstsq(np.column_stack([u[nearest], v[nearest]]), closest[nearest] - a[nearest], rcond=None)[
            0
        ]
        weights[index] = [nearest, la, mu, distance[nearest]]
        projected[index] = (1 - la - mu) * a[nearest] + la * b[nearest] + mu * c[nearest]
    return weights, projected


def transfer_elec(pnt, tri, el) -> np.ndarray:
    """Return the barycentric scalp vertex-to-electrode transfer matrix."""
    transfer = np.zeros((len(el), len(pnt)))
    for index, (triangle, la, mu, _distance) in enumerate(el):
        transfer[index, np.asarray(tri)[int(triangle)]] = [1 - la - mu, la, mu]
    return transfer
