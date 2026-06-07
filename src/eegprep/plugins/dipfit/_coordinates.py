"""Standalone DIPFIT coordinate, transform, and electrode-alignment helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize

from eegprep.plugins.dipfit._utils import DIPFITUnavailableError


_WARP_LEVELS = {
    "rigidbody": 1,
    "globalrescale": 2,
    "traditional": 3,
    "nonlin1": 4,
    "nonlin2": 5,
    "nonlin3": 6,
    "nonlin4": 7,
    "nonlin5": 8,
}


def mni2tal(points: Any) -> np.ndarray:
    """Convert MNI coordinates to approximate Talairach coordinates.

    This is a NumPy port of the Matthew Brett transform bundled with DIPFIT:
    a pitch correction plus different Z scalings above and below AC.
    """
    coordinates, transposed = _as_points(points, allow_transposed=True)
    matrix = mni2tal_matrix()
    homogeneous = np.column_stack([coordinates, np.ones(coordinates.shape[0])])
    below_ac = homogeneous[:, 2] < 0
    out = np.empty_like(homogeneous)
    out[below_ac] = homogeneous[below_ac] @ (matrix["rotn"] @ matrix["downZ"]).T
    out[~below_ac] = homogeneous[~below_ac] @ (matrix["rotn"] @ matrix["upZ"]).T
    result = out[:, :3]
    return result.T if transposed else result


def mni2tal_matrix() -> dict[str, np.ndarray]:
    """Return the Brett MNI-to-Talairach transform matrices."""
    return {
        "rotn": np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.9988, 0.0500, 0.0],
                [0.0, -0.0500, 0.9988, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        "upZ": np.diag([0.99, 0.97, 0.92, 1.0]),
        "downZ": np.diag([0.99, 0.97, 0.84, 1.0]),
    }


def sph2spm() -> np.ndarray:
    """Return DIPFIT's BESA spherical-to-SPM/MNI homogeneous transform."""
    return np.asarray(
        [
            [0.0101, -0.9400, 0.0, 0.5588],
            [1.1889, 0.0080, 0.0530, -18.0041],
            [-0.0005, -0.0000, 1.1268, 1.8045],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def headcoordinates(nasion: Any, lpa: Any, rpa: Any, flag: int = 0) -> np.ndarray:
    """Return the homogeneous transform into CTF, ASA, or FTG head coordinates."""
    nas = _as_vector3(nasion, "nasion")
    left = _as_vector3(lpa, "lpa")
    right = _as_vector3(rpa, "rpa")
    if flag == 0:
        origin = (left + right) / 2.0
        dirx = _unit(nas - origin, "nasion-origin")
        dirz = _unit(np.cross(dirx, left - right), "fiducial plane")
        diry = np.cross(dirz, dirx)
    elif flag == 1:
        dirz = _unit(np.cross(nas - right, left - right), "fiducial plane")
        diry = _unit(left - right, "left-right")
        dirx = _unit(np.cross(diry, dirz), "asa x-axis")
        origin = right + np.dot(nas - right, diry) * diry
    elif flag == 2:
        origin = nas
        dirx = _unit(left - origin, "pt1-pt2")
        diry_seed = right - origin
        dirz = _unit(np.cross(dirx, diry_seed), "ftg plane")
        diry = np.cross(dirz, dirx)
    else:
        raise ValueError("flag must be 0 (CTF), 1 (ASA), or 2 (FTG)")
    rotation = np.eye(4)
    rotation[:3, :3] = np.asarray([dirx, diry, dirz], dtype=float)
    translation = np.eye(4)
    translation[:3, 3] = -origin
    return rotation @ translation


def translate(offset: Any) -> np.ndarray:
    """Return a homogeneous translation matrix."""
    values = _as_vector3(offset, "translation")
    matrix = np.eye(4)
    matrix[:3, 3] = values
    return matrix


def rotate(angles_degrees: Any) -> np.ndarray:
    """Return FieldTrip/DIPFIT's degree-based z-y-x homogeneous rotation."""
    rx, ry, rz = np.deg2rad(_as_vector3(angles_degrees, "rotation"))
    cx, cy, cz = np.cos([rx, ry, rz])
    sx, sy, sz = np.sin([rx, ry, rz])
    return np.asarray(
        [
            [cz * cy, -sz * cy, sy, 0.0],
            [cz * sy * sx + sz * cx, -sz * sy * sx + cz * cx, -cy * sx, 0.0],
            [-cz * sy * cx + sz * sx, sz * sy * cx + cz * sx, cy * cx, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def scale(factors: Any) -> np.ndarray:
    """Return a homogeneous axis-scaling matrix."""
    values = _as_vector3(factors, "scale")
    return np.diag([values[0], values[1], values[2], 1.0])


def rigidbody(params: Any) -> np.ndarray:
    """Return the six-parameter rigid-body transform used by FieldTrip helpers."""
    values = _as_numeric_vector(params, 6, "rigidbody")
    return translate(values[:3]) @ rotate(values[3:6])


def globalrescale(params: Any) -> np.ndarray:
    """Return the seven-parameter rigid-body plus global scale transform."""
    values = _as_numeric_vector(params, 7, "globalrescale")
    return translate(values[:3]) @ rotate(values[3:6]) @ scale([values[6], values[6], values[6]])


def traditional(params: Any) -> np.ndarray:
    """Return the FieldTrip private traditional transform with degree rotations."""
    values = _as_numeric_vector(params, 9, "traditional")
    return translate(values[:3]) @ rotate(values[3:6]) @ scale(values[6:9])


def traditionaldipfit(params: Any) -> np.ndarray:
    """Return DIPFIT's nine-parameter transform with radian rotations."""
    values = _as_numeric_vector(params, None, "traditionaldipfit")
    if values.size == 6:
        values = np.concatenate([values, np.ones(3)])
    if values.size != 9:
        raise ValueError("traditionaldipfit parameters must contain 6 or 9 values")
    tx, ty, tz, rx, ry, rz, sx, sy, sz = values
    cx, cy, cz = np.cos([rx, ry, rz])
    sx_sin, sy_sin, sz_sin = np.sin([rx, ry, rz])
    rotation = np.eye(4)
    rotation[:3, :3] = [
        [cz * cy + sz_sin * sx_sin * sy_sin, sz_sin * cy + cz * sx_sin * sy_sin, cx * sy_sin],
        [-sz_sin * cx, cz * cx, sx_sin],
        [sz_sin * sx_sin * cy - cz * sy_sin, -cz * sx_sin * cy - sz_sin * sy_sin, cx * cy],
    ]
    scaling = np.diag([sx, sy, sz, 1.0])
    return translate([tx, ty, tz]) @ rotation @ scaling


def homogenous2traditional(matrix: Any) -> np.ndarray:
    """Estimate DIPFIT traditional parameters from a homogeneous transform."""
    transform = np.asarray(matrix, dtype=float)
    if transform.shape != (4, 4) or not np.allclose(transform[3], [0, 0, 0, 1]):
        raise ValueError("matrix must be a 4x4 homogeneous transform")
    tx, ty, tz = transform[:3, 3]
    unshifted = np.linalg.inv(translate([tx, ty, tz])) @ transform
    scales = np.linalg.norm(unshifted[:3, :3], axis=0)
    if np.any(scales == 0):
        raise ValueError("matrix contains a zero scale axis")
    rotation = unshifted @ np.linalg.inv(scale(scales))
    probe = rotation @ np.asarray([0.0, 0.0, 1.0, 0.0])
    ry = np.arcsin(np.clip(probe[0], -1.0, 1.0))
    rx = -np.arctan2(probe[1], probe[2])
    rx_matrix = _dipfit_axis_rotation([rx, 0.0, 0.0])
    ry_matrix = _dipfit_axis_rotation([0.0, ry, 0.0])
    rz_matrix = np.linalg.inv(ry_matrix) @ np.linalg.inv(rx_matrix) @ rotation
    z_probe = rz_matrix @ np.asarray([1.0, 0.0, 0.0, 0.0])
    rz = np.arcsin(np.clip(z_probe[1], -1.0, 1.0))
    return np.asarray([tx, ty, tz, rx, ry, rz, *scales], dtype=float)


def warp_apply(params: Any, points: Any, method: str | None = None) -> np.ndarray:
    """Apply a DIPFIT/FieldTrip linear or polynomial warp to points."""
    coordinates, _ = _as_points(points)
    matrix = np.asarray(params, dtype=float)
    selected = method.lower() if isinstance(method, str) else None
    if selected is None and matrix.shape == (4, 4):
        selected = "homogeneous"
    if selected in {"homogeneous", "homogenous"}:
        return apply_transform(matrix, coordinates)
    if selected in {None, "nonlinear", "nonlin0", "nonlin1", "nonlin2", "nonlin3", "nonlin4", "nonlin5"}:
        return _apply_polynomial_warp(matrix, coordinates, selected)
    if selected == "rigidbody":
        return apply_transform(rigidbody(matrix), coordinates)
    if selected == "globalrescale":
        return apply_transform(globalrescale(matrix), coordinates)
    if selected == "traditional":
        return apply_transform(traditional(matrix), coordinates)
    if selected == "traditionaldipfit":
        return apply_transform(traditionaldipfit(matrix), coordinates)
    raise ValueError(f"unrecognized transformation method: {method}")


def warp_error(params: Any, points: Any, target: Any, method: str = "traditional") -> float:
    """Return mean Euclidean distance after applying a warp."""
    warped = warp_apply(params, points, method) if np.asarray(params).size else _as_points(points)[0]
    target_points, _ = _as_points(target)
    if warped.shape != target_points.shape:
        raise ValueError("target points must have the same shape as input points")
    return float(np.mean(np.linalg.norm(warped - target_points, axis=1)))


def warp_optim(points: Any, target: Any, method: str = "traditional") -> tuple[np.ndarray, np.ndarray]:
    """Optimize a DIPFIT-style warp from ``points`` to ``target``."""
    source, _ = _as_points(points)
    target_points, _ = _as_points(target)
    if source.shape != target_points.shape:
        raise ValueError("target points must have the same shape as input points")
    selected = method.lower()
    if selected not in _WARP_LEVELS:
        raise ValueError("method must be rigidbody, globalrescale, traditional, or nonlin1..nonlin5")
    level = _WARP_LEVELS[selected]
    params: np.ndarray | None = None
    if level >= 1:
        params = _optimize_warp(np.zeros(6), source, target_points, "rigidbody")
    assert params is not None
    if level >= 2:
        params = _optimize_warp(np.concatenate([params, [1.0]]), source, target_points, "globalrescale")
    if level >= 3:
        params = _optimize_warp(np.concatenate([params, [params[6], params[6]]]), source, target_points, "traditional")
    if level >= 4:
        transform = traditional(params)
        params = np.column_stack([transform[:3, 3], transform[:3, :3]]).reshape(3, 4)
        params = _optimize_warp(params, source, target_points, "nonlin1")
    for order in range(2, max(level - 3, 1) + 1):
        needed = _polynomial_term_count(order)
        params = np.column_stack([params, np.zeros((3, needed - params.shape[1]))])
        params = _optimize_warp(params, source, target_points, f"nonlin{order}")
    return warp_apply(params, source, selected), np.asarray(params, dtype=float)


def electroderealign(cfg: dict[str, Any]) -> dict[str, Any]:
    """Realign electrode points to a template using DIPFIT-compatible methods."""
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dictionary")
    method = str(cfg.get("method", "") or "").lower()
    if method == "realignfiducials":
        method = "realignfiducial"
    if method == "warp":
        method = "traditional"
    if not method:
        raise ValueError("cfg['method'] is required")
    elec = _electrode_struct(cfg.get("elec"), "cfg['elec']")
    template_value = cfg.get("template")
    if method == "interactive":
        raise DIPFITUnavailableError(
            "electroderealign interactive GUI alignment is not available in standalone EEGPrep"
        )
    if template_value is None:
        raise DIPFITUnavailableError("electroderealign currently requires an in-memory template electrode structure")
    templates = template_value if isinstance(template_value, list) else [template_value]
    template_structs = [_electrode_struct(template, "cfg['template']") for template in templates]
    if method == "realignfiducial":
        return _realign_fiducials(elec, template_structs, cfg)
    if method not in _WARP_LEVELS:
        raise ValueError("unknown electroderealign method")
    labels, source, target = _matched_template_points(elec, template_structs, cfg)
    _, transform = warp_optim(source, target, method)
    return {
        "pnt": warp_apply(transform, elec["pnt"], method),
        "label": elec["label"],
        "m": transform,
        "cfg": dict(cfg),
        "matched_label": labels,
    }


def apply_transform(matrix: Any, points: Any) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to Nx3 points."""
    transform = np.asarray(matrix, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError("matrix must be 4x4")
    coordinates, _ = _as_points(points)
    homogeneous = np.column_stack([coordinates, np.ones(coordinates.shape[0])])
    return (homogeneous @ transform.T)[:, :3]


def fieldtripchan2eeglab(loc: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a FieldTrip-style channel structure into EEGPrep chanloc dicts."""
    labels = list(loc.get("label", []))
    points = loc.get("pnt", loc.get("elecpos"))
    coordinates, _ = _as_points(points)
    if len(labels) != coordinates.shape[0]:
        raise ValueError("FieldTrip channel labels and points must have the same length")
    return [
        {"labels": str(label), "X": float(point[0]), "Y": float(point[1]), "Z": float(point[2])}
        for label, point in zip(labels, coordinates, strict=True)
    ]


def _realign_fiducials(elec: dict[str, Any], templates: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    fiducials = list(cfg.get("fiducial", []) or [])
    if not fiducials:
        fiducials = _default_fiducials(elec["label"])
    if len(fiducials) != 3:
        raise ValueError("cfg['fiducial'] must contain three labels")
    elec_fid = _points_for_labels(elec, fiducials)
    template_fids = [_points_for_labels(template, fiducials) for template in templates]
    template_fid = np.mean(np.stack(template_fids, axis=2), axis=2)
    elec_to_common = headcoordinates(elec_fid[0], elec_fid[1], elec_fid[2])
    template_to_common = headcoordinates(template_fid[0], template_fid[1], template_fid[2])
    transform = np.linalg.inv(template_to_common) @ elec_to_common
    return {
        "pnt": apply_transform(transform, elec["pnt"]),
        "label": elec["label"],
        "m": transform,
        "cfg": dict(cfg),
    }


def _matched_template_points(
    elec: dict[str, Any], templates: list[dict[str, Any]], cfg: dict[str, Any]
) -> tuple[list[str], np.ndarray, np.ndarray]:
    case_sensitive = str(cfg.get("casesensitive", "yes")).lower() != "no"
    labels = _selected_labels(cfg.get("channel", "all"), elec["label"], case_sensitive)
    for template in templates:
        template_labels = _label_lookup(template["label"], case_sensitive)
        labels = [label for label in labels if _label_key(label, case_sensitive) in template_labels]
    if not labels:
        raise ValueError("no overlapping electrode labels between input and template")
    source = _points_for_labels(elec, labels, case_sensitive=case_sensitive)
    target_sets = [_points_for_labels(template, labels, case_sensitive=case_sensitive) for template in templates]
    target = np.mean(np.stack(target_sets, axis=2), axis=2)
    return labels, source, target


def _selected_labels(value: Any, labels: list[str], case_sensitive: bool) -> list[str]:
    if value is None or value == "all":
        return list(labels)
    requested = [
        str(item) for item in (value if isinstance(value, Iterable) and not isinstance(value, str) else [value])
    ]
    lookup = _label_lookup(labels, case_sensitive)
    selected = []
    for label in requested:
        key = _label_key(label, case_sensitive)
        if key not in lookup:
            raise ValueError(f"channel label not found: {label}")
        selected.append(lookup[key])
    return selected


def _default_fiducials(labels: list[str]) -> list[str]:
    lower = {label.lower() for label in labels}
    for option in (["nasion", "left", "right"], ["nasion", "lpa", "rpa"], ["nz", "lpa", "rpa"]):
        if set(option).issubset(lower):
            return option
    raise ValueError("could not determine three fiducials; specify cfg['fiducial']")


def _points_for_labels(struct: dict[str, Any], labels: Sequence[str], *, case_sensitive: bool = False) -> np.ndarray:
    lookup = _label_lookup(struct["label"], case_sensitive)
    points = []
    for label in labels:
        key = _label_key(label, case_sensitive)
        if key not in lookup:
            raise ValueError(f"fiducial/channel label not found: {label}")
        points.append(struct["pnt"][struct["label"].index(lookup[key])])
    return np.asarray(points, dtype=float)


def _label_lookup(labels: list[str], case_sensitive: bool) -> dict[str, str]:
    return {_label_key(label, case_sensitive): label for label in labels}


def _label_key(label: str, case_sensitive: bool) -> str:
    return str(label) if case_sensitive else str(label).lower()


def _electrode_struct(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DIPFITUnavailableError(f"{name} must be an in-memory electrode structure in standalone EEGPrep")
    points = value.get("pnt", value.get("elecpos"))
    labels = [str(label) for label in value.get("label", [])]
    coordinates, _ = _as_points(points)
    if len(labels) != coordinates.shape[0]:
        raise ValueError(f"{name} labels and points must have matching lengths")
    return {"label": labels, "pnt": coordinates}


def _optimize_warp(initial: np.ndarray, source: np.ndarray, target: np.ndarray, method: str) -> np.ndarray:
    shape = np.asarray(initial).shape

    def objective(flat: np.ndarray) -> float:
        return warp_error(flat.reshape(shape), source, target, method)

    result = minimize(
        objective,
        np.asarray(initial, dtype=float).ravel(),
        method="Powell",
        options={"maxiter": 1200, "xtol": 1e-7, "ftol": 1e-7, "disp": False},
    )
    return result.x.reshape(shape)


def _apply_polynomial_warp(matrix: np.ndarray, points: np.ndarray, method: str | None) -> np.ndarray:
    params = np.asarray(matrix, dtype=float)
    if params.ndim == 1:
        params = params.reshape(3, -1)
    if params.ndim != 2 or params.shape[0] != 3:
        raise ValueError("nonlinear warp parameters must be a 3xP matrix")
    order = _polynomial_order_from_terms(params.shape[1])
    if method and method.startswith("nonlin") and method != "nonlinear":
        requested_order = int(method.removeprefix("nonlin"))
        if _polynomial_term_count(requested_order) != params.shape[1]:
            raise ValueError("invalid size of nonlinear transformation matrix")
    terms = _polynomial_terms(points, order)
    return terms @ params.T


def _polynomial_order_from_terms(count: int) -> int:
    for order in range(0, 6):
        if _polynomial_term_count(order) == count:
            return order
    raise ValueError("invalid size of nonlinear transformation matrix")


def _polynomial_term_count(order: int) -> int:
    return (order + 1) * (order + 2) * (order + 3) // 6


def _polynomial_terms(points: np.ndarray, order: int) -> np.ndarray:
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    terms = []
    for degree in range(order + 1):
        for x_pow in range(degree, -1, -1):
            remaining = degree - x_pow
            for y_pow in range(remaining, -1, -1):
                z_pow = remaining - y_pow
                terms.append((x**x_pow) * (y**y_pow) * (z**z_pow))
    return np.column_stack(terms)


def _dipfit_axis_rotation(angles_radians: Sequence[float]) -> np.ndarray:
    rx, ry, rz = angles_radians
    cx, cy, cz = np.cos([rx, ry, rz])
    sx, sy, sz = np.sin([rx, ry, rz])
    rotation = np.eye(4)
    rotation[:3, :3] = [
        [cz * cy + sz * sx * sy, sz * cy + cz * sx * sy, cx * sy],
        [-sz * cx, cz * cx, sx],
        [sz * sx * cy - cz * sy, -cz * sx * cy - sz * sy, cx * cy],
    ]
    return rotation


def _as_points(value: Any, *, allow_transposed: bool = False) -> tuple[np.ndarray, bool]:
    array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        if array.size != 3:
            raise ValueError("points must contain 3 coordinates")
        return array.reshape(1, 3), False
    if array.ndim != 2:
        raise ValueError("points must be an Nx3 or 3xN array")
    if array.shape[1] == 3:
        return array, False
    if allow_transposed and array.shape[0] == 3:
        return array.T, True
    raise ValueError("points must be an Nx3 array")


def _as_vector3(value: Any, name: str) -> np.ndarray:
    return _as_numeric_vector(value, 3, name)


def _as_numeric_vector(value: Any, size: int | None, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).ravel()
    if size is not None and array.size != size:
        raise ValueError(f"{name} must contain {size} values")
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _unit(vector: np.ndarray, name: str) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError(f"cannot determine {name} direction from degenerate fiducials")
    return vector / norm


__all__ = [
    "apply_transform",
    "electroderealign",
    "fieldtripchan2eeglab",
    "globalrescale",
    "headcoordinates",
    "homogenous2traditional",
    "mni2tal",
    "mni2tal_matrix",
    "rigidbody",
    "rotate",
    "scale",
    "sph2spm",
    "traditional",
    "traditionaldipfit",
    "translate",
    "warp_apply",
    "warp_error",
    "warp_optim",
]
