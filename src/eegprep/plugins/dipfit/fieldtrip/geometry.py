"""FieldTrip mesh geometry used by the BEM electrode transfer and inside test.

Triangle indices in this module are zero-based. The formulas follow
``project_elec``, ``transfer_elec``, ``lmoutrn``, ``plinprojn``, ``ptriprojn``,
``solid_angle``, ``surface_inside``, and ``surface_nesting`` from FieldTrip
20250523.
"""

from __future__ import annotations

from typing import Any

import numpy as np


_SOLID_ANGLE_TOLERANCE = 1000.0 * np.finfo(float).eps


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


def surface_inside(dippos: np.ndarray, pos: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Return whether each point is inside a closed triangle mesh."""
    points = np.atleast_2d(np.asarray(dippos, dtype=float))
    vertices = np.asarray(pos, dtype=float)
    triangles = np.asarray(tri, dtype=int)
    cube_min = np.min(vertices, axis=0)
    cube_max = np.max(vertices, axis=0)
    center = np.mean(vertices, axis=0)
    sphere_radius = float(np.sqrt(np.min(np.sum((vertices - center) ** 2, axis=1))))
    inside = np.zeros(points.shape[0], dtype=bool)
    for index, point in enumerate(points):
        if np.any(point < cube_min) or np.any(point > cube_max):
            continue
        if float(np.linalg.norm(point - center)) < sphere_radius:
            inside[index] = True
            continue
        angles = solid_angle(vertices - point, triangles)
        if np.any(np.isnan(angles)):
            continue
        total = float(np.sum(angles))
        if abs(total) < _SOLID_ANGLE_TOLERANCE:
            continue
        if abs(abs(total) - 4.0 * np.pi) < _SOLID_ANGLE_TOLERANCE:
            inside[index] = True
    return inside


def surface_nesting(boundaries: list[dict[str, Any]], desired: str) -> np.ndarray:
    """Return boundary indices ordered from the requested nesting end."""
    count = len(boundaries)
    nesting = np.zeros((count, count), dtype=float)
    for outer in range(count):
        for inner in range(count):
            if outer == inner:
                continue
            probe = np.asarray(boundaries[outer]["pos"], dtype=float)[:1]
            nesting[outer, inner] = float(surface_inside(probe, boundaries[inner]["pos"], boundaries[inner]["tri"])[0])
    if int(np.sum(nesting)) != count * (count - 1) // 2:
        raise ValueError("the compartment nesting cannot be determined")
    row_sum = np.sum(nesting, axis=1)
    if desired == "insidefirst":
        return np.argsort(-row_sum, kind="stable")
    if desired == "outsidefirst":
        return np.argsort(row_sum, kind="stable")
    raise ValueError(f"unknown surface order {desired!r}")


def find_innermost_boundary(boundaries: list[dict[str, Any]]) -> int:
    """Return the source-compartment index."""
    return int(surface_nesting(boundaries, "insidefirst")[0])


def find_outermost_boundary(boundaries: list[dict[str, Any]]) -> int:
    """Return the skin-compartment index."""
    return int(surface_nesting(boundaries, "outsidefirst")[0])


def ft_inside_headmodel(dippos: np.ndarray, headmodel: dict[str, Any]) -> np.ndarray:
    """Return whether each dipole is inside the BEM source compartment."""
    source = int(headmodel["source"])
    boundary = headmodel["bnd"][source]
    return surface_inside(dippos, boundary["pos"], boundary["tri"])


def project_elec(elc: np.ndarray, pnt: np.ndarray, tri: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project electrodes onto a triangulation.

    Returns ``el`` as ``N x 4`` ``[triangle, la, mu, distance]`` and the
    projected positions. Triangle indices are zero-based.
    """
    electrodes = np.asarray(elc, dtype=float)
    vertices = np.asarray(pnt, dtype=float)
    triangles = np.asarray(tri, dtype=int)
    if electrodes.ndim == 1:
        electrodes = electrodes.reshape(1, 3)
    projected = np.zeros_like(electrodes)
    weights = np.zeros((electrodes.shape[0], 4), dtype=float)
    corners = (
        vertices[triangles[:, 0]],
        vertices[triangles[:, 1]],
        vertices[triangles[:, 2]],
    )
    for index, electrode in enumerate(electrodes):
        plane, distance = _ptriprojn(corners[0], corners[1], corners[2], electrode, within_triangle=True)
        nearest = int(np.argmin(np.abs(distance)))
        la, mu = _lmoutr(corners[0][nearest], corners[1][nearest], corners[2][nearest], plane[nearest])
        weights[index] = [nearest, la, mu, float(distance[nearest])]
        projected[index] = _routlm(corners[0][nearest], corners[1][nearest], corners[2][nearest], la, mu)
    return weights, projected


def transfer_elec(pnt: np.ndarray, tri: np.ndarray, el: np.ndarray) -> np.ndarray:
    """Return the vertex-to-electrode bilinear transfer."""
    vertices = np.asarray(pnt, dtype=float)
    triangles = np.asarray(tri, dtype=int)
    weights = np.asarray(el, dtype=float)
    transfer = np.zeros((weights.shape[0], vertices.shape[0]), dtype=float)
    for index, row in enumerate(weights):
        triangle = triangles[int(row[0])]
        la = float(row[1])
        mu = float(row[2])
        transfer[index, triangle[0]] = 1.0 - la - mu
        transfer[index, triangle[1]] = la
        transfer[index, triangle[2]] = mu
    return transfer


def _lmoutrn(
    v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, point: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target = np.asarray(point, dtype=float)
    if target.ndim == 1:
        target = np.repeat(target.reshape(1, 3), v1.shape[0], axis=0)
    vec0 = target - v1
    vec1 = v2 - v1
    vec3 = v3 - v1
    stacked = np.vstack([v1, v2, v3])
    origin = np.mean(stacked, axis=0)
    pseudo = _pinv_n_by_2(np.stack([vec1, vec3], axis=2))
    la = np.einsum("ij,ij->i", vec0, pseudo[:, 0, :])
    mu = np.einsum("ij,ij->i", vec0, pseudo[:, 1, :])
    projection = v1 + la[:, None] * vec1 + mu[:, None] * vec3
    origin_point = np.sum((target - origin) ** 2, axis=1)
    origin_projection = np.sum((projection - origin) ** 2, axis=1)
    distance = np.linalg.norm(target - projection, axis=1) * np.sign(origin_point - origin_projection)
    return la, mu, distance, projection


def _ptriprojn(
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
    point: np.ndarray,
    *,
    within_triangle: bool,
) -> tuple[np.ndarray, np.ndarray]:
    _la, mu, distance, projection = _lmoutrn(v1, v2, v3, point)
    la = _la
    if not within_triangle:
        return projection, distance
    outside_v3 = la < 0.0
    if np.any(outside_v3):
        projection[outside_v3], distance[outside_v3] = _plinprojn(
            v1[outside_v3], v3[outside_v3], point, on_segment=True
        )
    outside_v2 = (mu < 0.0) & (la >= 0.0)
    if np.any(outside_v2):
        projection[outside_v2], distance[outside_v2] = _plinprojn(
            v1[outside_v2], v2[outside_v2], point, on_segment=True
        )
    outside_opposite = (la + mu > 1.0) & (mu >= 0.0) & (la >= 0.0)
    if np.any(outside_opposite):
        projection[outside_opposite], distance[outside_opposite] = _plinprojn(
            v2[outside_opposite], v3[outside_opposite], point, on_segment=True
        )
    return projection, distance


def _plinprojn(
    start: np.ndarray, stop: np.ndarray, point: np.ndarray, *, on_segment: bool
) -> tuple[np.ndarray, np.ndarray]:
    segment = stop - start
    target = np.asarray(point, dtype=float)
    if target.ndim == 1:
        target = np.repeat(target.reshape(1, 3), start.shape[0], axis=0)
    numerator = np.sum((target - start) * segment, axis=1)
    denominator = np.sum(segment**2, axis=1)
    parameter = numerator / denominator
    if on_segment:
        parameter = np.clip(parameter, 0.0, 1.0)
    projection = start + parameter[:, None] * segment
    return projection, np.linalg.norm(target - projection, axis=1)


def _lmoutr(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, point: np.ndarray) -> tuple[float, float]:
    vec0 = point - v1
    vec1 = v2 - v1
    vec3 = v3 - v1
    coefficients, *_ = np.linalg.lstsq(np.column_stack([vec1, vec3]), vec0, rcond=None)
    return float(coefficients[0]), float(coefficients[1])


def _routlm(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray, la: float, mu: float) -> np.ndarray:
    return (1.0 - la - mu) * v1 + la * v2 + mu * v3


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
