"""Four-shell concentric-sphere EEG leadfield.

Port of FieldTrip ``forward/private/eeg_leadfield4.m`` and
``eeg_leadfield4_prepare.m`` from fieldtrip-20250523
(``662af494e2cb701c61e0bd1d52b7c271dc97c58b``). The series is the
Lutkenhoner form of Cuffin and Cohen (1979). The center of the spheres is
the origin. A dipole outside the brain shell is still evaluated; FieldTrip
leaves that check disabled on the EEGLAB path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import lpmv


DEFAULT_SERIES_ORDER = 60


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
    # Colatitude cosine. Clip 1 ulp past the Legendre domain after projection.
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
