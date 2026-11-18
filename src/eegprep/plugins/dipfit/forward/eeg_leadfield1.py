"""Single-sphere EEG leadfield.

Port of FieldTrip ``forward/private/eeg_leadfield1.m`` from fieldtrip-20250523
(``662af494e2cb701c61e0bd1d52b7c271dc97c58b``), after Luetkenhoner (1992) and
Kavanagh et al. (1978). The sphere center is the origin.
"""

from __future__ import annotations

from typing import Any

import numpy as np


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
        if tangential_scale < 1000.0 * np.finfo(float).eps:
            numerator = 2.0 / separation_norm**3 * tangential_vector + radial_gain * position
        else:
            direction = sensor / radius - separation / separation_norm
            numerator = (float(np.dot(position, direction)) / tangential_scale + 2.0 / separation_norm**3) * (
                tangential_vector
            ) + radial_gain * position
        leadfield[index] = numerator / scale
    return leadfield
