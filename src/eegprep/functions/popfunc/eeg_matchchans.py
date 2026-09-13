"""Match a small channel montage to a larger montage."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from eegprep.functions.popfunc._chanutils import chanlocs_as_list


def eeg_matchchans(
    big_locations: Any,
    small_locations: Any,
    noplot: str | None = None,
) -> tuple[list[int], np.ndarray, list[dict[str, Any]]]:
    """Return a unique nearest large-montage channel for each small channel.

    Returned channel indices are 0-based for direct use with Python channel
    arrays. The legacy plotting switch is accepted but matching itself is
    noninteractive.
    """
    if noplot is not None and str(noplot).lower() != "noplot":
        raise ValueError("the third argument must be 'noplot' when provided")
    big = [copy.deepcopy(location) for location in chanlocs_as_list(big_locations)]
    small = chanlocs_as_list(small_locations)
    if len(small) > len(big):
        raise ValueError("big_locations must contain at least as many channels as small_locations")
    for location in big:
        location["bigchan"] = []
        location["bigdist"] = []
    used: set[int] = set()
    selected: list[int] = []
    distances: list[float] = []
    for small_location in small:
        small_xyz = _unit_xyz(small_location)
        candidates = [
            (float(np.linalg.norm(small_xyz - _unit_xyz(location))), index)
            for index, location in enumerate(big)
            if index not in used
        ]
        distance, index = min(candidates)
        used.add(index)
        selected.append(index)
        distances.append(distance)
        big[index]["bigchan"] = index
        big[index]["bigdist"] = distance
    return selected, np.asarray(distances, dtype=float), [big[index] for index in selected]


def _unit_xyz(location: dict[str, Any]) -> np.ndarray:
    radius_value = location.get("sph_radius", 1)
    radius = 1.0 if radius_value is None or np.asarray(radius_value).size == 0 else float(radius_value)
    coordinates = [location.get(axis) for axis in ("X", "Y", "Z")]
    if any(value is None or np.asarray(value).size == 0 for value in coordinates):
        raise ValueError("channel locations must define X, Y, and Z")
    coordinates_array = np.asarray(coordinates, dtype=float)
    if not np.isfinite(radius) or radius <= 0 or not np.all(np.isfinite(coordinates_array)):
        raise ValueError("channel locations must contain finite coordinates and a positive spherical radius")
    return coordinates_array / radius


__all__ = ["eeg_matchchans"]
