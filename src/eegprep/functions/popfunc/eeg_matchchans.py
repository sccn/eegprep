"""Match a small channel montage to a larger montage."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from eegprep.functions.miscfunc.matlab_outputs import as_column
from eegprep.functions.popfunc._chanutils import chanlocs_as_list


def eeg_matchchans(
    big_locations: Any,
    small_locations: Any,
    noplot: str | None = None,
) -> tuple[list[int], np.ndarray, list[dict[str, Any]]]:
    """Return a unique nearest large-montage channel for each small channel.

    Returned channel indices are 1-based. The legacy plotting switch is
    accepted but matching itself is noninteractive.
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
        big[index]["bigchan"] = np.array([[float(index + 1)]])
        big[index]["bigdist"] = np.array([[distance]])
    return (
        as_column(np.asarray(selected, dtype=float) + 1),
        as_column(np.asarray(distances, dtype=float)),
        _location_struct([big[index] for index in selected]),
    )


def _location_struct(records: list[dict[str, Any]]) -> np.ndarray:
    names = list(records[0])
    row = np.empty((1, len(records)), dtype=[(name, object) for name in names])
    for column, record in enumerate(records):
        row[0, column] = tuple(_location_field(record.get(name)) for name in names)
    return row


def _location_field(value: Any) -> Any:
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return np.array([[float(value)]])
    if isinstance(value, np.ndarray):
        return value
    if value is None:
        return np.empty((0, 0))
    return value


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
