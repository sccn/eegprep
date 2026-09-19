"""Rotate channel coordinates so named electrodes reach requested X/Y values."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.sigprocfunc.convertlocs import convertlocs


def forcelocs(chanlocs: Any, *locations: Any) -> Any:
    """Rotate an electrode montage to force named channels to X/Y positions.

    Each location specification is ``(value, axis, label, ...)``. The mean
    coordinate of the named electrodes is rotated with the corresponding Z
    coordinate on the unit sphere. Specifications are applied in order.

    Args:
        chanlocs: EEGLAB channel-location dictionaries with X/Y/Z coordinates.
        *locations: One or more ``(value, "x"|"y", channel labels...)`` specs.

    Returns:
        A deep-copied montage with all coordinate representations refreshed.
    """
    output = convertlocs(chanlocs, "cart2all")
    locs = chanlocs_as_list(output)
    if not locs:
        raise ValueError("forcelocs requires at least one channel location")
    for specification in locations:
        value, axis, labels = _parse_specification(specification)
        indices = _matching_channels(locs, labels)
        coordinate = axis.upper()
        current = float(np.mean([float(locs[index][coordinate]) for index in indices]))
        current_z = float(np.mean([float(locs[index]["Z"]) for index in indices]))
        angle = _rotation_angle(current, current_z, value)
        for loc in locs:
            rotated, rotated_z = _rotate(float(loc[coordinate]), float(loc["Z"]), angle)
            loc[coordinate] = rotated
            loc["Z"] = rotated_z
        refreshed = convertlocs(locs, "cart2all")
        locs[:] = refreshed
    if isinstance(output, dict):
        return locs[0]
    return locs


def _parse_specification(specification: Any) -> tuple[float, str, list[str]]:
    values = list(specification)
    if len(values) < 3:
        raise ValueError("forcelocs specifications must be (value, axis, label, ...)")
    value = float(values[0])
    axis = str(values[1]).lower()
    labels = [str(label).lower() for label in values[2:]]
    if not np.isfinite(value):
        raise ValueError("forcelocs target coordinate must be finite")
    if axis not in {"x", "y"}:
        raise ValueError("forcelocs axis must be 'x' or 'y'")
    return value, axis, labels


def _matching_channels(locs: list[dict[str, Any]], labels: list[str]) -> list[int]:
    wanted = set(labels)
    matched = [index for index, loc in enumerate(locs) if str(loc.get("labels", "")).lower() in wanted]
    missing = sorted(wanted - {str(locs[index].get("labels", "")).lower() for index in matched})
    if missing:
        raise ValueError(f"forcelocs channel labels not found: {', '.join(missing)}")
    return matched


def _rotation_angle(current: float, current_z: float, target: float) -> float:
    radius = float(np.hypot(current, current_z))
    # A rotation cannot increase the selected mean's radius. EEGLAB's complex
    # square-root calculation effectively saturates an out-of-range request at
    # the positive-Z hemisphere boundary; clipping makes that behavior explicit.
    reachable = float(np.clip(target, -radius, radius))
    target_z = float(np.sqrt(max(0.0, radius * radius - reachable * reachable)))
    return float(np.angle(reachable + 1j * target_z) - np.angle(current + 1j * current_z))


def _rotate(value: float, z: float, angle: float) -> tuple[float, float]:
    rotated = (value + 1j * z) * np.exp(1j * angle)
    return float(np.real(rotated)), float(np.imag(rotated))


__all__ = ["forcelocs"]
