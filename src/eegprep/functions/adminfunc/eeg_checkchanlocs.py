"""Normalize EEGLAB channel-location structures."""

from __future__ import annotations

from copy import deepcopy
import logging
from typing import Any

import numpy as np

from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.sigprocfunc.convertlocs import convertlocs


logger = logging.getLogger(__name__)

_CHANNEL_FIELDS = (
    "labels",
    "theta",
    "radius",
    "X",
    "Y",
    "Z",
    "sph_theta",
    "sph_phi",
    "sph_radius",
    "type",
    "ref",
    "urchan",
)
_NUMERIC_FIELDS = frozenset({"theta", "radius", "X", "Y", "Z", "sph_theta", "sph_phi", "sph_radius", "urchan"})
_NO_DATA_TYPES = frozenset({"fid", "ignore"})
_NOSE_ROTATION_DEGREES = {"+y": 270.0, "-x": 180.0, "-y": 90.0}


def eeg_checkchanlocs(chans: Any, chaninfo: dict[str, Any] | None = None) -> Any:
    """Normalize channel locations and their shared metadata.

    EEG dictionaries are copied and returned with normalized ``chanlocs`` and
    ``chaninfo``. Passing a channel-location container directly returns
    ``(chanlocs, chaninfo, all_locations)``; the third value retains non-data
    fiducial and ignored locations. Missing numeric fields use ``None`` and
    missing text fields use an empty string.

    Args:
        chans: An EEG dictionary or a channel-location container.
        chaninfo: Shared location metadata for the lower-level container form.

    Returns:
        A normalized EEG dictionary, or the three-value lower-level result.
    """
    if chans is None:
        raise TypeError("chans must be an EEG dictionary or channel-location container")

    is_eeg = isinstance(chans, dict) and "data" in chans
    output = deepcopy(chans) if is_eeg else None
    raw_info = output.get("chaninfo", {}) if output is not None else chaninfo
    if raw_info is None or (isinstance(raw_info, np.ndarray) and raw_info.size == 0):
        raw_info = {}
    if not isinstance(raw_info, dict):
        raise TypeError("chaninfo must be a dictionary")
    info = deepcopy(raw_info)
    locations = chanlocs_as_list(output.get("chanlocs", [])) if output is not None else chanlocs_as_list(chans)
    locations = [deepcopy(location) for location in locations]

    existing_no_data = chanlocs_as_list(info.pop("nodatchans", []))
    tagged_locations = [
        (
            location,
            _datachan_value(location.get("datachan"), default=True) if output is None else True,
            output is None and "datachan" in location,
        )
        for location in locations
    ]
    tagged_locations.extend((deepcopy(location), False, True) for location in existing_no_data)
    normalized = [
        _normalize_location(location, index) for index, (location, _datachan, _explicit) in enumerate(tagged_locations)
    ]

    _move_deprecated_shared_fields(normalized, info)
    _clean_labels(normalized)
    normalized = _normalize_coordinate_systems(normalized)
    normalized, info = _normalize_nose_direction(normalized, info, output)

    strip_urchan = output is not None and not chanlocs_as_list(output.get("urchanlocs", []))
    data_locations: list[dict[str, Any]] = []
    no_data_locations: list[dict[str, Any]] = []
    all_locations: list[dict[str, Any]] = []
    for location, (_original, originally_data, explicit_datachan) in zip(normalized, tagged_locations):
        is_data = originally_data
        if not explicit_datachan:
            is_data = is_data and str(location.get("type", "")).strip().lower() not in _NO_DATA_TYPES
        location = deepcopy(location)
        if strip_urchan:
            location.pop("urchan", None)
        location.pop("datachan", None)
        all_location = deepcopy(location)
        all_location["datachan"] = int(is_data)
        all_locations.append(all_location)
        target = data_locations if is_data else no_data_locations
        if not is_data:
            location["datachan"] = 0
        target.append(location)

    info.setdefault("plotrad", None)
    info.setdefault("shrink", None)
    info.setdefault("nosedir", "+X")
    info["nodatchans"] = no_data_locations
    if "topoplot" not in info and normalized and _looks_like_meg(normalized[0]):
        info["topoplot"] = ["conv", "on", "headrad", 0.3]

    if output is None:
        return data_locations, info, all_locations

    output["chanlocs"] = data_locations
    output["chaninfo"] = info
    return output


def _datachan_value(value: Any, *, default: bool) -> bool:
    if value is None or (isinstance(value, (list, tuple, np.ndarray)) and np.asarray(value).size == 0):
        return default
    if isinstance(value, str):
        return value.strip().casefold() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _looks_like_meg(location: dict[str, Any]) -> bool:
    return "MLC11" in str(location.get("labels", "")) or "meg" in str(location.get("type", "")).casefold()


def _normalize_location(location: Any, index: int) -> dict[str, Any]:
    if not isinstance(location, dict):
        raise TypeError("Channel-location entries must be dictionaries")
    normalized = {
        key: deepcopy(value) for key, value in location.items() if key not in {"sph_phi_besa", "sph_theta_besa"}
    }
    for field in _CHANNEL_FIELDS:
        if field in _NUMERIC_FIELDS:
            normalized[field] = _numeric_value(normalized.get(field))
        else:
            normalized[field] = _text_value(normalized.get(field), label=field == "labels")
    if not normalized["labels"]:
        normalized["labels"] = f"E{index + 1}"
    return normalized


def _numeric_value(value: Any) -> float | int | None:
    if value is None or (isinstance(value, (list, tuple, np.ndarray)) and np.asarray(value).size == 0):
        return None
    if isinstance(value, bool):
        return None
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(scalar):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return scalar


def _text_value(value: Any, *, label: bool) -> str:
    if value is None or (isinstance(value, (list, tuple, np.ndarray)) and np.asarray(value).size == 0):
        return ""
    if isinstance(value, str):
        return value
    prefix = "E" if label and isinstance(value, (int, float, np.integer, np.floating)) else ""
    return f"{prefix}{value}"


def _move_deprecated_shared_fields(locations: list[dict[str, Any]], info: dict[str, Any]) -> None:
    if not locations:
        return
    plotrad = locations[0].get("plotrad")
    if plotrad not in (None, ""):
        try:
            info["plotrad"] = float(plotrad)
        except (TypeError, ValueError):
            info["plotrad"] = plotrad
    shrink = locations[0].get("shrink")
    if shrink not in (None, ""):
        try:
            shrink_value = float(shrink)
            info["plotrad"] = 0.5 / (1.0 - shrink_value)
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    for location in locations:
        location.pop("plotrad", None)
        location.pop("shrink", None)


def _clean_labels(locations: list[dict[str, Any]]) -> None:
    labels = [str(location["labels"]) for location in locations]
    numeric_eeg_labels = sum(_is_numeric(label.replace("EEG", "")) for label in labels)
    if any("EEG" in label for label in labels) and numeric_eeg_labels < 30:
        for location in locations:
            label = str(location["labels"])
            for prefix in ("EEG-", "EEG ", "EEG"):
                label = label.replace(prefix, "")
            location["labels"] = label
    for location in locations:
        label = str(location["labels"])
        label = label.replace("BrainVision RDA_", "").replace("RDA_", "")
        if len(label) >= 2 and label[0] == label[-1] and label[0] in {"'", '"'}:
            label = label[1:-1]
        location["labels"] = label
    lowered = [str(location["labels"]).casefold() for location in locations]
    if len(lowered) != len(set(lowered)):
        logger.warning("Some channels have the same label")


def _is_numeric(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _normalize_coordinate_systems(locations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    requires_conversion = any(
        (location["X"] is not None and location["theta"] is None)
        or (location["sph_theta"] is not None and location["theta"] is None)
        or (location["X"] is not None and location["sph_theta"] is None)
        for location in locations
    )
    if not requires_conversion:
        return locations
    converted = []
    for location in locations:
        try:
            converted.append(convertlocs(location, "auto"))
        except (TypeError, ValueError):
            logger.warning("Unable to convert electrode locations between coordinate systems")
            converted.append(location)
    return [_normalize_location(location, index) for index, location in enumerate(converted)]


def _normalize_nose_direction(
    locations: list[dict[str, Any]],
    info: dict[str, Any],
    eeg: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    direction = str(info.get("nosedir", "+X"))
    degrees = _NOSE_ROTATION_DEGREES.get(direction.lower())
    if degrees is None or not locations:
        return locations, info
    if not all(location["X"] is not None and location["Y"] is not None for location in locations):
        return locations, info

    radians = np.deg2rad(degrees)
    for location in locations:
        coordinate = complex(float(location["Y"]), float(location["X"])) * np.exp(-1j * radians)
        location["Y"] = float(coordinate.real)
        location["X"] = float(coordinate.imag)
        if location["theta"] is not None:
            location["theta"] = _wrap_degrees(float(location["theta"]) - degrees)
        if location["sph_theta"] is not None:
            location["sph_theta"] = _wrap_degrees(float(location["sph_theta"]) + degrees)

    info["originalnosedir"] = direction
    info["nosedir"] = "+X"
    if eeg is not None:
        transform = eeg.get("dipfit", {}).get("coord_transform") if isinstance(eeg.get("dipfit"), dict) else None
        if transform is not None:
            updated = np.asarray(transform, dtype=float).reshape(-1).copy()
            if updated.size == 0:
                updated = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            if updated.size < 6:
                raise ValueError("dipfit.coord_transform must be empty or contain at least six values")
            updated.flat[5] += radians
            eeg["dipfit"]["coord_transform"] = updated
    return locations, info


def _wrap_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0
