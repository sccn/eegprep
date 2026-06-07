"""Prepare standalone STUDY channel-neighbor structures."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.popfunc._plot_utils import numeric_vector
from eegprep.functions.popfunc._pop_utils import is_on, parse_key_value_args
from eegprep.functions.studyfunc._study_utils import as_alleeg_list, build_python_call, ensure_study
from eegprep.functions.studyfunc.std_checkset import std_checkset


def std_prepare_neighbors(
    STUDY: dict[str, Any],
    ALLEEG: list[dict[str, Any]] | None,
    *args: Any,
    force: str | bool = "off",
    channels: Any = None,
    method: str = "distance",
    neighbordist: Any = None,
    return_com: bool = False,
    **kwargs: Any,
) -> Any:
    """Prepare a FieldTrip-like neighbor list and LIMO adjacency matrix."""
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    force = options.pop("force", force)
    channels = options.pop("channels", channels)
    method = str(options.pop("method", method) or "distance").lower()
    neighbordist = options.pop("neighbordist", neighbordist)
    if options:
        raise ValueError(f"Unknown std_prepare_neighbors option(s): {', '.join(sorted(options))}")
    study, datasets = std_checkset(ensure_study(STUDY), as_alleeg_list(ALLEEG))
    if not is_on(force) and not _study_requests_neighbors(study):
        neighbors: list[dict[str, Any]] = []
        limostruct = {"expected_chanlocs": [], "channeighbstructmat": np.empty((0, 0), dtype=int)}
        return _neighbor_result(study, neighbors, limostruct, "", return_com=return_com)
    if method not in {"distance", "triangulation"}:
        raise NotImplementedError("std_prepare_neighbors supports standalone distance-based neighbors")
    locs = _select_locs(_merged_chanlocs(datasets), channels)
    if not locs:
        raise ValueError("std_prepare_neighbors requires channel locations")
    coordinates = _coordinates(locs)
    distance = _neighbor_distance(coordinates, neighbordist)
    adjacency = _adjacency(coordinates, distance)
    labels = [str(loc.get("labels") or index + 1) for index, loc in enumerate(locs)]
    neighbors = [
        {
            "label": labels[index],
            "neighblabel": [labels[other] for other in np.where(adjacency[index])[0].tolist()],
        }
        for index in range(len(labels))
    ]
    limostruct = {
        "expected_chanlocs": deepcopy(locs),
        "channeighbstructmat": adjacency.astype(int),
    }
    fieldtrip = study.setdefault("etc", {}).setdefault("statistics", {}).setdefault("fieldtrip", {})
    fieldtrip["channelneighbor"] = deepcopy(neighbors)
    fieldtrip["channelneighborparam"] = {"method": method, "neighbordist": float(distance)}
    command = build_python_call(
        ("STUDY", "neighbors", "limostruct"),
        "std_prepare_neighbors",
        "STUDY",
        "ALLEEG",
        force=force,
        channels=channels,
        method=method,
        neighbordist=neighbordist,
    )
    return _neighbor_result(study, neighbors, limostruct, command, return_com=return_com)


def _neighbor_result(
    study: dict[str, Any],
    neighbors: list[dict[str, Any]],
    limostruct: dict[str, Any],
    command: str,
    *,
    return_com: bool,
) -> Any:
    result = (study, neighbors, limostruct)
    return (*result, command) if return_com else result


def _study_requests_neighbors(study: dict[str, Any]) -> bool:
    statistics = study.get("etc", {}).get("statistics", {})
    fieldtrip = statistics.get("fieldtrip", {}) if isinstance(statistics, dict) else {}
    return (
        str(statistics.get("mode", "")).lower() == "fieldtrip"
        and str(fieldtrip.get("mcorrect", "")).lower() == "cluster"
        and (is_on(statistics.get("groupstats")) or is_on(statistics.get("condstats")))
    )


def _merged_chanlocs(datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for eeg in datasets:
        for loc in chanlocs_as_list(eeg.get("chanlocs")):
            label = str(loc.get("labels") or "")
            key = label.lower() if label else repr(loc)
            if key in seen:
                continue
            seen.add(key)
            merged.append(deepcopy(loc))
    return merged


def _select_locs(locs: list[dict[str, Any]], channels: Any) -> list[dict[str, Any]]:
    if channels is None or (isinstance(channels, str) and channels.lower() in {"", "channels"}):
        return locs
    if isinstance(channels, str):
        requested_labels = [channels]
    elif isinstance(channels, (list, tuple)) and all(isinstance(item, str) for item in channels):
        requested_labels = list(channels)
    else:
        indices = numeric_vector(channels, dtype=int)
        if indices.size == 0:
            return locs
        if np.any(indices < 1) or np.any(indices > len(locs)):
            raise ValueError(f"channels must be 1-based and within 1..{len(locs)}")
        return [locs[int(index) - 1] for index in indices]
    lookup = {str(loc.get("labels") or "").lower(): loc for loc in locs}
    missing = [label for label in requested_labels if label.lower() not in lookup]
    if missing:
        raise ValueError(f"Unknown channel label(s): {', '.join(missing)}")
    return [lookup[label.lower()] for label in requested_labels]


def _coordinates(locs: list[dict[str, Any]]) -> np.ndarray:
    xyz = []
    for loc in locs:
        if all(_finite(loc.get(axis)) for axis in ("X", "Y", "Z")):
            xyz.append([float(loc["X"]), float(loc["Y"]), float(loc["Z"])])
        elif _finite(loc.get("theta")) and _finite(loc.get("radius")):
            theta = np.deg2rad(float(loc["theta"]))
            radius = float(loc["radius"])
            xyz.append([radius * np.cos(theta), radius * np.sin(theta), 0.0])
        else:
            raise ValueError("channel locations require X/Y/Z or theta/radius coordinates")
    values = np.asarray(xyz, dtype=float)
    if values.shape[0] < 2:
        raise ValueError("at least two channels are required to prepare neighbors")
    return values


def _finite(value: Any) -> bool:
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _neighbor_distance(coordinates: np.ndarray, neighbordist: Any) -> float:
    values = numeric_vector(neighbordist, dtype=float)
    if values.size:
        distance = float(values[0])
        if distance <= 0:
            raise ValueError("neighbordist must be positive")
        return distance
    distances = _pairwise_distances(coordinates)
    np.fill_diagonal(distances, np.inf)
    nearest = np.min(distances, axis=1)
    return float(np.nanmedian(nearest) * 1.5)


def _adjacency(coordinates: np.ndarray, distance: float) -> np.ndarray:
    distances = _pairwise_distances(coordinates)
    adjacency = (distances <= distance) & (distances > 0)
    for index in range(adjacency.shape[0]):
        if not np.any(adjacency[index]):
            nearest = int(np.argmin(np.where(distances[index] == 0, np.inf, distances[index])))
            adjacency[index, nearest] = True
            adjacency[nearest, index] = True
    return adjacency


def _pairwise_distances(coordinates: np.ndarray) -> np.ndarray:
    delta = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    return np.sqrt(np.sum(delta * delta, axis=2))


__all__ = ["std_prepare_neighbors"]
