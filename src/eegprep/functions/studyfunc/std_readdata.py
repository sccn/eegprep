"""Read cached STUDY measure arrays."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.popfunc._plot_utils import numeric_vector
from eegprep.functions.studyfunc._study_utils import ensure_study


def std_readdata(
    STUDY: dict[str, Any],
    ALLEEG: list[dict[str, Any]] | None = None,
    *,
    datatype: str = "erp",
    channels: Any = None,
    clusters: Any = None,
    components: Any = None,
    design: int | None = None,
    **_kwargs: Any,
) -> tuple[dict[str, Any], list[np.ndarray], np.ndarray, np.ndarray]:
    """Read precomputed STUDY measures from EEGPrep's in-memory cache."""
    study = ensure_study(STUDY)
    measure = _datatype(datatype)
    if channels is not None:
        groups = _channel_groups(study, channels)
        return (
            study,
            [_data_array(group, measure) for group in groups],
            _x_axis(groups[0], measure),
            _y_axis(groups[0], measure),
        )
    if clusters is not None or components is not None:
        cluster = _component_cluster(study, clusters)
        data = _data_array(cluster, measure)
        if components is not None:
            component_axis = component_measure_axis(cluster, data.shape[1] if data.ndim >= 2 else 0)
            selected = component_measure_selection(components, component_axis)
            data = data[:, selected, ...]
        return study, [data], _x_axis(cluster, measure), _y_axis(cluster, measure)
    raise ValueError("std_readdata requires channels or clusters/components")


def std_readerp(STUDY: dict[str, Any], ALLEEG: list[dict[str, Any]] | None = None, **kwargs: Any):
    """Read cached STUDY ERP measures."""
    return std_readdata(STUDY, ALLEEG, datatype="erp", **kwargs)


def std_readspec(STUDY: dict[str, Any], ALLEEG: list[dict[str, Any]] | None = None, **kwargs: Any):
    """Read cached STUDY spectrum measures."""
    return std_readdata(STUDY, ALLEEG, datatype="spec", **kwargs)


def std_readersp(STUDY: dict[str, Any], ALLEEG: list[dict[str, Any]] | None = None, **kwargs: Any):
    """Read cached STUDY ERSP measures."""
    return std_readdata(STUDY, ALLEEG, datatype="ersp", **kwargs)


def _datatype(value: str) -> str:
    text = str(value or "erp").lower()
    if text == "timef":
        text = "ersp"
    if text not in {"erp", "spec", "ersp", "itc"}:
        raise ValueError("datatype must be 'erp', 'spec', 'ersp', or 'itc'")
    return text


def _channel_groups(study: dict[str, Any], channels: Any) -> list[dict[str, Any]]:
    groups = [group for group in study.get("changrp") or [] if isinstance(group, dict)]
    if not groups:
        raise ValueError("No channel measures are stored in STUDY.changrp")
    if _all_channels_requested(channels):
        return groups
    if isinstance(channels, str):
        requested = [channels]
    elif isinstance(channels, (list, tuple)) and all(isinstance(item, str) for item in channels):
        requested = list(channels)
    else:
        indices = numeric_vector(channels, dtype=int)
        if np.any(indices < 1) or np.any(indices > len(groups)):
            raise ValueError(f"channels must be 1-based and within 1..{len(groups)}")
        return [groups[int(index) - 1] for index in indices]
    lookup = {str(group.get("name") or "").lower(): group for group in groups}
    missing = [label for label in requested if label.lower() not in lookup]
    if missing:
        raise ValueError(f"Unknown precomputed channel group(s): {', '.join(missing)}")
    return [lookup[label.lower()] for label in requested]


def _component_cluster(study: dict[str, Any], clusters: Any) -> dict[str, Any]:
    cluster_list = [group for group in study.get("cluster") or [] if isinstance(group, dict)]
    if not cluster_list:
        raise ValueError("No component measures are stored in STUDY.cluster")
    if _parent_cluster_requested(clusters):
        return cluster_list[0]
    indices = numeric_vector(clusters, dtype=int)
    if indices.size != 1:
        raise ValueError("Only one component cluster can be read at a time")
    index = int(indices[0])
    if index < 1 or index > len(cluster_list):
        raise ValueError(f"clusters must be 1-based and within 1..{len(cluster_list)}")
    return cluster_list[index - 1]


def component_measure_axis(group: dict[str, Any], count: int) -> np.ndarray:
    """Return cached component IDs for a STUDY component-measure axis."""
    raw_measureinfo = group.get("measureinfo")
    measureinfo: dict[str, Any] = raw_measureinfo if isinstance(raw_measureinfo, dict) else {}
    values = numeric_vector(measureinfo.get("components"), dtype=int)
    if values.size == count:
        return values.astype(int)
    values = numeric_vector(group.get("comps"), dtype=int)
    unique_values = np.asarray(_unique_preserving_order(values.tolist()), dtype=int)
    if unique_values.size == count:
        return unique_values
    return np.arange(1, count + 1, dtype=int)


def component_measure_selection(components: Any, axis: np.ndarray) -> np.ndarray:
    """Map EEGLAB-facing component IDs to cached component-axis positions."""
    axis = np.asarray(axis, dtype=int).ravel()
    if isinstance(components, str) and components.lower() == "all":
        return np.arange(axis.size, dtype=int)
    indices = numeric_vector(components, dtype=int)
    if indices.size == 0:
        return np.arange(axis.size, dtype=int)
    selected = []
    missing = []
    for value in indices.tolist():
        matches = np.where(axis == int(value))[0]
        if matches.size == 0:
            missing.append(int(value))
        else:
            selected.append(int(matches[0]))
    if missing:
        available = ", ".join(str(value) for value in axis.tolist()) or "none"
        requested = ", ".join(str(value) for value in missing)
        raise ValueError(f"components {requested} are not cached; available component IDs: {available}")
    return np.asarray(selected, dtype=int)


def _unique_preserving_order(values: list[int]) -> list[int]:
    output = []
    for value in values:
        if value not in output:
            output.append(value)
    return output


def _all_channels_requested(channels: Any) -> bool:
    return channels is None or (isinstance(channels, str) and channels in {"", "channels"})


def _parent_cluster_requested(clusters: Any) -> bool:
    return clusters is None or (isinstance(clusters, str) and clusters in {"", "components"})


def _data_array(group: dict[str, Any], measure: str) -> np.ndarray:
    field = {"erp": "erpdata", "spec": "specdata", "ersp": "erspdata", "itc": "itcdata"}[measure]
    if field not in group:
        raise ValueError(f"{measure.upper()} measures have not been precomputed")
    return np.asarray(group[field], dtype=float)


def _x_axis(group: dict[str, Any], measure: str) -> np.ndarray:
    field = {"erp": "erptimes", "spec": "specfreqs", "ersp": "ersptimes", "itc": "itctimes"}[measure]
    return np.asarray(group.get(field, []), dtype=float)


def _y_axis(group: dict[str, Any], measure: str) -> np.ndarray:
    field = {"ersp": "erspfreqs", "itc": "itcfreqs"}.get(measure)
    if field is None:
        return np.asarray([], dtype=float)
    return np.asarray(group.get(field, []), dtype=float)


__all__ = [
    "component_measure_axis",
    "component_measure_selection",
    "std_readdata",
    "std_readerp",
    "std_readspec",
    "std_readersp",
]
