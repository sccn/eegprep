"""Read cached STUDY measure arrays."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.popfunc._plot_utils import numeric_vector
from eegprep.functions.studyfunc._cluster_utils import cluster_list, sets_array
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
    timerange: Any = None,
    freqrange: Any = None,
    subject: Any = None,
    infotype: str | None = None,
    **kwargs: Any,
) -> tuple[dict[str, Any], list[np.ndarray], np.ndarray, np.ndarray]:
    """Read precomputed STUDY measures from EEGPrep's in-memory cache."""
    study = ensure_study(STUDY)
    _ = design, ALLEEG, kwargs
    measure = _datatype(infotype or datatype)
    if channels is not None:
        groups = _channel_groups(study, channels)
        data = [
            _slice_measure_data(
                _subject_filter(_data_array(group, measure), study, subject), group, measure, timerange, freqrange
            )
            for group in groups
        ]
        return (
            study,
            [item[0] for item in data],
            data[0][1],
            data[0][2],
        )
    if clusters is not None or components is not None:
        cluster_index = _component_cluster_index(study, clusters)
        clusters_list = cluster_list(study)
        cluster = clusters_list[cluster_index - 1]
        source = clusters_list[0] if cluster_index > 1 else cluster
        data = _data_array(source, measure)
        if cluster_index > 1 and _subject_values(subject):
            raise ValueError("subject filter requires a parent component cache with a dataset axis")
        if cluster_index > 1:
            data = _cluster_component_data(source, cluster, data, components)
        elif components is not None:
            component_axis = component_measure_axis(cluster, data.shape[1] if data.ndim >= 2 else 0)
            selected = component_measure_selection(components, component_axis)
            data = data[:, selected, ...]
        data, x_axis, y_axis = _slice_measure_data(data, source, measure, timerange, freqrange)
        if cluster_index == 1:
            data = _subject_filter(data, study, subject)
        return study, [data], x_axis, y_axis
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


def std_readitc(STUDY: dict[str, Any], ALLEEG: list[dict[str, Any]] | None = None, **kwargs: Any):
    """Read cached STUDY ITC measures."""
    return std_readdata(STUDY, ALLEEG, datatype="itc", **kwargs)


def std_readtopo(
    STUDY: dict[str, Any],
    ALLEEG: list[dict[str, Any]] | None = None,
    *,
    clusters: Any = None,
    components: Any = None,
    **_kwargs: Any,
) -> tuple[dict[str, Any], list[np.ndarray], np.ndarray]:
    """Read cached STUDY component scalp topographies."""
    _ = ALLEEG
    study = ensure_study(STUDY)
    clusters_list = cluster_list(study)
    cluster_index = _component_cluster_index(study, clusters)
    cluster = clusters_list[cluster_index - 1]
    source = clusters_list[0] if cluster_index > 1 else cluster
    if "topo" not in source:
        raise ValueError("component scalp topographies have not been precomputed")
    data = np.asarray(source["topo"], dtype=float)
    if cluster_index > 1:
        data = _cluster_component_data(source, cluster, data, components)
    elif components is not None:
        component_axis = component_measure_axis(source, data.shape[1] if data.ndim >= 2 else 0)
        selected = component_measure_selection(components, component_axis)
        data = data[:, selected, :]
    channel_axis = np.arange(1, data.shape[-1] + 1, dtype=int) if data.ndim else np.asarray([], dtype=int)
    return study, [data], channel_axis


def std_readpac(
    STUDY: dict[str, Any],
    ALLEEG: list[dict[str, Any]] | None = None,
    *,
    channels: Any = None,
    clusters: Any = None,
    components: Any = None,
    timerange: Any = None,
    freqrange: Any = None,
    **kwargs: Any,
) -> tuple[dict[str, Any], list[np.ndarray], np.ndarray, np.ndarray]:
    """Read EEGPrep-owned cached STUDY PAC data when present."""
    _ = ALLEEG
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise ValueError(f"Unknown std_readpac option(s): {unsupported}")
    if components is not None:
        raise ValueError("std_readpac does not yet support component selection for PAC caches")
    study = ensure_study(STUDY)
    if channels is not None:
        groups = _channel_groups(study, channels)
    else:
        cluster_index = _component_cluster_index(study, clusters)
        groups = [cluster_list(study)[cluster_index - 1]]
    missing = [group for group in groups if "pacdata" not in group]
    if missing:
        raise NotImplementedError(
            "STUDY PAC reading requires EEGPrep-owned pacdata caches; external LIMO/PAC toolbox output is not "
            "silently emulated"
        )
    # PAC caches selected together share the first group's axes; shape checks below catch incompatible groups.
    first = groups[0]
    times = np.asarray(first.get("pactimes", []), dtype=float)
    freqs = np.asarray(first.get("pacfreqs", []), dtype=float)
    time_mask = _range_mask(times, timerange)
    freq_mask = _range_mask(freqs, freqrange)
    return (
        study,
        [_slice_pac_cache(group, freq_mask, time_mask) for group in groups],
        times[time_mask],
        freqs[freq_mask],
    )


def _datatype(value: str) -> str:
    text = str(value or "erp").lower()
    if text == "timef":
        text = "ersp"
    if text == "itc":
        return "itc"
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


def _component_cluster_index(study: dict[str, Any], clusters: Any) -> int:
    cluster_list = [group for group in study.get("cluster") or [] if isinstance(group, dict)]
    if not cluster_list:
        raise ValueError("No component measures are stored in STUDY.cluster")
    if _parent_cluster_requested(clusters):
        return 1
    indices = numeric_vector(clusters, dtype=int)
    if indices.size != 1:
        raise ValueError("Only one component cluster can be read at a time")
    index = int(indices[0])
    if index < 1 or index > len(cluster_list):
        raise ValueError(f"clusters must be 1-based and within 1..{len(cluster_list)}")
    return index


def component_measure_axis(group: dict[str, Any], count: int) -> np.ndarray:
    """Return cached component IDs for a STUDY component-measure axis."""
    return _cached_axis_values(group, "components", "comps", count, fallback_start=1)


def component_dataset_axis(group: dict[str, Any], count: int) -> np.ndarray:
    """Return cached STUDY dataset IDs for a component-measure axis."""
    return _cached_axis_values(group, "datasets", "sets", count, fallback_start=1)


def _cached_axis_values(
    group: dict[str, Any], measureinfo_key: str, fallback_key: str, count: int, *, fallback_start: int
) -> np.ndarray:
    raw_measureinfo = group.get("measureinfo")
    measureinfo: dict[str, Any] = raw_measureinfo if isinstance(raw_measureinfo, dict) else {}
    values = numeric_vector(measureinfo.get(measureinfo_key), dtype=int)
    if values.size == count:
        return values.astype(int)
    values = numeric_vector(group.get(fallback_key), dtype=int)
    unique_values = np.asarray(_unique_preserving_order(values.tolist()), dtype=int)
    if unique_values.size == count:
        return unique_values
    return np.arange(fallback_start, fallback_start + count, dtype=int)


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


def _cluster_component_data(
    parent: dict[str, Any], cluster: dict[str, Any], data: np.ndarray, components: Any
) -> np.ndarray:
    if data.ndim < 2:
        raise ValueError("Component measure cache must have dataset and component axes")
    sets = sets_array(cluster.get("sets")).astype(int)
    comps = np.asarray(cluster.get("comps") or [], dtype=int).ravel()
    if sets.shape[1] != comps.size:
        raise ValueError("STUDY.cluster sets and comps lengths do not match")
    if comps.size == 0:
        raise ValueError("Selected STUDY cluster contains no components")
    if components is not None:
        requested = numeric_vector(components, dtype=int)
        if requested.size:
            keep = np.isin(comps, requested)
            sets = sets[:, keep]
            comps = comps[keep]
    if comps.size == 0:
        raise ValueError("Selected STUDY cluster contains no cached components")
    dataset_axis = component_dataset_axis(parent, data.shape[0])
    component_axis = component_measure_axis(parent, data.shape[1])
    rows = []
    for study_set, component in zip(sets[0].tolist(), comps.tolist()):
        dataset_position = _axis_position(dataset_axis, int(study_set), f"dataset {study_set}")
        component_position = _axis_position(component_axis, int(component), f"component {component}")
        rows.append(data[dataset_position, component_position, ...])
    return np.asarray(rows, dtype=float)


def _axis_position(axis: np.ndarray, value: int, label: str) -> int:
    matches = np.where(axis == value)[0]
    if matches.size == 0:
        raise ValueError(f"Component measure cache is missing {label}")
    return int(matches[0])


def _all_channels_requested(channels: Any) -> bool:
    return channels is None or (isinstance(channels, str) and channels in {"", "channels"})


def _parent_cluster_requested(clusters: Any) -> bool:
    return clusters is None or (isinstance(clusters, str) and clusters in {"", "components"})


def _data_array(group: dict[str, Any], measure: str) -> np.ndarray:
    field = {"erp": "erpdata", "spec": "specdata", "ersp": "erspdata", "itc": "itcdata"}[measure]
    if field not in group:
        raise ValueError(f"{measure.upper()} measures have not been precomputed")
    return np.asarray(group[field], dtype=float)


def _slice_measure_data(
    data: np.ndarray, group: dict[str, Any], measure: str, timerange: Any, freqrange: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_axis = _x_axis(group, measure)
    y_axis = _y_axis(group, measure)
    values = np.asarray(data, dtype=float)
    if measure == "erp":
        mask = _range_mask(x_axis, timerange)
        return values[..., mask], x_axis[mask], y_axis
    if measure == "spec":
        mask = _range_mask(x_axis, freqrange)
        return values[..., mask], x_axis[mask], y_axis
    time_mask = _range_mask(x_axis, timerange)
    freq_mask = _range_mask(y_axis, freqrange)
    return values[..., freq_mask, :][..., time_mask], x_axis[time_mask], y_axis[freq_mask]


def _slice_pac_cache(group: dict[str, Any], freq_mask: np.ndarray, time_mask: np.ndarray) -> np.ndarray:
    values = np.asarray(group["pacdata"], dtype=float)
    if values.ndim < 2:
        raise ValueError("PAC cache must have frequency and time axes")
    if values.shape[-2] != freq_mask.size or values.shape[-1] != time_mask.size:
        raise ValueError("PAC cache shape must end with pacfreqs and pactimes axes")
    return values[..., freq_mask, :][..., time_mask]


def _range_mask(axis: np.ndarray, bounds: Any) -> np.ndarray:
    axis = np.asarray(axis, dtype=float).ravel()
    if axis.size == 0:
        return np.asarray([], dtype=bool)
    values = numeric_vector(bounds, dtype=float)
    if values.size == 0:
        return np.ones(axis.size, dtype=bool)
    if values.size != 2:
        raise ValueError("range filters must contain [min max]")
    mask = (axis >= values[0]) & (axis <= values[1])
    if not np.any(mask):
        raise ValueError("range filter does not include any cached measure samples")
    return mask


def _subject_filter(data: np.ndarray, study: dict[str, Any], subject: Any) -> np.ndarray:
    subjects = _subject_values(subject)
    if not subjects:
        return data
    datasetinfo = study.get("datasetinfo") or []
    if data.ndim == 0 or data.shape[0] != len(datasetinfo):
        raise ValueError("subject filter requires a dataset-axis cache")
    keep = [index for index, info in enumerate(datasetinfo) if str(info.get("subject") or "") in subjects]
    if not keep:
        raise ValueError("subject filter did not match any STUDY datasets")
    return data[np.asarray(keep, dtype=int), ...]


def _subject_values(subject: Any) -> set[str]:
    if subject is None or subject == "":
        return set()
    if isinstance(subject, str):
        return {subject}
    if isinstance(subject, np.ndarray):
        subject = subject.tolist()
    if isinstance(subject, (list, tuple, set)):
        return {str(item) for item in subject}
    return {str(subject)}


def _x_axis(group: dict[str, Any], measure: str) -> np.ndarray:
    field = {"erp": "erptimes", "spec": "specfreqs", "ersp": "ersptimes", "itc": "itctimes"}[measure]
    return np.asarray(group.get(field, []), dtype=float)


def _y_axis(group: dict[str, Any], measure: str) -> np.ndarray:
    field = {"ersp": "erspfreqs", "itc": "itcfreqs"}.get(measure)
    if field is None:
        return np.asarray([], dtype=float)
    return np.asarray(group.get(field, []), dtype=float)


__all__ = [
    "component_dataset_axis",
    "component_measure_axis",
    "component_measure_selection",
    "std_readdata",
    "std_readerp",
    "std_readitc",
    "std_readpac",
    "std_readspec",
    "std_readersp",
    "std_readtopo",
]
