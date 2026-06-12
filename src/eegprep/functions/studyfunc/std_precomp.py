"""Precompute STUDY channel and component measures."""

from __future__ import annotations

from copy import deepcopy
import logging
from typing import Any

import numpy as np

from eegprep.functions.popfunc._plot_utils import (
    channel_labels,
    component_activations,
    eeg_epoch_data,
    eeg_times_ms,
    numeric_vector,
    python_literal,
)
from eegprep.functions.popfunc._pop_utils import is_on, parse_key_value_args, parse_text_tokens
from eegprep.functions.sigprocfunc.spectopo import compute_spectra
from eegprep.functions.studyfunc._study_utils import as_alleeg_list, build_python_call, ensure_study
from eegprep.functions.studyfunc.std_checkset import std_checkset
from eegprep.functions.timefreqfunc.newtimef import newtimef


MEASURE_NAMES = ("erp", "spec", "ersp", "itc")
logger = logging.getLogger(__name__)


def std_precomp(
    STUDY: dict[str, Any],
    ALLEEG: list[dict[str, Any]] | None,
    chanorcomp: Any = "channels",
    *args: Any,
    erp: str | bool = "off",
    spec: str | bool = "off",
    ersp: str | bool = "off",
    itc: str | bool = "off",
    scalp: str | bool = "off",
    allcomps: str | bool = "off",
    recompute: str | bool = "off",
    design: int | None = None,
    erpparams: Any = None,
    specparams: Any = None,
    erspparams: Any = None,
    return_com: bool = False,
    **kwargs: Any,
) -> Any:
    """Precompute ERP, spectrum, ERSP, and ITC measures for a STUDY.

    Measures are stored directly in ``STUDY.changrp`` for channels and in the
    parent ``STUDY.cluster`` entry for components. Field names follow EEGLAB's
    cached-measure names while avoiding EEGLAB sidecar files at runtime.
    """
    datasets = as_alleeg_list(ALLEEG)
    if not datasets:
        raise ValueError("std_precomp requires ALLEEG datasets")
    study, datasets = std_checkset(ensure_study(STUDY), datasets)
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    erp = options.pop("erp", erp)
    spec = options.pop("spec", spec)
    ersp = options.pop("ersp", ersp)
    itc = options.pop("itc", itc)
    scalp = options.pop("scalp", scalp)
    allcomps = options.pop("allcomps", allcomps)
    recompute = options.pop("recompute", recompute)
    design = options.pop("design", design if design is not None else study.get("currentdesign") or 1)
    erpparams = options.pop("erpparams", erpparams)
    specparams = options.pop("specparams", specparams)
    erspparams = options.pop("erspparams", erspparams)
    ignored = {"interp", "savetrials", "rmicacomps", "rmclust", "cell", "erpim", "erpimparams"}
    ignored_present = sorted(key for key in options if key in ignored)
    if ignored_present:
        logger.warning("std_precomp: ignoring EEGLAB-only option(s): %s", ", ".join(ignored_present))
    unsupported = sorted(key for key in options if key not in ignored)
    if unsupported:
        raise ValueError(f"Unknown std_precomp option(s): {', '.join(unsupported)}")

    computed = _selected_measures(erp=erp, spec=spec, ersp=ersp, itc=itc)
    if not computed:
        raise ValueError("std_precomp requires at least one measure enabled")
    kind = _measure_kind(chanorcomp)
    force = is_on(recompute)
    if kind == "channels":
        study["changrp"] = _precompute_channels(
            datasets,
            chanorcomp,
            computed,
            int(design),
            cached=_cached_by_name(study.get("changrp")),
            force=force,
            erpparams=_params_dict(erpparams),
            specparams=_params_dict(specparams),
            erspparams=_params_dict(erspparams),
        )
    else:
        cluster = _precompute_components(
            study,
            datasets,
            chanorcomp,
            computed,
            int(design),
            cached=_first_cluster(study.get("cluster")),
            force=force,
            allcomps=is_on(allcomps),
            scalp=is_on(scalp),
            erpparams=_params_dict(erpparams),
            specparams=_params_dict(specparams),
            erspparams=_params_dict(erspparams),
        )
        clusters = list(study.get("cluster") or [])
        if not clusters:
            clusters = [{"name": "ParentCluster", "sets": [], "comps": [], "child": []}]
        clusters[0] = {**clusters[0], **cluster}
        study["cluster"] = clusters
    study.setdefault("etc", {}).setdefault("eegprep", {})["study_measures"] = {
        "kind": kind,
        "design": int(design),
        "computed": computed,
        "storage": "STUDY in-memory JSON-safe arrays",
        "recompute": "on" if is_on(recompute) else "off",
    }
    study["saved"] = "no"
    command = _history_command(
        chanorcomp,
        computed,
        allcomps=allcomps,
        scalp=scalp,
        recompute=recompute,
        design=int(design),
        erpparams=erpparams,
        specparams=specparams,
        erspparams=erspparams,
    )
    return (study, datasets, command) if return_com else (study, datasets)


def _precompute_channels(
    datasets: list[dict[str, Any]],
    chanorcomp: Any,
    computed: list[str],
    design: int,
    *,
    cached: dict[str, dict[str, Any]],
    force: bool,
    erpparams: dict[str, Any],
    specparams: dict[str, Any],
    erspparams: dict[str, Any],
) -> list[dict[str, Any]]:
    labels = channel_labels(datasets[0])
    selected = _channel_indices(chanorcomp, labels)
    groups = []
    for channel_index in selected:
        label = labels[channel_index]
        prior = cached.get(label, {})
        entry: dict[str, Any] = {
            "name": label,
            "channels": [label],
            "sets": list(range(1, len(datasets) + 1)),
            "inds": [int(channel_index + 1)],
            "measureinfo": {
                "kind": "channels",
                "design": design,
                "computed": list(computed),
                "datasets": list(range(1, len(datasets) + 1)),
                "channels": [label],
            },
        }
        if "erp" in computed:
            if _keep_cached(prior, "erpdata", force):
                _carry(entry, prior, ("erpdata", "erptimes"))
            else:
                entry["erpdata"], entry["erptimes"] = _channel_erp(datasets, channel_index, erpparams)
        if "spec" in computed:
            if _keep_cached(prior, "specdata", force):
                _carry(entry, prior, ("specdata", "specfreqs"))
            else:
                entry["specdata"], entry["specfreqs"] = _channel_spec(datasets, channel_index, specparams)
        ersp_cached = _keep_cached(prior, "erspdata", force) if "ersp" in computed else True
        itc_cached = _keep_cached(prior, "itcdata", force) if "itc" in computed else True
        if ("ersp" in computed and not ersp_cached) or ("itc" in computed and not itc_cached):
            tf = _channel_time_frequency(datasets, channel_index, erspparams)
        else:
            tf = None
        if "ersp" in computed:
            if ersp_cached:
                _carry(entry, prior, ("erspdata", "ersptimes", "erspfreqs", "erspbase"))
            else:
                entry["erspdata"] = tf["erspdata"]
                entry["ersptimes"] = tf["times"]
                entry["erspfreqs"] = tf["freqs"]
                entry["erspbase"] = tf["powbase"]
        if "itc" in computed:
            if itc_cached:
                _carry(entry, prior, ("itcdata", "itctimes", "itcfreqs"))
            else:
                entry["itcdata"] = tf["itcdata"]
                entry["itctimes"] = tf["times"]
                entry["itcfreqs"] = tf["freqs"]
        groups.append(entry)
    return groups


def _precompute_components(
    study: dict[str, Any],
    datasets: list[dict[str, Any]],
    chanorcomp: Any,
    computed: list[str],
    design: int,
    *,
    cached: dict[str, Any],
    force: bool,
    allcomps: bool,
    scalp: bool,
    erpparams: dict[str, Any],
    specparams: dict[str, Any],
    erspparams: dict[str, Any],
) -> dict[str, Any]:
    try:
        activations = [component_activations(eeg) for eeg in datasets]
    except ValueError as exc:
        raise ValueError("component precompute requires ICA activations or weights for every dataset") from exc
    selected, selection_mask, pair_sets, pair_comps = _component_selection(
        chanorcomp, study, activations, allcomps=allcomps
    )
    cluster: dict[str, Any] = {
        "name": "ParentCluster",
        "sets": [pair_sets],
        "comps": pair_comps,
        "child": [],
        "measureinfo": {
            "kind": "components",
            "design": design,
            "computed": list(computed),
            "datasets": list(range(1, len(datasets) + 1)),
            "components": (selected + 1).tolist(),
        },
    }
    if scalp:
        cluster["topo"] = _component_topographies(datasets, selected, selection_mask)
    if "erp" in computed:
        if _keep_cached(cached, "erpdata", force):
            _carry(cluster, cached, ("erpdata", "erptimes"))
        else:
            cluster["erpdata"], cluster["erptimes"] = _component_erp(
                datasets, activations, selected, selection_mask, erpparams
            )
    if "spec" in computed:
        if _keep_cached(cached, "specdata", force):
            _carry(cluster, cached, ("specdata", "specfreqs"))
        else:
            cluster["specdata"], cluster["specfreqs"] = _component_spec(
                datasets, activations, selected, selection_mask, specparams
            )
    ersp_cached = _keep_cached(cached, "erspdata", force) if "ersp" in computed else True
    itc_cached = _keep_cached(cached, "itcdata", force) if "itc" in computed else True
    if ("ersp" in computed and not ersp_cached) or ("itc" in computed and not itc_cached):
        tf = _component_time_frequency(datasets, activations, selected, selection_mask, erspparams)
    else:
        tf = None
    if "ersp" in computed:
        if ersp_cached:
            _carry(cluster, cached, ("erspdata", "ersptimes", "erspfreqs", "erspbase"))
        else:
            cluster["erspdata"] = tf["erspdata"]
            cluster["ersptimes"] = tf["times"]
            cluster["erspfreqs"] = tf["freqs"]
            cluster["erspbase"] = tf["powbase"]
    if "itc" in computed:
        if itc_cached:
            _carry(cluster, cached, ("itcdata", "itctimes", "itcfreqs"))
        else:
            cluster["itcdata"] = tf["itcdata"]
            cluster["itctimes"] = tf["times"]
            cluster["itcfreqs"] = tf["freqs"]
    return cluster


def _channel_erp(
    datasets: list[dict[str, Any]], channel_index: int, params: dict[str, Any]
) -> tuple[list[list[float]], list[float]]:
    times = _shared_times(datasets)
    baseline = params.get("rmbase")
    values = []
    for eeg in datasets:
        data = eeg_epoch_data(eeg)[channel_index, :, :]
        data = _remove_baseline(data, times, baseline)
        values.append(np.nanmean(data, axis=1))
    return np.asarray(values, dtype=float).tolist(), times.tolist()


def _component_erp(
    datasets: list[dict[str, Any]],
    activations: list[np.ndarray],
    selected: np.ndarray,
    selection_mask: np.ndarray,
    params: dict[str, Any],
) -> tuple[list[Any], list[float]]:
    times = _shared_times(datasets)
    baseline = params.get("rmbase")
    values = []
    for dataset_index, (eeg, acts) in enumerate(zip(datasets, activations)):
        maps = np.asarray(eeg.get("icawinv", []), dtype=float)
        dataset_values = []
        for component_position, component_index in enumerate(selected):
            if not selection_mask[dataset_index, component_position]:
                dataset_values.append(np.full(times.shape, np.nan))
                continue
            scale = _component_scale(maps, int(component_index))
            data = _remove_baseline(acts[component_index, :, :], times, baseline)
            dataset_values.append(np.nanmean(data, axis=1) * scale)
        values.append(np.asarray(dataset_values, dtype=float))
    return np.asarray(values, dtype=float).tolist(), times.tolist()


def _channel_spec(
    datasets: list[dict[str, Any]], channel_index: int, params: dict[str, Any]
) -> tuple[list[list[float]], list[float]]:
    spectra = []
    freqs = None
    for eeg in datasets:
        data = eeg_epoch_data(eeg)[channel_index : channel_index + 1, :, :]
        spectrum, frequency_values, _specstd = compute_spectra(
            data,
            int(eeg.get("pnts", data.shape[1]) or data.shape[1]),
            float(eeg.get("srate", 1.0) or 1.0),
            winsize=_optional_int(params.get("winsize")),
            overlap=_optional_int(params.get("overlap"), default=0),
            nfft=_optional_int(params.get("nfft")),
        )
        freqs = _check_axis(freqs, frequency_values, "spectrum frequencies")
        spectra.append(spectrum[0])
    return np.asarray(spectra, dtype=float).tolist(), np.asarray(freqs, dtype=float).tolist()


def _component_spec(
    datasets: list[dict[str, Any]],
    activations: list[np.ndarray],
    selected: np.ndarray,
    selection_mask: np.ndarray,
    params: dict[str, Any],
) -> tuple[list[Any], list[float]]:
    spectra: list[list[np.ndarray | None]] = []
    freqs = None
    for dataset_index, (eeg, acts) in enumerate(zip(datasets, activations)):
        dataset_values = []
        for component_position, component_index in enumerate(selected):
            if not selection_mask[dataset_index, component_position]:
                dataset_values.append(None)
                continue
            spectrum, frequency_values, _specstd = compute_spectra(
                acts[component_index : component_index + 1, :, :],
                int(eeg.get("pnts", acts.shape[1]) or acts.shape[1]),
                float(eeg.get("srate", 1.0) or 1.0),
                winsize=_optional_int(params.get("winsize")),
                overlap=_optional_int(params.get("overlap"), default=0),
                nfft=_optional_int(params.get("nfft")),
            )
            freqs = _check_axis(freqs, frequency_values, "spectrum frequencies")
            dataset_values.append(np.asarray(spectrum[0], dtype=float))
        spectra.append(dataset_values)
    if freqs is None:
        raise ValueError("component spectrum precompute has no selected components")
    return _fill_missing_component_rows(spectra, len(freqs)).tolist(), np.asarray(freqs, dtype=float).tolist()


def _channel_time_frequency(
    datasets: list[dict[str, Any]], channel_index: int, params: dict[str, Any]
) -> dict[str, Any]:
    values = []
    itc_values = []
    powbase_values = []
    times = None
    freqs = None
    for eeg in datasets:
        result = _newtimef(eeg_epoch_data(eeg)[channel_index, :, :], eeg, params)
        times = _check_axis(times, result.times, "time-frequency times")
        freqs = _check_axis(freqs, result.freqs, "time-frequency frequencies")
        values.append(result.ersp)
        itc_values.append(np.abs(result.itc))
        powbase_values.append(result.powbase)
    return {
        "erspdata": np.asarray(values, dtype=float).tolist(),
        "itcdata": np.asarray(itc_values, dtype=float).tolist(),
        "powbase": np.asarray(powbase_values, dtype=float).tolist(),
        "times": np.asarray(times, dtype=float).tolist(),
        "freqs": np.asarray(freqs, dtype=float).tolist(),
    }


def _component_time_frequency(
    datasets: list[dict[str, Any]],
    activations: list[np.ndarray],
    selected: np.ndarray,
    selection_mask: np.ndarray,
    params: dict[str, Any],
) -> dict[str, Any]:
    values: list[list[np.ndarray | None]] = []
    itc_values: list[list[np.ndarray | None]] = []
    powbase_values: list[list[np.ndarray | None]] = []
    times = None
    freqs = None
    for dataset_index, (eeg, acts) in enumerate(zip(datasets, activations)):
        dataset_ersp = []
        dataset_itc = []
        dataset_powbase = []
        for component_position, component_index in enumerate(selected):
            if not selection_mask[dataset_index, component_position]:
                dataset_ersp.append(None)
                dataset_itc.append(None)
                dataset_powbase.append(None)
                continue
            result = _newtimef(acts[component_index, :, :], eeg, params)
            times = _check_axis(times, result.times, "time-frequency times")
            freqs = _check_axis(freqs, result.freqs, "time-frequency frequencies")
            dataset_ersp.append(result.ersp)
            dataset_itc.append(np.abs(result.itc))
            dataset_powbase.append(result.powbase)
        values.append(dataset_ersp)
        itc_values.append(dataset_itc)
        powbase_values.append(dataset_powbase)
    if times is None or freqs is None:
        raise ValueError("component time-frequency precompute has no selected components")
    tf_shape = (len(freqs), len(times))
    return {
        "erspdata": _fill_missing_component_rows(values, tf_shape).tolist(),
        "itcdata": _fill_missing_component_rows(itc_values, tf_shape).tolist(),
        "powbase": _fill_missing_component_rows(powbase_values, len(freqs)).tolist(),
        "times": np.asarray(times, dtype=float).tolist(),
        "freqs": np.asarray(freqs, dtype=float).tolist(),
    }


def _component_scale(maps: np.ndarray, component_index: int) -> float:
    if maps.size == 0:
        return 1.0
    if component_index >= maps.shape[1]:
        raise ValueError("requested components exceed EEG.icawinv columns")
    return float(np.sqrt(np.nanmean(maps[:, component_index] ** 2)))


def _fill_missing_component_rows(values: list[list[np.ndarray | None]], shape: int | tuple[int, ...]) -> np.ndarray:
    fill_shape = (shape,) if isinstance(shape, int) else shape
    return np.asarray(
        [
            [np.full(fill_shape, np.nan) if item is None else item for item in dataset_values]
            for dataset_values in values
        ],
        dtype=float,
    )


def _newtimef(data: np.ndarray, eeg: dict[str, Any], params: dict[str, Any]):
    return newtimef(
        data,
        int(eeg.get("pnts", data.shape[0]) or data.shape[0]),
        [float(eeg.get("xmin", 0) or 0) * 1000.0, float(eeg.get("xmax", 0) or 0) * 1000.0],
        float(eeg.get("srate", 1.0) or 1.0),
        params.get("cycles", [3, 0.8]),
        winsize=params.get("winsize"),
        freqs=params.get("freqs"),
        nfreqs=params.get("nfreqs", 40),
        freqscale=params.get("freqscale"),
        timesout=params.get("timesout", 80),
        padratio=params.get("padratio"),
        overlap=params.get("overlap"),
        baseline=params.get("baseline", 0),
        plot="off",
    )


def _component_topographies(
    datasets: list[dict[str, Any]], selected: np.ndarray, selection_mask: np.ndarray
) -> list[Any]:
    topographies = []
    for dataset_index, eeg in enumerate(datasets):
        maps = np.asarray(eeg.get("icawinv", []), dtype=float)
        if maps.size == 0:
            raise ValueError("scalp component precompute requires EEG.icawinv")
        dataset_maps = []
        for component_position, component_index in enumerate(selected):
            if not selection_mask[dataset_index, component_position]:
                dataset_maps.append(np.full(maps.shape[0], np.nan))
                continue
            if component_index >= maps.shape[1]:
                raise ValueError("requested components exceed EEG.icawinv columns")
            dataset_maps.append(maps[:, component_index])
        topographies.append(np.asarray(dataset_maps, dtype=float))
    return np.asarray(topographies, dtype=float).tolist()


def _keep_cached(prior: dict[str, Any], data_field: str, force: bool) -> bool:
    return not force and data_field in prior


def _carry(target: dict[str, Any], prior: dict[str, Any], fields: tuple[str, ...]) -> None:
    for field in fields:
        if field in prior:
            target[field] = prior[field]


def _cached_by_name(changrp: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(changrp, list):
        return {}
    return {entry["name"]: entry for entry in changrp if isinstance(entry, dict) and entry.get("name")}


def _first_cluster(cluster: Any) -> dict[str, Any]:
    if isinstance(cluster, list) and cluster and isinstance(cluster[0], dict):
        return cluster[0]
    return {}


def _measure_kind(chanorcomp: Any) -> str:
    if isinstance(chanorcomp, str) and chanorcomp.lower() == "components":
        return "components"
    return "channels"


def _selected_measures(**kwargs: Any) -> list[str]:
    return [name for name in MEASURE_NAMES if is_on(kwargs[name])]


def _channel_indices(chanorcomp: Any, labels: list[str]) -> np.ndarray:
    if isinstance(chanorcomp, str) and chanorcomp.lower() == "channels":
        return np.arange(len(labels), dtype=int)
    if isinstance(chanorcomp, str):
        requested = parse_text_tokens(chanorcomp, parse_ints=True)
    elif isinstance(chanorcomp, np.ndarray):
        requested = chanorcomp.ravel().tolist()
    elif isinstance(chanorcomp, (int, np.integer, float, np.floating)):
        requested = [chanorcomp]
    else:
        requested = list(chanorcomp or [])
    if not requested:
        return np.arange(len(labels), dtype=int)
    if all(isinstance(item, (int, np.integer, float, np.floating)) for item in requested):
        values = numeric_vector(requested, dtype=int)
        if np.any(values < 1) or np.any(values > len(labels)):
            raise ValueError(f"channels must be 1-based and within 1..{len(labels)}")
        return values - 1
    lookup = {label.lower(): index for index, label in enumerate(labels)}
    missing = [str(label) for label in requested if str(label).lower() not in lookup]
    if missing:
        raise ValueError(f"Unknown channel label(s): {', '.join(missing)}")
    return np.asarray([lookup[str(label).lower()] for label in requested], dtype=int)


def _component_indices(
    chanorcomp: Any,
    study: dict[str, Any],
    activations: list[np.ndarray],
    *,
    allcomps: bool,
) -> np.ndarray:
    count = min(acts.shape[0] for acts in activations)
    if count <= 0:
        raise ValueError("component precompute requires ICA activations or weights")
    if not isinstance(chanorcomp, str) or chanorcomp.lower() != "components":
        values = numeric_vector(chanorcomp, dtype=int)
    elif allcomps:
        return np.arange(count, dtype=int)
    else:
        values = _datasetinfo_components(study, count)
    if values.size == 0:
        return np.arange(count, dtype=int)
    if np.any(values < 1) or np.any(values > count):
        raise ValueError(f"components must be 1-based and within 1..{count}")
    return np.unique(values - 1)


def _component_selection(
    chanorcomp: Any,
    study: dict[str, Any],
    activations: list[np.ndarray],
    *,
    allcomps: bool,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    if not activations or min(acts.shape[0] for acts in activations) <= 0:
        raise ValueError("component precompute requires ICA activations or weights")
    if not isinstance(chanorcomp, str) or chanorcomp.lower() != "components" or allcomps:
        selected = _component_indices(chanorcomp, study, activations, allcomps=allcomps)
        mask = np.ones((len(activations), selected.size), dtype=bool)
        pair_sets = [study_set for study_set in range(1, len(activations) + 1) for _component in selected]
        pair_comps = [int(component + 1) for _study_set in range(len(activations)) for component in selected]
        return selected, mask, pair_sets, pair_comps

    pair_sets = []
    pair_comps = []
    for study_set, acts in enumerate(activations, start=1):
        selected = _dataset_component_values(study, study_set, acts.shape[0])
        for component in selected:
            pair_sets.append(study_set)
            pair_comps.append(int(component))
    component_axis = np.asarray(_unique_preserving_order(pair_comps), dtype=int)
    mask = np.zeros((len(activations), component_axis.size), dtype=bool)
    for study_set, component in zip(pair_sets, pair_comps):
        mask[study_set - 1, int(np.where(component_axis == component)[0][0])] = True
    return component_axis - 1, mask, pair_sets, pair_comps


def _dataset_component_values(study: dict[str, Any], study_set: int, count: int) -> np.ndarray:
    info = (study.get("datasetinfo") or [{}])[study_set - 1]
    values = (
        numeric_vector(info.get("comps"), dtype=int) if isinstance(info, dict) and info.get("comps") else np.asarray([])
    )
    if values.size == 0:
        values = np.arange(1, count + 1, dtype=int)
    if np.any(values < 1) or np.any(values > count):
        raise ValueError(f"dataset {study_set} components must be 1-based and within 1..{count}")
    return np.unique(values.astype(int))


def _datasetinfo_components(study: dict[str, Any], count: int) -> np.ndarray:
    values = []
    for info in study.get("datasetinfo") or []:
        comps = info.get("comps") if isinstance(info, dict) else None
        if comps is not None:
            values.extend(numeric_vector(comps, dtype=int).tolist())
    if not values:
        return np.arange(1, count + 1, dtype=int)
    return np.asarray(values, dtype=int)


def _unique_preserving_order(values: list[int]) -> list[int]:
    output = []
    for value in values:
        if value not in output:
            output.append(value)
    return output


def _shared_times(datasets: list[dict[str, Any]]) -> np.ndarray:
    times = None
    for eeg in datasets:
        times = _check_axis(times, eeg_times_ms(eeg), "ERP times")
    return np.asarray(times, dtype=float)


def _check_axis(existing: Any, candidate: Any, name: str) -> np.ndarray:
    candidate = np.asarray(candidate, dtype=float).ravel()
    if existing is None:
        return candidate
    existing = np.asarray(existing, dtype=float).ravel()
    if existing.shape != candidate.shape or not np.allclose(existing, candidate, equal_nan=True):
        raise ValueError(f"datasets do not share compatible {name}")
    return existing


def _remove_baseline(data: np.ndarray, times: np.ndarray, baseline: Any) -> np.ndarray:
    values = np.asarray(data, dtype=float)
    bounds = numeric_vector(baseline) if baseline is not None else np.asarray([])
    if bounds.size == 0:
        return values
    if bounds.size != 2:
        raise ValueError("rmbase must contain [min max] in milliseconds")
    mask = (times >= bounds[0]) & (times <= bounds[1])
    if not np.any(mask):
        raise ValueError("rmbase does not include any samples")
    return values - np.nanmean(values[mask, :], axis=0, keepdims=True)


def _params_dict(value: Any) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return deepcopy(value)
    if not isinstance(value, (list, tuple)):
        raise ValueError("measure parameters must be dictionaries or key/value sequences")
    if len(value) % 2:
        raise ValueError("measure parameters must contain key/value pairs")
    return {str(value[index]).lower(): value[index + 1] for index in range(0, len(value), 2)}


def _optional_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    vector = numeric_vector(value, dtype=int)
    if vector.size == 0:
        return default
    return int(vector[0])


def _history_command(
    chanorcomp: Any,
    computed: list[str],
    *,
    allcomps: Any,
    scalp: Any,
    recompute: Any,
    design: int,
    erpparams: Any,
    specparams: Any,
    erspparams: Any,
) -> str:
    kwargs: dict[str, Any] = {name: "on" if name in computed else "off" for name in MEASURE_NAMES}
    kwargs["design"] = design
    if is_on(allcomps):
        kwargs["allcomps"] = "on"
    if is_on(scalp):
        kwargs["scalp"] = "on"
    if is_on(recompute):
        kwargs["recompute"] = "on"
    if erpparams:
        kwargs["erpparams"] = erpparams
    if specparams:
        kwargs["specparams"] = specparams
    if erspparams:
        kwargs["erspparams"] = erspparams
    chan_repr = python_literal(chanorcomp)
    return build_python_call(("STUDY", "ALLEEG"), "std_precomp", "STUDY", "ALLEEG", chan_repr, **kwargs)


__all__ = ["std_precomp"]
