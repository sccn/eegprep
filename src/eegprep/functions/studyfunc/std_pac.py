"""Compute EEGPrep-owned STUDY PAC caches."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from eegprep.functions.popfunc._plot_utils import (
    channel_labels,
    component_activations,
    eeg_epoch_data,
    numeric_vector,
)
from eegprep.functions.popfunc._pop_utils import is_on, parse_key_value_args
from eegprep.functions.studyfunc._study_utils import as_alleeg_list, build_python_call, ensure_study, range_mask
from eegprep.functions.studyfunc.std_checkset import std_checkset
from eegprep.functions.studyfunc.std_savedat import std_savedat
from eegprep.functions.timefreqfunc.pac import pac


def std_pac(
    EEG_or_STUDY: dict[str, Any],
    ALLEEG: list[dict[str, Any]] | None = None,
    *args: Any,
    components1: Any = None,
    components2: Any = None,
    channels1: Any = None,
    channels2: Any = None,
    outputfile: str = "",
    plot: str | bool = "off",
    recompute: str | bool = "off",
    getparams: str | bool = "off",
    timerange: Any = None,
    freqrange: Any = None,
    padratio: Any = 1,
    freqs: Any = None,
    cycles: Any = None,
    freqphase: Any = None,
    cyclephase: Any = None,
    interp: Any = None,
    rmcomps: Any = None,
    freqscale: str = "log",
    return_com: bool = False,
    **kwargs: Any,
) -> Any:
    """Compute phase-amplitude coupling for an EEG dataset or STUDY.

    Single-dataset calls return ``(pacdata, times, freqs, parameters)``. STUDY
    calls return ``(STUDY, ALLEEG)`` and store the computed cache under
    ``STUDY.changrp[*].pacdata`` using EEGPrep's in-memory measure contract.
    """
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    components1 = options.pop("components1", components1)
    components2 = options.pop("components2", components2)
    channels1 = options.pop("channels1", channels1)
    channels2 = options.pop("channels2", channels2)
    outputfile = str(options.pop("outputfile", outputfile) or "")
    plot = options.pop("plot", plot)
    recompute = options.pop("recompute", recompute)
    getparams = options.pop("getparams", getparams)
    timerange = options.pop("timerange", options.pop("timewindow", timerange))
    freqrange = options.pop("freqrange", freqrange)
    padratio = options.pop("padratio", padratio)
    freqs = options.pop("freqs", freqs)
    cycles = options.pop("cycles", cycles)
    freqphase = options.pop("freqphase", freqphase)
    cyclephase = options.pop("cyclephase", cyclephase)
    interp = options.pop("interp", interp)
    rmcomps = options.pop("rmcomps", rmcomps)
    freqscale = str(options.pop("freqscale", freqscale) or "log")
    if is_on(plot):
        options.setdefault("newfig", "on")

    if _looks_like_study(EEG_or_STUDY):
        return _std_pac_study(
            EEG_or_STUDY,
            ALLEEG,
            components1=components1,
            components2=components2,
            channels1=channels1,
            channels2=channels2,
            recompute=recompute,
            getparams=getparams,
            timerange=timerange,
            freqrange=freqrange,
            padratio=padratio,
            freqs=freqs,
            cycles=cycles,
            freqphase=freqphase,
            cyclephase=cyclephase,
            interp=interp,
            rmcomps=rmcomps,
            freqscale=freqscale,
            return_com=return_com,
            pac_options=options,
        )

    pacdata, times, pacfreqs, parameters = _std_pac_dataset(
        EEG_or_STUDY,
        components1=components1,
        components2=components2,
        channels1=channels1,
        channels2=channels2,
        timerange=timerange,
        freqrange=freqrange,
        padratio=padratio,
        freqs=freqs,
        cycles=cycles,
        freqphase=freqphase,
        cyclephase=cyclephase,
        interp=interp,
        rmcomps=rmcomps,
        freqscale=freqscale,
        pac_options=options,
    )
    if outputfile:
        std_savedat(
            Path(outputfile),
            {"pacdata": pacdata, "pactimes": times, "pacfreqs": pacfreqs, "parameters": parameters},
        )
    command = _dataset_history_command(
        components1=components1,
        components2=components2,
        channels1=channels1,
        channels2=channels2,
        outputfile=outputfile,
        timerange=timerange,
        freqrange=freqrange,
        padratio=padratio,
        freqs=freqs,
        cycles=cycles,
        freqphase=freqphase,
        cyclephase=cyclephase,
        freqscale=freqscale,
        pac_options=options,
    )
    return (pacdata, times, pacfreqs, parameters, command) if return_com else (pacdata, times, pacfreqs, parameters)


def _std_pac_study(
    STUDY: dict[str, Any],
    ALLEEG: list[dict[str, Any]] | None,
    *,
    components1: Any,
    components2: Any,
    channels1: Any,
    channels2: Any,
    recompute: Any,
    getparams: Any,
    timerange: Any,
    freqrange: Any,
    padratio: Any,
    freqs: Any,
    cycles: Any,
    freqphase: Any,
    cyclephase: Any,
    interp: Any,
    rmcomps: Any,
    freqscale: str,
    return_com: bool,
    pac_options: dict[str, Any],
) -> Any:
    if components1 is not None or components2 is not None:
        raise NotImplementedError("STUDY component PAC caches require a component-clustering contract")
    if interp not in (None, "", [], ()):
        raise NotImplementedError("std_pac STUDY interpolation should be run explicitly with std_interp first")
    if rmcomps not in (None, "", [], ()):
        raise NotImplementedError("std_pac STUDY ICA-component removal is not silently applied")
    datasets = as_alleeg_list(ALLEEG)
    if not datasets:
        raise ValueError("std_pac STUDY mode requires ALLEEG datasets")
    study, datasets = std_checkset(ensure_study(STUDY), datasets)
    params = _parameters(datasets[0], freqs, cycles, freqphase, cyclephase, padratio, freqscale, pac_options)
    if is_on(getparams):
        return (
            ({}, np.asarray([]), np.asarray([]), params)
            if not return_com
            else ({}, np.asarray([]), np.asarray([]), params, "")
        )
    phase_freqs = numeric_vector(params["freqphase"], dtype=float)
    if phase_freqs.size != 1:
        raise ValueError("std_pac STUDY caches require one freqphase value")

    selected1 = _channel_indices(channels1, channel_labels(datasets[0]))
    selected2 = _channel_indices(channels2, channel_labels(datasets[0]))
    groups = _ensure_channel_groups(study, datasets[0], selected1)
    for group in groups:
        if "pacdata" in group and not is_on(recompute):
            continue
        amp_channel = int(group["inds"][0])
        cache_rows = []
        for eeg in datasets:
            dataset_pairs = []
            for phase_channel in selected2:
                pacdata, times, pacfreqs, _parameters_dict = _std_pac_dataset(
                    eeg,
                    components1=None,
                    components2=None,
                    channels1=[amp_channel],
                    channels2=[int(phase_channel) + 1],
                    timerange=timerange,
                    freqrange=freqrange,
                    padratio=padratio,
                    freqs=freqs,
                    cycles=cycles,
                    freqphase=freqphase,
                    cyclephase=cyclephase,
                    interp=None,
                    rmcomps=None,
                    freqscale=freqscale,
                    pac_options=pac_options,
                )
                dataset_pairs.append(np.squeeze(pacdata, axis=(0, 1)))
            cache_rows.append(np.asarray(dataset_pairs, dtype=float))
        cache = np.asarray(cache_rows, dtype=float)
        if cache.shape[1] == 1:
            cache = cache[:, 0, :, :]
        group["pacdata"] = cache.tolist()
        group["pactimes"] = np.asarray(times, dtype=float).tolist()
        group["pacfreqs"] = np.asarray(pacfreqs, dtype=float).tolist()
        group["pacphasefreqs"] = phase_freqs.tolist()
        group["pacchannels2"] = (selected2 + 1).astype(int).tolist()
        measureinfo = group.setdefault("measureinfo", {})
        computed = list(measureinfo.get("computed") or [])
        if "pac" not in computed:
            computed.append("pac")
        measureinfo.update(
            {
                "kind": "channels",
                "computed": computed,
                "pac": {
                    "channels1": [amp_channel],
                    "channels2": (selected2 + 1).astype(int).tolist(),
                    "parameters": params,
                    "storage": "STUDY.changrp in-memory pacdata",
                },
            }
        )
    study["changrp"] = _replace_channel_groups(study, groups)
    study.setdefault("etc", {}).setdefault("eegprep", {})["study_pac"] = {
        "channels1": (selected1 + 1).astype(int).tolist(),
        "channels2": (selected2 + 1).astype(int).tolist(),
        "parameters": params,
        "storage": "STUDY.changrp in-memory pacdata",
    }
    study["saved"] = "no"
    command = build_python_call(
        ("STUDY", "ALLEEG"),
        "std_pac",
        "STUDY",
        "ALLEEG",
        channels1=(selected1 + 1).astype(int).tolist(),
        channels2=(selected2 + 1).astype(int).tolist(),
        recompute=recompute,
        timerange=timerange,
        freqrange=freqrange,
        padratio=padratio,
        freqs=freqs,
        cycles=cycles,
        freqphase=freqphase,
        cyclephase=cyclephase,
        freqscale=freqscale,
        **pac_options,
    )
    return (study, datasets, command) if return_com else (study, datasets)


def _std_pac_dataset(
    EEG: dict[str, Any],
    *,
    components1: Any,
    components2: Any,
    channels1: Any,
    channels2: Any,
    timerange: Any,
    freqrange: Any,
    padratio: Any,
    freqs: Any,
    cycles: Any,
    freqphase: Any,
    cyclephase: Any,
    interp: Any,
    rmcomps: Any,
    freqscale: str,
    pac_options: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if interp not in (None, "", [], ()):
        raise NotImplementedError("std_pac does not silently interpolate channels; run pop_interp/std_interp first")
    if rmcomps not in (None, "", [], ()):
        raise NotImplementedError("std_pac does not silently remove ICA components")
    params = _parameters(EEG, freqs, cycles, freqphase, cyclephase, padratio, freqscale, pac_options)
    amp_sources, _amp_ids = _source_data(EEG, channels1, components1, default="amplitude")
    phase_sources, _phase_ids = _source_data(EEG, channels2, components2, default="phase")
    phase_freqs = numeric_vector(params["freqphase"], dtype=float)
    if phase_freqs.size != 1:
        raise ValueError("std_pac requires one freqphase value; use pac() directly for phase-frequency grids")
    pair_values = []
    times = None
    pacfreqs = None
    for amp_signal in amp_sources:
        phase_values = []
        for phase_signal in phase_sources:
            result = pac(
                amp_signal,
                phase_signal,
                float(EEG.get("srate", 1.0) or 1.0),
                freqs=params["freqs"],
                freqs2=params["freqphase"],
                wavelet=params["cycles"],
                wavelet2=params["cyclephase"],
                padratio=params["padratio"],
                freqscale=params["freqscale"],
                tlimits=_eeg_tlimits(EEG),
                **pac_options,
            )
            values = np.abs(np.squeeze(result.pac, axis=1))
            values, result_times, result_freqs = _apply_ranges(
                values, result.times, result.freqs1, timerange, freqrange
            )
            times = _check_axis(times, result_times, "PAC times")
            pacfreqs = _check_axis(pacfreqs, result_freqs, "PAC frequencies")
            phase_values.append(values)
        pair_values.append(np.asarray(phase_values, dtype=float))
    return (
        np.asarray(pair_values, dtype=float),
        np.asarray(times, dtype=float),
        np.asarray(pacfreqs, dtype=float),
        params,
    )


def _source_data(
    EEG: dict[str, Any], channels: Any, components: Any, *, default: str
) -> tuple[list[np.ndarray], list[int]]:
    if channels is not None and components is not None:
        raise ValueError("std_pac cannot mix channel and component selectors for one input side")
    if components is not None:
        activations = component_activations(EEG)
        indices = _component_indices(components, activations.shape[0])
        return [activations[int(index), :, :] for index in indices], (indices + 1).astype(int).tolist()
    labels = channel_labels(EEG)
    if channels is None and _has_ica(EEG):
        activations = component_activations(EEG)
        indices = np.arange(activations.shape[0], dtype=int)
        return [activations[int(index), :, :] for index in indices], (indices + 1).astype(int).tolist()
    if channels is None and default == "phase":
        channels = "channels"
    indices = _channel_indices(channels, labels)
    data = eeg_epoch_data(EEG)
    return [data[int(index), :, :] for index in indices], (indices + 1).astype(int).tolist()


def _has_ica(EEG: dict[str, Any]) -> bool:
    weights = np.asarray(EEG.get("icaweights", []), dtype=float)
    return weights.size > 0 and weights.ndim == 2


def _channel_indices(value: Any, labels: list[str]) -> np.ndarray:
    if value is None or value == "" or (isinstance(value, str) and value.lower() == "channels"):
        return np.arange(len(labels), dtype=int)
    if isinstance(value, str):
        requested = [value]
    elif isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        requested = list(value)
    else:
        indices = numeric_vector(value, dtype=int)
        if indices.size == 0:
            return np.arange(len(labels), dtype=int)
        if np.any(indices < 1) or np.any(indices > len(labels)):
            raise ValueError(f"channels must be 1-based and within 1..{len(labels)}")
        return indices.astype(int) - 1
    lookup = {label.lower(): index for index, label in enumerate(labels)}
    missing = [label for label in requested if label.lower() not in lookup]
    if missing:
        raise ValueError(f"Unknown channel label(s): {', '.join(missing)}")
    return np.asarray([lookup[label.lower()] for label in requested], dtype=int)


def _component_indices(value: Any, count: int) -> np.ndarray:
    if value is None or value == "" or (isinstance(value, str) and value.lower() in {"components", "all"}):
        return np.arange(count, dtype=int)
    indices = numeric_vector(value, dtype=int)
    if indices.size == 0:
        return np.arange(count, dtype=int)
    if np.any(indices < 1) or np.any(indices > count):
        raise ValueError(f"components must be 1-based and within 1..{count}")
    return indices.astype(int) - 1


def _parameters(
    EEG: dict[str, Any],
    freqs: Any,
    cycles: Any,
    freqphase: Any,
    cyclephase: Any,
    padratio: Any,
    freqscale: str,
    pac_options: dict[str, Any],
) -> dict[str, Any]:
    srate = float(EEG.get("srate", 1.0) or 1.0)
    return {
        "cycles": _vector_or_default(cycles, [8.0]),
        "padratio": int(numeric_vector(padratio, dtype=int)[0]) if numeric_vector(padratio, dtype=int).size else 1,
        "freqphase": _vector_or_default(freqphase, [5.0]),
        "cyclephase": _vector_or_default(cyclephase, [3.0]),
        "freqscale": str(freqscale or "log"),
        "freqs": _vector_or_default(freqs, [12.0, srate / 2.0]),
        "pac_options": deepcopy(pac_options),
    }


def _vector_or_default(value: Any, default: list[float]) -> list[float]:
    values = numeric_vector(value, dtype=float)
    if values.size == 0:
        values = np.asarray(default, dtype=float)
    return values.astype(float).tolist()


def _apply_ranges(
    values: np.ndarray, times: np.ndarray, freqs: np.ndarray, timerange: Any, freqrange: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_mask = _range_mask(times, timerange, "timerange")
    freq_mask = _range_mask(freqs, freqrange, "freqrange")
    return values[freq_mask, :][:, time_mask], np.asarray(times)[time_mask], np.asarray(freqs)[freq_mask]


def _range_mask(axis: np.ndarray, bounds: Any, name: str) -> np.ndarray:
    return range_mask(axis, bounds, name=name, empty_message=f"{name} does not include any PAC samples")


def _check_axis(existing: Any, candidate: np.ndarray, name: str) -> np.ndarray:
    candidate = np.asarray(candidate, dtype=float).ravel()
    if existing is None:
        return candidate
    if candidate.shape != np.asarray(existing).shape or not np.allclose(candidate, existing, equal_nan=True):
        raise ValueError(f"datasets do not share compatible {name}")
    return np.asarray(existing, dtype=float)


def _ensure_channel_groups(study: dict[str, Any], eeg: dict[str, Any], selected: np.ndarray) -> list[dict[str, Any]]:
    labels = channel_labels(eeg)
    existing = [group for group in study.get("changrp") or [] if isinstance(group, dict)]
    by_index: dict[int, dict[str, Any]] = {}
    for group in existing:
        indices = numeric_vector(group.get("inds"), dtype=int)
        if indices.size:
            by_index[int(indices[0])] = deepcopy(group)
    groups = []
    for channel_index in selected:
        eeglab_index = int(channel_index + 1)
        group = by_index.get(eeglab_index)
        if group is None:
            label = labels[int(channel_index)]
            group = {
                "name": label,
                "channels": [label],
                "sets": list(range(1, len(study.get("datasetinfo") or []) + 1)),
                "inds": [eeglab_index],
                "measureinfo": {
                    "kind": "channels",
                    "datasets": list(range(1, len(study.get("datasetinfo") or []) + 1)),
                    "channels": [label],
                    "computed": [],
                },
            }
        groups.append(group)
    return groups


def _replace_channel_groups(study: dict[str, Any], updated: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = [deepcopy(group) for group in study.get("changrp") or [] if isinstance(group, dict)]
    lookup = {
        int(numeric_vector(group.get("inds"), dtype=int)[0]): index
        for index, group in enumerate(groups)
        if numeric_vector(group.get("inds"), dtype=int).size
    }
    for group in updated:
        indices = numeric_vector(group.get("inds"), dtype=int)
        if indices.size == 0:
            groups.append(group)
            continue
        key = int(indices[0])
        if key in lookup:
            groups[lookup[key]] = group
        else:
            groups.append(group)
    return groups


def _eeg_tlimits(EEG: dict[str, Any]) -> list[float]:
    return [float(EEG.get("xmin", 0.0) or 0.0) * 1000.0, float(EEG.get("xmax", 0.0) or 0.0) * 1000.0]


def _looks_like_study(value: dict[str, Any]) -> bool:
    return isinstance(value, dict) and ("datasetinfo" in value or "design" in value or "changrp" in value)


def _dataset_history_command(**kwargs: Any) -> str:
    pac_options = kwargs.pop("pac_options")
    kwargs.update(pac_options)
    return build_python_call(
        ("PACDATA", "PACTIMES", "PACFREQS", "PARAMETERS"),
        "std_pac",
        "EEG",
        **kwargs,
    )


__all__ = ["std_pac"]
