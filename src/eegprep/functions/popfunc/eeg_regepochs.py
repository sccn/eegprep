"""Split continuous EEG into regularly recurring epochs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from eegprep.functions.miscfunc.value_parsing import parse_key_value_args
from eegprep.functions.popfunc.pop_epoch import pop_epoch
from eegprep.functions.popfunc.pop_mergeset import pop_mergeset
from eegprep.functions.popfunc.pop_rmbase import pop_rmbase


_DEFAULTS = {
    "recurrence": 1.0,
    "limits": None,
    "rmbase": 0.0,
    "eventtype": "X",
    "eventdata": None,
    "extractepochs": "on",
}


def eeg_regepochs(EEG: dict[str, Any] | list[dict[str, Any]], *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Insert regular events and optionally extract consecutive epochs.

    Event latencies remain EEGLAB-compatible one-based sample positions;
    ``urevent`` pointers stored in EEGPrep dictionaries are zero-based. Legacy
    positional ``recurrence, limits, rmbase`` arguments and EEGLAB-style
    key/value pairs are both supported.
    """
    dataset = _dataset(EEG)
    options = _options(args, kwargs)
    recurrence = float(options["recurrence"])
    sampling_rate = float(dataset.get("srate", 0))
    data = np.asarray(dataset.get("data"))
    inferred_points = data.shape[1] if data.ndim in {2, 3} else 0
    points = int(dataset.get("pnts", inferred_points))
    if sampling_rate <= 0 or points < 1:
        raise ValueError("eeg_regepochs: EEG.srate and EEG.pnts must be positive")
    if int(dataset.get("trials", 1)) != 1:
        raise ValueError("eeg_regepochs: input dataset must be continuous")
    if recurrence <= 0:
        raise ValueError("eeg_regepochs: recurrence must be positive")

    duration = float(dataset.get("xmax", (points - 1) / sampling_rate)) + 1 / sampling_rate
    event_count = int(np.floor(duration / recurrence + np.finfo(float).eps * 8))
    if event_count < 1:
        raise ValueError("eeg_regepochs: recurrence is longer than the recording")
    limits = _limits(options["limits"], recurrence)
    event_type = str(options["eventtype"])
    extra = _event_data(options["eventdata"])

    events = _records(dataset.get("event"))
    urevents = _records(dataset.get("urevent"))
    for event in events:
        if "type" in event:
            event["type"] = str(event["type"])
    if events and not urevents:
        urevents = [
            {key: deepcopy(value) for key, value in event.items() if key not in {"epoch", "urevent"}}
            for event in events
        ]
        for index, event in enumerate(events):
            event["urevent"] = index
    for index in range(event_count):
        latency = recurrence * index * sampling_rate + 1
        urevent = {"type": event_type, "latency": latency, **deepcopy(extra)}
        event = {**deepcopy(urevent), "urevent": len(urevents)}
        urevents.append(urevent)
        events.append(event)
    events.sort(key=lambda event: float(event.get("latency", np.inf)))
    dataset["event"] = events
    dataset["urevent"] = urevents

    extract = str(options["extractepochs"]).casefold()
    if extract == "off":
        dataset["saved"] = "no"
        return dataset
    if extract != "on":
        raise ValueError("eeg_regepochs: extractepochs must be 'on' or 'off'")

    setname = str(dataset.get("setname", ""))
    newname = f"{setname} - {recurrence:g}-s epochs" if setname else f"{recurrence:g}-s epochs"
    epoched, _indices = pop_epoch(
        dataset,
        [event_type],
        limits,
        "newname",
        newname,
        "epochinfo",
        "yes",
    )
    baseline_end = float(options["rmbase"])
    if not np.isnan(baseline_end) and limits[0] < baseline_end:
        epoched = pop_rmbase(epoched, [limits[0] * 1000, baseline_end * 1000], gui=False)
    return epoched


def _dataset(EEG: dict[str, Any] | list[dict[str, Any]]) -> dict[str, Any]:
    if isinstance(EEG, list):
        if not EEG:
            raise ValueError("eeg_regepochs: EEG must not be empty")
        if len(EEG) == 1:
            return deepcopy(EEG[0])
        return pop_mergeset(EEG, list(range(1, len(EEG) + 1)), gui=False)
    if not isinstance(EEG, dict) or "event" not in EEG:
        raise ValueError("eeg_regepochs: EEG must be a dataset dictionary with an event table")
    return deepcopy(EEG)


def _options(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    if args and not isinstance(args[0], (str, bytes)):
        if len(args) > 3:
            raise ValueError("eeg_regepochs: positional form accepts recurrence, limits, and rmbase")
        names = ("recurrence", "limits", "rmbase")
        supplied = dict(zip(names, args))
        supplied.update({str(key).lower(): value for key, value in kwargs.items()})
    else:
        supplied = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    unknown = sorted(set(supplied) - set(_DEFAULTS))
    if unknown:
        raise ValueError(f"eeg_regepochs: unsupported option(s): {', '.join(unknown)}")
    return {**_DEFAULTS, **supplied}


def _limits(value: Any, recurrence: float) -> list[float]:
    limits = np.asarray([0, recurrence] if value is None else value, dtype=float).reshape(-1)
    if limits.size != 2 or not np.isfinite(limits).all() or limits[0] >= limits[1]:
        raise ValueError("eeg_regepochs: limits must be an increasing two-value vector")
    return limits.tolist()


def _event_data(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return deepcopy(value)
    values = list(value)
    if len(values) % 2 or not all(isinstance(values[index], str) for index in range(0, len(values), 2)):
        raise ValueError("eeg_regepochs: eventdata must be a mapping or key/value sequence")
    return dict(zip(values[::2], values[1::2]))


def _records(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(record, dict) for record in value):
        raise ValueError("eeg_regepochs: event tables must contain dictionaries")
    return [deepcopy(record) for record in value]


__all__ = ["eeg_regepochs"]
