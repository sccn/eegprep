"""Extract event values on a per-epoch basis."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.popfunc._event_utils import events_as_list
from eegprep.functions.popfunc.eeg_point2lat import eeg_point2lat


_OPTION_NAMES = {"type", "timewin", "fieldname", "trials"}


def eeg_getepochevent(
    EEG: dict[str, Any] | list[dict[str, Any]], *args: Any, **kwargs: Any
) -> tuple[Any, list[list[Any]]]:
    """Return a selected event-field value for each epoch.

    Both EEGLAB's positional form ``(type, timewin, fieldname)`` and its
    key/value form are accepted. Trial selections are 1-based; stored event
    fields such as ``urevent`` are returned in EEGPrep's 0-based form.
    """
    options = _parse_options(args, kwargs)
    if isinstance(EEG, list):
        return _multiple_datasets(EEG, options)
    event_types = options["type"]
    time_window = np.asarray(options["timewin"], dtype=float)
    field_name = str(options["fieldname"])
    events = events_as_list(EEG.get("event", []))
    trial_count = int(EEG.get("trials", 1))
    epoch_values: list[Any] = [np.nan] * trial_count
    all_epoch_values: list[list[Any]] = [[] for _ in range(trial_count)]

    for event in events:
        if not _matches_type(event.get("type"), event_types) or field_name not in event:
            continue
        epoch = _event_epoch(event)
        latency_ms = float(
            eeg_point2lat(
                [event["latency"]],
                [epoch],
                EEG["srate"],
                [EEG["xmin"] * 1000, EEG["xmax"] * 1000],
                1e-3,
            )[0]
        )
        if latency_ms < time_window[0] or latency_ms > time_window[1]:
            continue
        value = _field_value(event[field_name], field_name, float(EEG["srate"]), latency_ms)
        if value is None:
            continue
        index = epoch - 1
        if index < 0 or index >= trial_count:
            raise ValueError("event epoch is outside EEG.trials")
        if field_name != "latency" or not all_epoch_values[index]:
            epoch_values[index] = _numeric_value(value)
        all_epoch_values[index].append(value)

    selected_trials = options["trials"]
    if selected_trials is not None and np.asarray(selected_trials).size:
        indices = (np.asarray(selected_trials, dtype=int).ravel() - 1).tolist()
        if any(index < 0 or index >= trial_count for index in indices):
            raise ValueError("trials must be 1-based and within EEG.trials")
        epoch_values = [epoch_values[index] for index in indices]
        all_epoch_values = [all_epoch_values[index] for index in indices]
    try:
        output_values: Any = np.asarray(epoch_values, dtype=float)
    except (TypeError, ValueError):
        output_values = epoch_values
    return output_values, all_epoch_values


def _multiple_datasets(datasets: list[dict[str, Any]], options: dict[str, Any]) -> tuple[Any, list[list[Any]]]:
    trial_groups = _dataset_trial_groups(options["trials"], len(datasets))
    combined_values: list[Any] = []
    combined_all_values: list[list[Any]] = []
    for dataset, trials in zip(datasets, trial_groups):
        dataset_options = {**options, "trials": trials}
        values, all_values = eeg_getepochevent(dataset, **dataset_options)
        combined_values.extend(np.asarray(values).ravel().tolist())
        combined_all_values.extend(all_values)
    try:
        output_values: Any = np.asarray(combined_values, dtype=float)
    except (TypeError, ValueError):
        output_values = combined_values
    return output_values, combined_all_values


def _dataset_trial_groups(trials: Any, dataset_count: int) -> list[Any]:
    if trials is None or np.asarray(trials, dtype=object).size == 0:
        return [[] for _ in range(dataset_count)]
    if dataset_count == 1:
        return [trials]
    if isinstance(trials, (str, bytes)) or np.isscalar(trials):
        raise ValueError("trials must provide one selection per dataset")
    groups = list(trials)
    if len(groups) != dataset_count:
        raise ValueError("trials must provide one selection per dataset")
    return groups


def _parse_options(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {"type": [], "timewin": [-np.inf, np.inf], "fieldname": "latency", "trials": []}
    legacy_time_window = len(args) > 1 and _is_numeric_pair(args[1])
    if (
        not legacy_time_window
        and args
        and isinstance(args[0], str)
        and args[0].lower() in _OPTION_NAMES
        and len(args) % 2 == 0
    ):
        for key, value in zip(args[::2], args[1::2]):
            options[str(key).lower()] = value
    elif args:
        options["type"] = args[0]
        if len(args) > 1 and np.asarray(args[1]).size:
            options["timewin"] = args[1]
        if len(args) > 2:
            options["fieldname"] = args[2]
        if len(args) > 3:
            raise TypeError("old eeg_getepochevent form accepts at most three options")
    options.update({str(key).lower(): value for key, value in kwargs.items()})
    unknown = set(options) - _OPTION_NAMES
    if unknown:
        raise TypeError(f"unknown eeg_getepochevent options: {sorted(unknown)}")
    if not np.asarray(options["timewin"]).size:
        options["timewin"] = [-np.inf, np.inf]
    return options


def _is_numeric_pair(value: Any) -> bool:
    array = np.asarray(value)
    return array.size == 2 and np.issubdtype(array.dtype, np.number)


def _matches_type(value: Any, requested: Any) -> bool:
    if requested is None or (isinstance(requested, (list, tuple, np.ndarray)) and len(requested) == 0):
        return True
    requested_values = [requested] if isinstance(requested, (str, int, float)) else list(requested)
    return any(_type_text(value) == _type_text(candidate) for candidate in requested_values)


def _event_epoch(event: dict[str, Any]) -> int:
    value = event.get("epoch", 1)
    if isinstance(value, np.ndarray):
        value = value.ravel()[0]
    elif isinstance(value, (list, tuple)):
        value = value[0]
    return int(value)


def _field_value(value: Any, field: str, srate: float, latency_ms: float) -> Any:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.ravel()[0]
    elif isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    if field == "latency":
        return latency_ms
    if field == "duration":
        return float(value) / srate * 1000
    return value


def _numeric_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    output = 0.0
    for position, character in enumerate(value, start=1):
        code = ord(character)
        adjusted = (
            code - 47
            if 48 <= code <= 57
            else code - 64
            if 65 <= code <= 90
            else code - 96
            if 97 <= code <= 122
            else code
        )
        output += adjusted / 36**position
    return output


def _type_text(value: Any) -> str:
    if isinstance(value, (int, float, np.integer, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


__all__ = ["eeg_getepochevent"]
