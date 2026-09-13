"""Extract event latencies from one or more data channels."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset, strict_mode
from eegprep.functions.popfunc._file_io import events_to_records
from eegprep.functions.popfunc._pop_utils import format_history_value, parse_key_value_args


def pop_chanevent(
    EEG: dict[str, Any],
    chan: int | list[int] | tuple[int, ...],
    *args: Any,
    return_com: bool = False,
    **kwargs: Any,
) -> dict[str, Any] | tuple[dict[str, Any], str]:
    """Import events from rising, falling, or both edges of data channels."""
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    edge = str(options.get("edge", "both")).lower()
    if edge not in {"both", "leading", "trailing"}:
        raise ValueError("edge must be 'both', 'leading', or 'trailing'")
    duration = str(options.get("duration", "off")).lower() in {"on", "yes", "true", "1"}
    if duration and edge != "leading":
        raise ValueError("duration extraction requires leading edges")
    channels = [int(chan)] if isinstance(chan, int) else [int(item) for item in chan]
    data = np.asarray(EEG["data"])
    if data.ndim != 2:
        raise ValueError("pop_chanevent currently supports continuous 2-D data")
    if any(channel < 1 or channel > data.shape[0] for channel in channels):
        raise ValueError("chan indices must be 1-based and within EEG.nbchan")
    events = []
    for channel in channels:
        x = data[channel - 1, :]
        nbtype = options.get("nbtype", np.nan)
        named_type = not _is_nan(nbtype) or np.unique(x).size == 2
        if "oper" in options and options["oper"]:
            x = _apply_oper(x, str(options["oper"]))
        events.extend(
            _events_from_channel(
                x,
                edge=edge,
                duration=duration,
                edgelen=int(options.get("edgelen", 1)),
                typename=str(options.get("typename", f"chan{channel}")),
                named_type=named_type,
            )
        )
    out = deepcopy(EEG)
    delete_events = str(options.get("delevent", "on")).lower() in {"on", "yes", "true", "1"}
    if delete_events:
        imported_events, imported_urevents = _events_with_new_urevents(events, 0)
        out["event"] = imported_events
        out["urevent"] = imported_urevents
    else:
        original_events, original_urevents = _events_with_existing_urevents(
            events_to_records(out.get("event")),
            events_to_records(out.get("urevent")),
        )
        imported_events, imported_urevents = _events_with_new_urevents(events, len(original_urevents))
        out["event"] = original_events + imported_events
        out["event"].sort(key=lambda item: float(item.get("latency", np.inf)))
        out["urevent"] = original_urevents + imported_urevents
    if str(options.get("delchan", "on")).lower() in {"on", "yes", "true", "1"}:
        keep = [index for index in range(data.shape[0]) if index + 1 not in channels]
        out["data"] = data[keep, :]
        out["nbchan"] = len(keep)
        out["chanlocs"] = [
            loc for index, loc in enumerate(list(out.get("chanlocs", [])), start=1) if index not in channels
        ]
    out["saved"] = "no"
    with strict_mode(False):
        out = eeg_checkset(out)
    command = _history_command(channels, options)
    out["history"] = command if not out.get("history") else f"{out['history'].rstrip()}\n{command}"
    return (out, command) if return_com else out


def _events_from_channel(
    x: np.ndarray,
    *,
    edge: str,
    duration: bool,
    edgelen: int,
    typename: str,
    named_type: bool,
) -> list[dict[str, Any]]:
    values = np.asarray(x)
    diff = np.diff(np.abs(np.r_[values, values[-1]]))
    leading = _keep_first_of_close(np.flatnonzero(diff > 0) + 1, edgelen)
    trailing = _keep_last_of_close(np.flatnonzero(diff < 0) + 2, edgelen)
    if edge == "leading":
        latencies = leading
    elif edge == "trailing":
        latencies = trailing
    else:
        latencies = np.sort(np.r_[leading, trailing])
    events = []
    for latency in latencies:
        if edge == "both":
            event_edge = "trailing" if latency in trailing else "leading"
        else:
            event_edge = edge
        event_type = typename if named_type else _event_value(values, int(latency), event_edge)
        event = {"type": event_type, "latency": int(latency)}
        if duration:
            falling_boundaries = trailing - 1
            next_trailing = falling_boundaries[falling_boundaries >= latency]
            event["duration"] = int(next_trailing[0] - latency) if next_trailing.size else int(values.size - latency)
        events.append(event)
    return events


def _events_with_existing_urevents(
    events: list[dict[str, Any]],
    urevents: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized_urevents = [dict(event) for event in urevents]
    normalized_events = []
    for event in events:
        normalized = dict(event)
        urevent_index = _valid_urevent_index(normalized.get("urevent"), len(normalized_urevents))
        if urevent_index is None:
            urevent_index = len(normalized_urevents)
            normalized_urevents.append(_urevent_record(normalized))
        normalized["urevent"] = urevent_index
        normalized_events.append(normalized)
    return normalized_events, normalized_urevents


def _events_with_new_urevents(
    events: list[dict[str, Any]],
    offset: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized_events = []
    urevents = []
    for index, event in enumerate(events):
        urevent = _urevent_record(event)
        event_with_ref = dict(urevent)
        event_with_ref["urevent"] = offset + index
        normalized_events.append(event_with_ref)
        urevents.append(urevent)
    return normalized_events, urevents


def _valid_urevent_index(value: Any, count: int) -> int | None:
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    return index if 0 <= index < count else None


def _urevent_record(event: dict[str, Any]) -> dict[str, Any]:
    record = dict(event)
    record.pop("urevent", None)
    return record


def _keep_first_of_close(values: np.ndarray, edgelen: int) -> np.ndarray:
    if values.size < 2 or edgelen <= 1:
        return values
    return values[np.r_[True, np.diff(values) >= edgelen]]


def _keep_last_of_close(values: np.ndarray, edgelen: int) -> np.ndarray:
    if values.size < 2 or edgelen <= 1:
        return values
    return values[np.r_[np.diff(values) >= edgelen, True]]


def _event_value(values: np.ndarray, latency: int, edge: str) -> Any:
    if edge == "leading":
        return values[latency].item()
    return values[latency - 2].item()


def _is_nan(value: Any) -> bool:
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


def _apply_oper(x: np.ndarray, oper: str) -> np.ndarray:
    if oper.strip() == "X>0":
        return x > 0
    if oper.strip().startswith("X>"):
        return x > float(oper.strip()[2:])
    if oper.strip().startswith("X<"):
        return x < float(oper.strip()[2:])
    raise ValueError("Only simple X>threshold or X<threshold preprocessing expressions are supported")


def _history_command(channels: list[int], options: dict[str, Any]) -> str:
    channel_value: int | list[int] = channels[0] if len(channels) == 1 else channels
    pieces = [format_history_value(channel_value)]
    for key in ["oper", "edge", "edgelen", "duration", "delchan", "delevent"]:
        if key in options:
            pieces.extend([format_history_value(key), format_history_value(options[key])])
    return f"EEG = pop_chanevent(EEG, {', '.join(pieces)});"
