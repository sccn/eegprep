"""Import event information into an EEGPrep EEG dataset."""

from __future__ import annotations

from copy import deepcopy
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import fmin

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset, strict_mode
from eegprep.functions.popfunc._file_io import events_to_records, read_table_records
from eegprep.functions.popfunc._pop_utils import format_history_value, parse_key_value_args, parse_text_tokens


logger = logging.getLogger(__name__)


def pop_importevent(
    EEG: dict[str, Any],
    *args: Any,
    return_com: bool = False,
    **kwargs: Any,
) -> dict[str, Any] | tuple[dict[str, Any], str]:
    """Import events from a text table or record array.

    Event latencies are converted to 1-based sample positions. Existing events
    are appended by default, matching EEGLAB; pass ``append='no'`` to replace
    them. ``indices`` contains EEGLAB-facing 1-based event indices to update.
    """
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    event_source = options.get("event", options.get("filename"))
    if event_source is None:
        raise ValueError("pop_importevent requires an event file or event records")
    fields = _fields(options.get("fields"))
    records = _load_records(event_source, fields, options)
    old_events = events_to_records(EEG.get("event"))
    indices = _event_indices(options.get("indices"), len(old_events))
    append = _is_on(options.get("append", "yes")) or bool(indices)
    latency_present = any("latency" in record for record in records)
    if not latency_present and any("latency" in event for event in old_events):
        append = False

    align = _align_value(options)
    if not math.isnan(align):
        _validate_alignment(align, old_events, len(records))
        if append:
            logger.warning("pop_importevent: alignment and append were both requested; applying EEGLAB behavior")

    events, imported_indices = _merge_records(old_events, records, append=append, indices=indices)
    _recompute_latencies(
        events,
        imported_indices,
        old_events,
        srate=float(EEG.get("srate", 1) or 1),
        timeunit=_timeunit(options),
        align=align,
        optimalign=_is_on(options.get("optimalign", "on")),
        optimoffset=_is_on(options.get("optimoffset", "off")),
        optimmeas=str(options.get("optimmeas", "mean")).lower(),
    )

    out = deepcopy(EEG)
    out["event"] = events
    out["saved"] = "no"
    with strict_mode(False):
        out = eeg_checkset(out, "eventconsistency")
    out["event"], out["urevent"] = _rebuild_urevents(events_to_records(out["event"]))
    with strict_mode(False):
        out = eeg_checkset(out)
    command = _history_command(event_source, options)
    out["history"] = _append_history(out.get("history", ""), command)
    return (out, command) if return_com else out


def _load_records(event_source: Any, fields: list[str] | None, options: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(event_source, (str, bytes, Path)):
        return read_table_records(
            str(event_source),
            fields=fields,
            skipline=int(options.get("skipline", 0) or 0),
            delimiter=options.get("delim"),
        )
    if isinstance(event_source, np.ndarray):
        event_source = event_source.tolist()
    if isinstance(event_source, dict):
        return [dict(event_source)]
    rows = list(event_source)
    if not rows:
        return []
    if all(isinstance(record, dict) for record in rows):
        return [dict(record) for record in rows]
    if not fields:
        raise ValueError("pop_importevent requires fields for event arrays")
    records = []
    for row in rows:
        values = np.asarray(row, dtype=object).ravel().tolist()
        if len(values) != len(fields):
            raise ValueError("All event rows must have the same number of values as fields")
        records.append(dict(zip(fields, values)))
    return records


def _event_indices(value: Any, count: int) -> list[int]:
    if value is None:
        return []
    values = np.asarray(value).ravel().tolist()
    indices = []
    for item in values:
        index = int(item)
        if index < 1 or index > count:
            raise ValueError("event indices must be 1-based and within EEG.event")
        indices.append(index - 1)
    return indices


def _merge_records(
    old_events: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    append: bool,
    indices: list[int],
) -> tuple[list[dict[str, Any]], list[int]]:
    if indices:
        if len(indices) != len(records):
            raise ValueError("The number of event rows must match the number of indices")
        events = deepcopy(old_events)
        for event_index, record in zip(indices, records):
            events[event_index].update(deepcopy(record))
        return events, indices
    if append:
        events = deepcopy(old_events) + deepcopy(records)
        return events, list(range(len(old_events), len(events)))
    if old_events and len(records) == len(old_events):
        events = deepcopy(old_events)
        for event, record in zip(events, records):
            event.update(deepcopy(record))
        return events, list(range(len(events)))
    return deepcopy(records), list(range(len(records)))


def _recompute_latencies(
    events: list[dict[str, Any]],
    indices: list[int],
    old_events: list[dict[str, Any]],
    *,
    srate: float,
    timeunit: float,
    align: float,
    optimalign: bool,
    optimoffset: bool,
    optimmeas: str,
) -> None:
    if not indices or not any("latency" in events[index] for index in indices):
        if any("duration" in events[index] for index in indices):
            raise ValueError("A duration field cannot be defined without a latency field")
        return
    if optimmeas not in {"mean", "median"}:
        raise ValueError("optimmeas must be 'mean' or 'median'")

    numeric_timeunit = not math.isnan(timeunit)
    for initial_index, event_index in enumerate(indices, start=1):
        event = events[event_index]
        if "latency" not in event:
            continue
        latency = float(event["latency"])
        if numeric_timeunit:
            event["init_index"] = initial_index
            event["init_time"] = latency * timeunit
            event["latency"] = latency * srate * timeunit
            if "duration" in event:
                event["duration"] = float(event["duration"]) * srate * timeunit
        else:
            event["latency"] = latency

    if not math.isnan(align):
        event_anchor_index = 0 if align >= 0 else int(-align)
        imported_anchor = float(events[event_anchor_index]["latency"])
        old_anchor = float(old_events[int(align) if align >= 0 else 0]["latency"])
        for event_index in indices:
            events[event_index]["latency"] = float(events[event_index]["latency"]) - imported_anchor + old_anchor

    scale = 1.0
    offset = 0.0
    if optimalign and not math.isnan(align):
        scale, offset = _optimal_alignment(events, old_events, align, optimmeas, optimoffset)
    if not 0.99 <= scale <= 1.01:
        scale = 1.0

    if not math.isnan(align) and scale != 1.0:
        event_anchor_index = 0 if align >= 0 else int(-align)
        anchor = float(events[event_anchor_index]["latency"])
        for event_index in indices:
            latency = float(events[event_index]["latency"])
            events[event_index]["latency"] = (latency - anchor) * scale + anchor + offset
    elif numeric_timeunit:
        for event_index in indices:
            latency = float(events[event_index]["latency"])
            events[event_index]["latency"] = round((latency + 1) * 1000 * scale + offset) / 1000


def _optimal_alignment(
    events: list[dict[str, Any]],
    old_events: list[dict[str, Any]],
    align: float,
    measure: str,
    optimize_offset: bool,
) -> tuple[float, float]:
    new_latencies = np.asarray([float(event["latency"]) for event in events], dtype=float)
    old_latencies = np.asarray([float(event["latency"]) for event in old_events], dtype=float)
    new_anchor = new_latencies[0]
    old_anchor = old_latencies[int(align)] if align >= 0 else old_latencies[0]
    new_relative = new_latencies - new_anchor
    old_relative = old_latencies - old_anchor

    def objective(parameters: np.ndarray) -> float:
        scale = float(parameters[0])
        offset = float(parameters[1]) if parameters.size == 2 else 0.0
        differences = np.abs(scale * new_relative[:, np.newaxis] - old_relative[np.newaxis, :] + offset)
        nearest = differences.min(axis=0)
        return float(np.median(nearest) if measure == "median" else np.mean(nearest))

    initial = np.asarray([1.0, 0.0] if optimize_offset else [1.0])
    parameters = fmin(objective, initial, disp=False)
    parameters = fmin(objective, parameters, disp=False)
    return float(parameters[0]), float(parameters[1]) if parameters.size == 2 else 0.0


def _validate_alignment(align: float, old_events: list[dict[str, Any]], imported_count: int) -> None:
    if not old_events:
        raise ValueError("Cannot align imported events without pre-existing events")
    if any("latency" not in event for event in old_events):
        raise ValueError("Pre-existing events must have latency values for alignment")
    if align >= 0 and int(align) >= len(old_events):
        raise ValueError("align refers to a pre-existing event outside EEG.event")
    if align < 0 and int(-align) >= imported_count:
        raise ValueError("negative align refers to an imported event outside the event table")


def _align_value(options: dict[str, Any]) -> float:
    value = options.get("align", math.nan)
    align = float(value)
    if not math.isnan(align) and not align.is_integer():
        raise ValueError("align must be an integer or NaN")
    return align


def _is_on(value: Any) -> bool:
    return str(value).lower() in {"on", "yes", "true", "1"}


def _fields(value: Any) -> list[str] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        return [str(item) for item in parse_text_tokens(value)]
    return [str(item) for item in value]


def _timeunit(options: dict[str, Any]) -> float:
    if "timeunit" not in options:
        return 1.0
    value = options["timeunit"]
    if isinstance(value, str) and value.lower() == "nan":
        return float("nan")
    return float(value)


def _rebuild_urevents(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized_events = []
    urevents = []
    for index, event in enumerate(events):
        urevent = _urevent_record(event)
        event_with_ref = dict(urevent)
        event_with_ref["urevent"] = index
        normalized_events.append(event_with_ref)
        urevents.append(urevent)
    return normalized_events, urevents


def _urevent_record(event: dict[str, Any]) -> dict[str, Any]:
    record = dict(event)
    record.pop("urevent", None)
    return record


def _history_command(event_source: Any, options: dict[str, Any]) -> str:
    pieces = [format_history_value("event"), format_history_value(event_source)]
    for key in [
        "fields",
        "append",
        "skipline",
        "timeunit",
        "delim",
        "indices",
        "align",
        "optimalign",
        "optimoffset",
        "optimmeas",
    ]:
        if key in options:
            pieces.extend([format_history_value(key), format_history_value(options[key])])
    return f"EEG = pop_importevent(EEG, {', '.join(pieces)});"


def _append_history(history: str, command: str) -> str:
    return command if not history else f"{history.rstrip()}\n{command}"
