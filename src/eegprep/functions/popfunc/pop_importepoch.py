"""Import epoch metadata into an EEGPrep EEG dataset."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset, strict_mode
from eegprep.functions.popfunc._file_io import events_to_records, read_table_records
from eegprep.functions.popfunc._pop_utils import format_history_value, parse_key_value_args, parse_text_tokens


def pop_importepoch(
    EEG: dict[str, Any],
    filename: Any = None,
    fieldlist: list[str] | tuple[str, ...] | str | None = None,
    *args: Any,
    return_com: bool = False,
    **kwargs: Any,
) -> dict[str, Any] | tuple[dict[str, Any], str]:
    """Import per-epoch metadata and construct its corresponding events.

    Each input row describes one epoch. ``latencyfields`` create additional
    events at epoch-relative times, while ``typefield`` names the time-locking
    event at zero. Latencies and durations are stored in samples.
    """
    if filename is None:
        filename = kwargs.pop("filename", None)
    if filename is None:
        raise ValueError("pop_importepoch requires an epoch info file")
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    fields = _tokens(fieldlist)
    records = _load_records(filename, fields, headerlines=_scalar_int(options.get("headerlines", 0)))
    if fields is None:
        fields = list(records[0]) if records else []
    trials = int(EEG.get("trials", 1) or 1)
    if trials <= 1:
        raise ValueError("pop_importepoch requires an epoched dataset")
    if len(records) != trials:
        raise ValueError("The number of imported epoch rows must match EEG.trials")

    latency_fields = _tokens(options.get("latencyfields")) or []
    duration_fields = _duration_fields(options.get("durationfields"), len(latency_fields))
    type_field = str(options.get("typefield", "") or "")
    _validate_fields(fields, latency_fields, duration_fields, type_field)

    out = deepcopy(EEG)
    out["epoch"] = records
    out["saved"] = "no"
    old_events = events_to_records(out.get("event"))
    clear_events = _is_on(options.get("clearevents", "on")) or any("epoch" not in event for event in old_events)
    events = [] if clear_events else old_events
    srate = float(out.get("srate", 1) or 1)
    pnts = int(out.get("pnts", 0) or 0)
    xmin = float(out.get("xmin", 0) or 0)
    timeunit = float(options.get("timeunit", 1 / srate))

    if xmin <= 0:
        for trial, record in enumerate(records, start=1):
            events.append(
                {
                    "epoch": trial,
                    "type": record[type_field] if type_field else "TLE",
                    "latency": -xmin * srate + 1 + (trial - 1) * pnts,
                    "duration": 0,
                }
            )

    for latency_field, duration_field in zip(latency_fields, duration_fields):
        for trial, record in enumerate(records, start=1):
            events.append(
                {
                    "epoch": trial,
                    "type": latency_field,
                    "latency": (float(record[latency_field]) * timeunit - xmin) * srate + 1 + (trial - 1) * pnts,
                    "duration": _duration(record, duration_field, timeunit=timeunit, srate=srate),
                }
            )

    if not events:
        events = [{"epoch": trial} for trial in range(1, trials + 1)]
    other_fields = _other_fields(fields, latency_fields, duration_fields, type_field)
    for event in events:
        epoch_index = int(event["epoch"]) - 1
        for field in other_fields:
            event["epoch" + field if field in {"type", "latency"} else field] = records[epoch_index][field]

    out["event"] = events
    if _is_empty(out.get("eventdescription")):
        out["eventdescription"] = _event_descriptions(events)
    with strict_mode(False):
        out = eeg_checkset(out, "eventconsistency")
    out["event"], out["urevent"] = _rebuild_urevents(events_to_records(out["event"]))
    with strict_mode(False):
        out = eeg_checkset(out)
    command = _history_command(filename, fields, options)
    out["history"] = command if not out.get("history") else f"{out['history'].rstrip()}\n{command}"
    return (out, command) if return_com else out


def _load_records(source: Any, fields: list[str] | None, *, headerlines: int) -> list[dict[str, Any]]:
    if isinstance(source, (str, bytes, Path)):
        return read_table_records(source, fields=fields, skipline=headerlines)
    if isinstance(source, np.ndarray):
        source = source.tolist()
    if isinstance(source, dict):
        return [dict(source)]
    rows = list(source)
    if not rows:
        return []
    if all(isinstance(row, dict) for row in rows):
        return [dict(row) for row in rows]
    if not fields:
        raise ValueError("pop_importepoch requires field names for epoch arrays")
    records = []
    for row in rows:
        values = np.asarray(row, dtype=object).ravel().tolist()
        if len(values) != len(fields):
            raise ValueError("There must be as many field names as columns in the epoch array")
        records.append(dict(zip(fields, values)))
    return records


def _duration_fields(value: Any, count: int) -> list[Any]:
    if value is None or (isinstance(value, str) and not value):
        return [0] * count
    if isinstance(value, str):
        values = parse_text_tokens(value)
    else:
        values = np.asarray(value, dtype=object).ravel().tolist()
    if len(values) != count:
        raise ValueError("There must be one duration field for each latency field")
    return values


def _validate_fields(
    fields: list[str],
    latency_fields: list[str],
    duration_fields: list[Any],
    type_field: str,
) -> None:
    missing = [field for field in latency_fields if field not in fields]
    missing.extend(str(field) for field in duration_fields if field not in {0, "0"} and str(field) not in fields)
    if type_field and type_field not in fields:
        missing.append(type_field)
    if missing:
        raise ValueError(f"Epoch field(s) not found: {', '.join(missing)}")


def _duration(record: dict[str, Any], field: Any, *, timeunit: float, srate: float) -> float:
    if field in {0, "0", None, ""}:
        return 0.0
    return float(record[str(field)]) * timeunit * srate


def _other_fields(
    fields: list[str],
    latency_fields: list[str],
    duration_fields: list[Any],
    type_field: str,
) -> list[str]:
    excluded = set(latency_fields)
    excluded.update(str(field) for field in duration_fields if field not in {0, "0"})
    if type_field:
        excluded.add(type_field)
    return [field for field in fields if field not in excluded]


def _event_descriptions(events: list[dict[str, Any]]) -> list[str]:
    descriptions = {
        "epoch": "Epoch number",
        "type": "Event type",
        "latency": "Event latency",
        "duration": "Event duration",
    }
    fields = []
    for event in events:
        for field in event:
            if field not in fields:
                fields.append(field)
    return [descriptions.get(field, "") for field in fields]


def _rebuild_urevents(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized_events = []
    urevents = []
    for index, event in enumerate(events):
        urevent = dict(event)
        urevent.pop("urevent", None)
        normalized = dict(urevent)
        normalized["urevent"] = index
        normalized_events.append(normalized)
        urevents.append(urevent)
    return normalized_events, urevents


def _scalar_int(value: Any) -> int:
    values = np.asarray(value).ravel()
    if values.size != 1:
        raise ValueError("headerlines must be a scalar")
    return int(values[0])


def _is_on(value: Any) -> bool:
    return str(value).lower() in {"on", "yes", "true", "1"}


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, np.ndarray):
        return value.size == 0
    return not value


def _tokens(value: Any) -> list[str] | None:
    if value is None or (isinstance(value, str) and not value):
        return None
    if isinstance(value, str):
        return [str(item) for item in parse_text_tokens(value)]
    return [str(item) for item in value]


def _history_command(filename: Any, fields: list[str] | None, options: dict[str, Any]) -> str:
    pieces = [format_history_value(filename), format_history_value(fields or [])]
    for key in ["latencyfields", "durationfields", "typefield", "timeunit", "headerlines", "clearevents"]:
        if key in options:
            pieces.extend([format_history_value(key), format_history_value(options[key])])
    return f"EEG = pop_importepoch(EEG, {', '.join(pieces)});"
