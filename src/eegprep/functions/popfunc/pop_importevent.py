"""Import event information into an EEGPrep EEG dataset."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset, strict_mode
from eegprep.functions.popfunc._file_io import events_to_records, read_table_records, records_to_events
from eegprep.functions.popfunc._pop_utils import format_history_value, parse_key_value_args, parse_text_tokens


def pop_importevent(
    EEG: dict[str, Any],
    *args: Any,
    return_com: bool = False,
    **kwargs: Any,
) -> dict[str, Any] | tuple[dict[str, Any], str]:
    """Import events from a text table or record array."""
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    event_source = options.get("event", options.get("filename"))
    if event_source is None:
        raise ValueError("pop_importevent requires an event file or event records")
    fields = _fields(options.get("fields"))
    if isinstance(event_source, (str, bytes, Path)):
        records = read_table_records(str(event_source), fields=fields, skipline=int(options.get("skipline", 0) or 0))
    else:
        records = [dict(record) for record in event_source]
    events = records_to_events(records, srate=float(EEG.get("srate", 1) or 1), timeunit=_timeunit(options))
    append = str(options.get("append", "no")).lower() in {"on", "yes", "true", "1"}
    out = deepcopy(EEG)
    if append:
        events = events_to_records(out.get("event")) + events
        events.sort(key=lambda item: float(item.get("latency", np.inf)))
    out["event"] = events
    out["urevent"] = [dict(event) for event in events]
    out["saved"] = "no"
    with strict_mode(False):
        out = eeg_checkset(out)
    command = _history_command(event_source, options)
    out["history"] = _append_history(out.get("history", ""), command)
    return (out, command) if return_com else out


def _fields(value: Any) -> list[str] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        return [str(item) for item in parse_text_tokens(value)]
    return [str(item) for item in value]


def _timeunit(options: dict[str, Any]) -> float | None:
    if "timeunit" not in options:
        return None
    value = options["timeunit"]
    if isinstance(value, str) and value.lower() == "nan":
        return float("nan")
    return float(value)


def _history_command(event_source: Any, options: dict[str, Any]) -> str:
    pieces = [format_history_value("event"), format_history_value(event_source)]
    for key in ["fields", "append", "skipline", "timeunit", "align", "optimalign"]:
        if key in options:
            pieces.extend([format_history_value(key), format_history_value(options[key])])
    return f"EEG = pop_importevent(EEG, {', '.join(pieces)});"


def _append_history(history: str, command: str) -> str:
    return command if not history else f"{history.rstrip()}\n{command}"
