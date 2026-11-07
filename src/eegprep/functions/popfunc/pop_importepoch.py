"""Import epoch metadata into an EEGPrep EEG dataset."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset, strict_mode
from eegprep.functions.popfunc._file_io import events_to_records, read_table_records
from eegprep.functions.popfunc._pop_utils import format_history_value, parse_key_value_args, parse_text_tokens


def pop_importepoch(
    EEG: dict[str, Any],
    filename: Any = None,
    fieldlist: list[str] | tuple[str, ...] | str | None = None,
    *args: Any,
    return_com: bool = False,
    gui: bool | None = None,
    renderer: Any = None,
    **kwargs: Any,
) -> dict[str, Any] | tuple[dict[str, Any], str]:
    """Import per-epoch metadata and construct its corresponding events.

    Each input row describes one epoch. ``latencyfields`` create additional
    events at epoch-relative times, while ``typefield`` names the time-locking
    event at zero. Latencies and durations are stored in samples.
    """
    filename = filename if filename is not None else kwargs.pop('filename', None)
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    if gui is True or (gui is None and filename is None):
        initial = dict(options, filename=str(filename or ''), fieldlist=' '.join(_tokens(fieldlist) or []))
        result = inputgui(pop_importepoch_dialog_spec(EEG), initial_values=initial, renderer=renderer)
        if result is None:
            return (EEG, '') if return_com else EEG
        filename, fieldlist = result.pop('filename'), result.pop('fieldlist')
        options = {key: value for key, value in result.items() if value != ''}
        options['clearevents'] = 'on' if result['clearevents'] else 'off'
        options['timeunit'] = float(result['timeunit'])
        options['headerlines'] = int(result['headerlines'])
    if filename is None or (isinstance(filename, str) and not filename):
        raise ValueError('pop_importepoch: epoch file is missing. Select a file and enter its field names.')
    fields = _tokens(fieldlist)
    if isinstance(filename, (str, Path)) and not fields:
        raise ValueError(
            "pop_importepoch: field names are required for an epoch file. "
            "Supply fieldlist in column order and headerlines=1 when the file has a header."
        )
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
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return [str(item) for item in value.ravel()]
    return [str(item) for item in value]


def _history_command(filename: Any, fields: list[str] | None, options: dict[str, Any]) -> str:
    pieces = [format_history_value(filename), format_history_value(fields or [])]
    for key in ["latencyfields", "durationfields", "typefield", "timeunit", "headerlines", "clearevents"]:
        if key in options:
            pieces.extend([format_history_value(key), format_history_value(options[key])])
    return f"EEG = pop_importepoch(EEG, {', '.join(pieces)});"


def pop_importepoch_dialog_spec(EEG) -> DialogSpec:
    """Return the epoch-file fields and event conversion options."""
    browse = CallbackSpec('select_file', {'button': 'browse', 'target': 'filename', 'caption': 'Select a text file'})
    controls = [
        ControlSpec('text', 'Epoch file or array', font_weight='bold'),
        ControlSpec('pushbutton', 'Browse', tag='browse', callback=browse),
        ControlSpec('edit', tag='filename'),
        ControlSpec('spacer'),
        ControlSpec('text', 'File input field (col.) names', font_weight='bold'),
        ControlSpec('edit', tag='fieldlist'),
    ]
    for label, tag, hint in [
        ('Field name(s) containing event latencies', 'latencyfields', '(Ex: RT)'),
        ('Field name(s) containing event durations', 'durationfields', '0 = no duration'),
        ('Field containing time-locking event type(s)', 'typefield', 'Default: TLE'),
    ]:
        controls.extend([ControlSpec('text', label), ControlSpec('edit', tag=tag), ControlSpec('text', hint)])
    controls.extend(
        [
            ControlSpec('spacer'),
            ControlSpec('text', 'Time unit (sec)'),
            ControlSpec('edit', tag='timeunit', value='1'),
            ControlSpec('spacer'),
            ControlSpec('text', 'Number of file header lines'),
            ControlSpec('edit', tag='headerlines', value='0'),
            ControlSpec('spacer'),
            ControlSpec('text', 'Remove old epoch and event info (set = yes)'),
            ControlSpec('checkbox', tag='clearevents', value=not len(EEG.get('event', []))),
            ControlSpec('spacer'),
        ]
    )
    return DialogSpec(
        title='Import epoch info (data epochs only) -- pop_importepoch()',
        function_name='pop_importepoch',
        help_text='pop_importepoch',
        eeglab_source='functions/popfunc/pop_importepoch.m',
        geometry=([1, 1, 1.86], 1, [1, 0.66], *[[2.5, 1, 1]] * 3, 1, [2, 0.5, 0.5], [2, 0.5, 0.5], [2, 0.3, 0.7]),
        controls=tuple(controls),
        weighted_columns=True,
        size=(820, 350),
    )
