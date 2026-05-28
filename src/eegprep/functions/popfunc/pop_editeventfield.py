"""EEGLAB-style event-field editor."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset
from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import ControlSpec, DialogSpec
from eegprep.functions.popfunc._event_utils import (
    event_field_names,
    events_as_list,
    normalize_event_indices,
    sort_events,
    value_sequence,
)
from eegprep.functions.popfunc._pop_utils import format_history_value, parse_text_tokens


_RESERVED_OPTIONS = {"delold", "indices", "timeunit", "skipline", "delim", "rename"}


def pop_editeventfield(
    EEG: dict[str, Any],
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
):
    """Add, remove, rename, or update fields in ``EEG.event``.

    User-facing event indices are 1-based, matching EEGLAB. Event latencies
    passed through the ``latency`` field are interpreted as seconds by default
    and converted to EEGLAB sample latencies.
    """
    if not isinstance(EEG, dict):
        raise ValueError("pop_editeventfield: EEG must be a dataset dictionary")
    options = _ordered_options(args, kwargs)
    if gui is None:
        gui = not options
    if gui:
        result = _run_gui(EEG, renderer=renderer)
        if result is None:
            return (EEG, "") if return_com else EEG
        options = result

    output = _apply_options(EEG, options)
    command = _history_command(options)
    return (output, command) if return_com else output


def pop_editeventfield_dialog_spec(EEG: dict[str, Any]) -> DialogSpec:
    """Return the EEGLAB-like ``pop_editeventfield`` dialog spec."""
    fields = event_field_names(EEG.get("event"))
    field_choices = "|".join(["No field selected", *fields])
    return DialogSpec(
        title="Edit event field(s) -- pop_editeventfield()",
        function_name="pop_editeventfield",
        eeglab_source="functions/popfunc/pop_editeventfield.m",
        size=(760, 360),
        content_margins=(42, 26, 42, 30),
        row_spacing=8,
        help_text="pophelp('pop_editeventfield')",
        geometry=((1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)),
        controls=(
            ControlSpec("text", "Event fields", font_weight="bold"),
            ControlSpec("text", "File/array/value", font_weight="bold"),
            ControlSpec("text", "Event indices ([]=all)", font_weight="bold"),
            ControlSpec("edit", tag="field", value=""),
            ControlSpec("edit", tag="values", value=""),
            ControlSpec("edit", tag="indices", value=""),
            ControlSpec("text", "Rename old->new"),
            ControlSpec("edit", tag="rename", value=""),
            ControlSpec("text", "Delete field"),
            ControlSpec("popupmenu", field_choices, tag="deletefield", value=1),
            ControlSpec("text", "Field description"),
            ControlSpec("edit", tag="description", value=""),
            ControlSpec("text", "Field type"),
            ControlSpec("popupmenu", "Keep|Char|Num", tag="fieldtype", value=1),
            ControlSpec("text", "Delete old events first"),
            ControlSpec("checkbox", tag="delold", value=False),
            ControlSpec("spacer"),
            ControlSpec("spacer"),
        ),
        known_differences=(
            "EEGPrep presents a compact field editor; command-line behavior covers EEGLAB's add/delete/rename/type/info paths.",
        ),
    )


def _run_gui(EEG: dict[str, Any], *, renderer: Any | None = None) -> list[tuple[str, Any]] | None:
    result = inputgui(pop_editeventfield_dialog_spec(EEG), renderer=renderer)
    if result is None:
        return None
    options: list[tuple[str, Any]] = []
    indices = _parse_numeric_text(result.get("indices"))
    if indices:
        options.append(("indices", indices))
    if result.get("delold"):
        options.append(("delold", "yes"))
    rename = str(result.get("rename") or "").strip()
    if rename:
        options.append(("rename", rename))
    delete_index = int(result.get("deletefield") or 1)
    fields = event_field_names(EEG.get("event"))
    if delete_index > 1 and delete_index - 2 < len(fields):
        options.append((fields[delete_index - 2], []))
    field = str(result.get("field") or "").strip()
    if field:
        options.append((field, _parse_value_text(result.get("values"))))
        description = str(result.get("description") or "").strip()
        if description:
            options.append((f"{field}info", description))
        fieldtype = int(result.get("fieldtype") or 1)
        if fieldtype == 2:
            options.append((f"{field}type", "Char"))
        elif fieldtype == 3:
            options.append((f"{field}type", "Num"))
    return options


def _apply_options(EEG: dict[str, Any], options: list[tuple[str, Any]]) -> dict[str, Any]:
    output = deepcopy(EEG)
    events = events_as_list(output.get("event"))
    eventdescription = _description_list(
        output.get("eventdescription"), event_field_names(events, include_urevent=True)
    )
    option_map = {str(key).lower(): value for key, value in options}
    indices_value = option_map.get("indices")
    delold = str(option_map.get("delold", "no")).lower() in {"yes", "on", "1", "true"}
    timeunit = float(option_map.get("timeunit", 1) or 1)
    srate = float(output.get("srate", 1) or 1)

    for key, value in options:
        key_text = str(key)
        lower_key = key_text.lower()
        if lower_key in {"indices", "delold", "timeunit", "skipline", "delim"}:
            continue
        if lower_key == "rename":
            _rename_field(events, str(value), eventdescription)
            urevents = events_as_list(output.get("urevent"))
            if urevents:
                _rename_field(urevents, str(value), [])
                output["urevent"] = urevents
            continue
        if _is_empty(value):
            _delete_field(events, key_text, eventdescription)
            urevents = events_as_list(output.get("urevent"))
            if urevents:
                _delete_field(urevents, key_text, [])
                output["urevent"] = urevents
            continue
        if lower_key.endswith("info") and lower_key != "info":
            _set_description(events, eventdescription, key_text[:-4], str(value))
            continue
        if lower_key.endswith("type") and lower_key != "type":
            _convert_field_type(events, key_text[:-4], str(value))
            continue
        if delold:
            values = _coerce_field_values(value)
            events = [{} for _item in values]
            indices = list(range(len(values)))
            eventdescription = [""]
            delold = False
        else:
            indices = normalize_event_indices(indices_value, len(events), allow_empty=True)
            if not indices:
                indices = list(range(len(events))) if events else []
            values = _coerce_field_values(value)
            if not indices:
                indices = list(range(len(values)))
            while len(events) < max(indices, default=-1) + 1:
                events.append({})
        values = value_sequence(values, len(indices))
        for event_index, item in zip(indices, values):
            events[event_index][key_text] = _field_value(key_text, item, srate=srate, timeunit=timeunit)
            _update_matching_urevent(output, events[event_index], key_text)
        _ensure_description_slot(eventdescription, event_field_names(events, include_urevent=True), key_text)

    output["event"] = sort_events(events)
    output["eventdescription"] = eventdescription
    output["saved"] = "no"
    return eeg_checkset(output)


def _ordered_options(args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[tuple[str, Any]]:
    if len(args) % 2:
        raise ValueError("pop_editeventfield arguments must be key/value pairs")
    options = [(str(args[index]), args[index + 1]) for index in range(0, len(args), 2)]
    options.extend((str(key), value) for key, value in kwargs.items())
    return options


def _history_command(options: list[tuple[str, Any]]) -> str:
    if not options:
        return ""
    pieces = []
    for key, value in options:
        pieces.append(format_history_value(str(key)))
        pieces.append(format_history_value(value, cell_for_sequence="any_strings"))
    return f"EEG = pop_editeventfield( EEG, {', '.join(pieces)});"


def _description_list(value: Any, fields: list[str]) -> list[str]:
    if isinstance(value, dict):
        return [str(value.get(field, "")) for field in fields]
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list):
        return [str(item) for item in value]
    return ["" for _field in fields]


def _rename_field(events: list[dict[str, Any]], rename: str, eventdescription: list[str]) -> None:
    if "->" not in rename:
        raise ValueError("rename must use 'old->new' syntax")
    old, new = [part.strip() for part in rename.split("->", 1)]
    fields = event_field_names(events, include_urevent=True)
    if old not in fields:
        raise ValueError(f"event field not found: {old}")
    for event in events:
        if old in event:
            event[new] = event.pop(old)
    try:
        index = fields.index(old)
        while len(eventdescription) <= index:
            eventdescription.append("")
        fields[index] = new
    except ValueError:
        return


def _set_description(
    events: list[dict[str, Any]],
    eventdescription: list[str],
    field: str,
    description: str,
) -> None:
    fields = event_field_names(events, include_urevent=True)
    _ensure_description_slot(eventdescription, fields, field)
    fields = event_field_names(events, include_urevent=True)
    eventdescription[fields.index(field)] = description


def _ensure_description_slot(eventdescription: list[str], fields: list[str], field: str) -> None:
    if field not in fields:
        fields.append(field)
    while len(eventdescription) <= fields.index(field):
        eventdescription.append("")


def _convert_field_type(events: list[dict[str, Any]], field: str, field_type: str) -> None:
    if field in {"latency", "epoch", "urevent"}:
        raise ValueError("This field type cannot be changed")
    to_char = field_type.lower() == "char"
    for event in events:
        if field not in event:
            continue
        event[field] = str(event[field]) if to_char else _to_number(event[field])


def _delete_field(events: list[dict[str, Any]], field: str, eventdescription: list[str]) -> None:
    fields = event_field_names(events, include_urevent=True)
    for event in events:
        event.pop(field, None)
    if field in fields:
        index = fields.index(field)
        if index < len(eventdescription):
            eventdescription.pop(index)


def _coerce_field_values(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.ravel().tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return [value]


def _field_value(field: str, value: Any, *, srate: float, timeunit: float) -> Any:
    if field == "latency":
        return float(value) * timeunit * srate + 1
    if field == "duration":
        return float(value) * srate
    return value


def _update_matching_urevent(output: dict[str, Any], event: dict[str, Any], field: str) -> None:
    urevent_index = event.get("urevent")
    if _is_empty(urevent_index):
        return
    urevents = events_as_list(output.get("urevent"))
    try:
        index = int(urevent_index) - 1
    except (TypeError, ValueError):
        return
    if 0 <= index < len(urevents):
        urevents[index][field] = event.get(field)
        output["urevent"] = urevents


def _parse_value_text(value: Any) -> Any:
    text = str(value or "").strip()
    if not text:
        return ""
    values = parse_text_tokens(text, parse_ints=False)
    if len(values) > 1:
        return [_to_number(item) for item in values]
    return _to_number(values[0]) if values else text


def _parse_numeric_text(value: Any) -> list[int]:
    text = str(value or "").strip()
    if not text:
        return []
    return [int(float(token)) for token in text.strip("[]").replace(",", " ").split() if token]


def _to_number(value: Any) -> Any:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return value.item() if isinstance(value, np.generic) else value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    return int(numeric) if numeric.is_integer() else numeric


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, np.ndarray):
        return value.size == 0
    if isinstance(value, (list, tuple, dict, str)):
        return len(value) == 0
    return False


__all__ = ["pop_editeventfield", "pop_editeventfield_dialog_spec"]
