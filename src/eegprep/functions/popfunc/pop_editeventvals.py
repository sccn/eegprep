"""EEGLAB-style event-value editor."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset
from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc._event_utils import event_field_names, events_as_list, normalize_one_based_indices
from eegprep.functions.popfunc._pop_utils import format_history_value, is_empty_value as _is_empty
from eegprep.functions.popfunc.eeg_lat2point import eeg_lat2point
from eegprep.functions.popfunc.eeg_point2lat import eeg_point2lat


def pop_editeventvals(
    EEG: dict[str, Any],
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
):
    """Edit individual ``EEG.event`` values.

    This supports EEGLAB's command-line actions: ``changefield``, ``delete``,
    ``insert``/``append``/``add``, and ``sort``. Event indices are 1-based.
    """
    if not isinstance(EEG, dict):
        raise ValueError("pop_editeventvals: EEG must be a dataset dictionary")
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


def pop_editeventvals_dialog_spec(EEG: dict[str, Any]) -> DialogSpec:
    """Return the EEGLAB-like ``pop_editeventvals`` dialog spec."""
    events = events_as_list(EEG.get("event"))
    fields = event_field_names(events)
    current = events[0] if events else {}
    field_tags = {field: f"field_{field}" for field in fields}
    field_displays = tuple(
        {field_tags[field]: _display_event_value(EEG, event, field) for field in fields} for event in events
    )
    nav_enabled = len(events) > 1
    _nav_base = {
        "eventnum_tag": "eventnum",
        "max_index": len(events),
        "field_displays": field_displays,
    }

    def _nav(tag: str, delta: int) -> CallbackSpec | None:
        if not nav_enabled:
            return None
        return CallbackSpec(name="navigate_event", params={**_nav_base, "delta": delta, "source": tag})

    controls: list[ControlSpec] = [
        ControlSpec("text", f"Edit event field values (currently {len(events)} events)", font_weight="bold"),
        ControlSpec("pushbutton", "Delete event", tag="delete_button", enabled=False),
    ]
    geometry: list[tuple[float, ...]] = [(2, 0.5)]
    for field in fields:
        label = _display_field_label(EEG, field)
        controls.extend(
            [
                ControlSpec("spacer"),
                ControlSpec("pushbutton", label, enabled=False),
                ControlSpec("edit", tag=field_tags[field], value=_display_event_value(EEG, current, field)),
                ControlSpec("spacer"),
            ]
        )
        geometry.append((1, 1, 1, 1))
    controls.extend(
        [
            ControlSpec("spacer"),
            ControlSpec("spacer"),
            ControlSpec("spacer"),
            ControlSpec("text", "Event Num", font_weight="bold"),
            ControlSpec("spacer"),
            ControlSpec("spacer"),
            ControlSpec("spacer"),
            ControlSpec("pushbutton", "Insert event", tag="insert_button", enabled=False),
            ControlSpec("pushbutton", "<<", tag="back10", enabled=nav_enabled, callback=_nav("back10", -10)),
            ControlSpec("pushbutton", "<", tag="back1", enabled=nav_enabled, callback=_nav("back1", -1)),
            ControlSpec("edit", tag="eventnum", value="1"),
            ControlSpec("pushbutton", ">", tag="next1", enabled=nav_enabled, callback=_nav("next1", 1)),
            ControlSpec("pushbutton", ">>", tag="next10", enabled=nav_enabled, callback=_nav("next10", 10)),
            ControlSpec("pushbutton", "Append event", tag="append_button", enabled=False),
            ControlSpec("spacer"),
            ControlSpec("text", "Original value below", tag="original"),
            ControlSpec("spacer"),
        ]
    )
    geometry.extend(
        [
            (1.2, 0.6, 0.6, 1, 0.6, 0.6, 1.2),
            (1.2, 0.6, 0.6, 1, 0.6, 0.6, 1.2),
            (2, 1, 2),
        ]
    )
    return DialogSpec(
        title="Edit event values -- pop_editeventvals()",
        function_name="pop_editeventvals",
        eeglab_source="functions/popfunc/pop_editeventvals.m",
        size=(680, max(360, 170 + 34 * len(fields))),
        content_margins=(42, 26, 42, 30),
        row_spacing=7,
        help_text="pophelp('pop_editeventvals')",
        geometry=tuple(geometry),
        controls=tuple(controls),
        known_differences=(
            "EEGPrep edits one event per dialog submission; navigation refreshes the form for the targeted event but Delete/Insert/Append remain disabled.",
        ),
    )


def _run_gui(EEG: dict[str, Any], *, renderer: Any | None = None) -> list[tuple[str, Any]] | None:
    events = events_as_list(EEG.get("event"))
    if not events:
        raise ValueError("Getevent: cannot deal with empty event structure")
    fields = event_field_names(events)
    result = inputgui(pop_editeventvals_dialog_spec(EEG), renderer=renderer)
    if result is None:
        return None
    event_index = int(float(result.get("eventnum") or 1))
    current = events[event_index - 1]
    options: list[tuple[str, Any]] = []
    for field in fields:
        tag = f"field_{field}"
        value = _parse_gui_value(result.get(tag))
        if value != _display_event_value(EEG, current, field):
            options.append(("changefield", [event_index, field, value]))
    return options


def _apply_options(EEG: dict[str, Any], options: list[tuple[str, Any]]) -> dict[str, Any]:
    output = deepcopy(EEG)
    events = events_as_list(output.get("event"))
    if not events:
        raise ValueError("Getevent: cannot deal with empty event structure")
    for key, value in options:
        action = str(key).lower()
        if action in {"changefield", "assign"}:
            event_index, field, edit_value = _change_args(value)
            _change_field(output, events, event_index, str(field), edit_value)
        elif action == "delete":
            for event_index in sorted(
                normalize_one_based_indices(value, len(events), label="event index"), reverse=True
            ):
                events.pop(event_index)
        elif action in {"insert", "append", "add"}:
            event_index, new_event = _insert_args(events, value)
            if action == "append":
                event_index += 1
            insert_position = max(0, min(event_index, len(events)))
            events.insert(insert_position, new_event)
            _allocate_inserted_urevent(output, events, insert_position)
        elif action == "sort":
            events = _sort_events(events, value)
        else:
            raise ValueError(f"Unsupported pop_editeventvals action: {key}")
    output["event"] = events
    output["saved"] = "no"
    return eeg_checkset(output, "eventconsistency")


def _change_field(
    output: dict[str, Any], events: list[dict[str, Any]], event_index: int, field: str, value: Any
) -> None:
    index = normalize_one_based_indices(event_index, len(events), label="event index")[0]
    converted = _internal_event_value(output, events[index], field, value)
    events[index][field] = converted
    urevents = events_as_list(output.get("urevent"))
    urevent = events[index].get("urevent")
    if not _is_empty(urevent):
        urindex = int(urevent) - 1
        if 0 <= urindex < len(urevents):
            urevents[urindex][field] = converted
            output["urevent"] = urevents


def _internal_event_value(output: dict[str, Any], event: dict[str, Any], field: str, value: Any) -> Any:
    if field == "latency" and not _is_empty(value):
        if int(output.get("trials", 1) or 1) > 1:
            newlat, _ = eeg_lat2point(
                float(value),
                event.get("epoch", 1),
                float(output.get("srate", 1)),
                [float(output.get("xmin", 0)) * 1000, float(output.get("xmax", 0)) * 1000],
                1e-3,
            )
            return float(newlat.item())
        return (float(value) - float(output.get("xmin", 0))) * float(output.get("srate", 1)) + 1
    if field == "duration" and not _is_empty(value):
        scale = float(output.get("srate", 1)) / (1000 if int(output.get("trials", 1) or 1) > 1 else 1)
        return float(value) * scale
    return value


def _display_event_value(EEG: dict[str, Any], event: dict[str, Any], field: str) -> Any:
    value = event.get(field, "")
    if _is_empty(value):
        return ""
    if field == "latency":
        if int(EEG.get("trials", 1) or 1) > 1:
            return float(
                eeg_point2lat(
                    float(value),
                    event.get("epoch", 1),
                    float(EEG.get("srate", 1)),
                    [float(EEG.get("xmin", 0)) * 1000, float(EEG.get("xmax", 0)) * 1000],
                    1e-3,
                ).item()
            )
        return (float(value) - 1) / float(EEG.get("srate", 1)) + float(EEG.get("xmin", 0))
    if field == "duration":
        scale = float(EEG.get("srate", 1)) / (1000 if int(EEG.get("trials", 1) or 1) > 1 else 1)
        return float(value) / scale
    return value


def _display_field_label(EEG: dict[str, Any], field: str) -> str:
    if field == "latency":
        return "Latency (ms)" if int(EEG.get("trials", 1) or 1) > 1 else "Latency (sec)"
    if field == "duration":
        return "Duration (ms)" if int(EEG.get("trials", 1) or 1) > 1 else "Duration (sec)"
    return field.capitalize()


def _ordered_options(args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[tuple[str, Any]]:
    if len(args) % 2:
        raise ValueError("pop_editeventvals arguments must be key/value pairs")
    options = [(str(args[index]), args[index + 1]) for index in range(0, len(args), 2)]
    options.extend((str(key), value) for key, value in kwargs.items())
    return options


def _change_args(value: Any) -> tuple[int, str, Any]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("changefield requires [event_index, field, value]")
    return int(value[0]), str(value[1]), value[2]


def _insert_args(events: list[dict[str, Any]], value: Any) -> tuple[int, dict[str, Any]]:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(value) - 1, _empty_event_like(events)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("insert/append requires an event index")
    index = int(value[0]) - 1
    new_event = _empty_event_like(events)
    fields = event_field_names(events, include_urevent=True)
    for field, field_value in zip(fields, value[1:]):
        new_event[field] = field_value
    return index, new_event


def _empty_event_like(events: list[dict[str, Any]]) -> dict[str, Any]:
    fields = event_field_names(events, include_urevent=True)
    return {field: "" for field in fields}


def _sort_events(events: list[dict[str, Any]], value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("sort requires at least a field name")
    field1 = str(value[0])
    dir1 = int(value[1]) if len(value) > 1 and value[1] not in {None, ""} else 0
    field2 = str(value[2]) if len(value) > 2 and value[2] not in {None, ""} else ""
    dir2 = int(value[3]) if len(value) > 3 and value[3] not in {None, ""} else 0
    sorted_events = events
    if field2:
        sorted_events = sorted(sorted_events, key=lambda event: _sort_value(event.get(field2)), reverse=bool(dir2))
    return sorted(sorted_events, key=lambda event: _sort_value(event.get(field1)), reverse=bool(dir1))


def _sort_value(value: Any) -> Any:
    if _is_empty(value):
        return np.inf
    return value


def _allocate_inserted_urevent(output: dict[str, Any], events: list[dict[str, Any]], insert_position: int) -> None:
    if not any("urevent" in event for event in events):
        return
    urevents = events_as_list(output.get("urevent"))
    inserted = events[insert_position]
    inserted["urevent"] = len(urevents) + 1
    urevent = {field: deepcopy(value) for field, value in inserted.items() if field != "urevent"}
    urevents.append(urevent)
    output["urevent"] = urevents


def _parse_gui_value(value: Any) -> Any:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        numeric = float(text)
    except ValueError:
        return text
    return int(numeric) if numeric.is_integer() else numeric


def _history_command(options: list[tuple[str, Any]]) -> str:
    if not options:
        return ""
    pieces = []
    for key, value in options:
        pieces.append(format_history_value(str(key)))
        pieces.append(format_history_value(value, cell_for_sequence="any_strings"))
    return f"EEG = pop_editeventvals(EEG, {', '.join(pieces)});"


__all__ = ["pop_editeventvals", "pop_editeventvals_dialog_spec"]
