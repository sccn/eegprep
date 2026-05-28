"""EEGLAB-style event selection."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset
from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc._event_utils import (
    event_field_names,
    events_as_list,
    is_boundary_event,
    normalize_event_indices,
)
from eegprep.functions.popfunc._pop_utils import format_history_value, parse_key_value_args, parse_text_tokens
from eegprep.functions.popfunc.eeg_point2lat import eeg_point2lat
from eegprep.functions.popfunc.pop_select import pop_select


def pop_selectevent(
    EEG: dict[str, Any] | list[dict[str, Any]],
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
):
    """Select, rename, or delete events using EEGLAB ``pop_selectevent`` semantics."""
    options = parse_key_value_args(args, kwargs)
    if gui is None:
        gui = not bool(options)
    if gui:
        result = _run_gui(EEG[0] if isinstance(EEG, list) else EEG, renderer=renderer)
        if result is None:
            return (EEG, "") if return_com else (EEG, [])
        options.update(result)
    if isinstance(EEG, list):
        outputs = []
        indices = []
        for dataset in EEG:
            output, selected = _apply_selectevent_one(dataset, options)
            outputs.append(output)
            indices.append(selected)
        command = _history_command(options)
        return (outputs, command) if return_com else (outputs, indices)
    output, selected = _apply_selectevent_one(EEG, options)
    command = _history_command(options)
    return (output, command) if return_com else (output, selected)


def pop_selectevent_dialog_spec(EEG: dict[str, Any]) -> DialogSpec:
    """Return the EEGLAB-like ``pop_selectevent`` dialog spec."""
    events = events_as_list(EEG.get("event"))
    fields = _ordered_event_fields(events)
    event_types = tuple(sorted({str(event.get("type")) for event in events if event.get("type") not in {None, ""}}))
    controls: list[ControlSpec] = [
        ControlSpec("text", "Field", font_weight="bold"),
        ControlSpec("text", "Selection", font_weight="bold"),
        ControlSpec("text", "Set=NOT THESE", font_weight="bold"),
    ]
    geometry: list[tuple[float, ...]] = [(0.6, 2.1, 0.8)]
    for field in fields:
        if field in {"latency", "duration"}:
            controls.extend(
                [
                    ControlSpec("text", _display_field_label(EEG, field)),
                    ControlSpec("edit", tag=f"min_{field}", value=""),
                    ControlSpec("edit", tag=f"max_{field}", value=""),
                    ControlSpec("checkbox", tag=f"not_{field}", value=False),
                ]
            )
            geometry.append((0.55, 0.55, 0.55, 0.22))
        elif field == "type":
            controls.extend(
                [
                    ControlSpec("text", "type"),
                    ControlSpec("edit", tag="type", value=""),
                    ControlSpec(
                        "pushbutton",
                        "...",
                        tag="type_button",
                        callback=CallbackSpec(
                            "select_event_types",
                            params={"button": "type_button", "target": "type", "event_types": event_types},
                        ),
                    ),
                    ControlSpec("checkbox", tag="not_type", value=False),
                ]
            )
            geometry.append((0.55, 1.1, 0.3, 0.22))
        else:
            controls.extend(
                [
                    ControlSpec("text", field),
                    ControlSpec("edit", tag=field, value=""),
                    ControlSpec("spacer"),
                    ControlSpec("checkbox", tag=f"not_{field}", value=False),
                ]
            )
            geometry.append((0.55, 1.3, 0.1, 0.22))
    controls.extend(
        [
            ControlSpec("text", "Event indices"),
            ControlSpec("edit", tag="indices", value=""),
            ControlSpec("spacer"),
            ControlSpec("checkbox", tag="not_indices", value=False),
            ControlSpec("spacer"),
            ControlSpec("text", "Event selection", font_weight="bold"),
            ControlSpec("spacer"),
            ControlSpec("checkbox", "Select all events NOT selected above", tag="invertevent", value=False),
            ControlSpec("spacer"),
            ControlSpec(
                "checkbox",
                "Keep only selected events and remove all other events",
                tag="deleteevents",
                value=int(EEG.get("trials", 1) or 1) == 1,
            ),
            ControlSpec("text", "Rename selected event type(s) as type:"),
            ControlSpec("edit", tag="renametype", value=""),
            ControlSpec("text", "Retain old event type name(s) in field named:"),
            ControlSpec("edit", tag="oldtypefield", value=""),
        ]
    )
    geometry.extend([(0.55, 1.3, 0.1, 0.22), (1,), (0.1, 2, 0.5, 0.5), (0.1, 2, 0.5, 0.5), (1, 1)])
    if int(EEG.get("trials", 1) or 1) > 1:
        controls.extend(
            [
                ControlSpec("text", "Epoch selection", font_weight="bold"),
                ControlSpec(
                    "checkbox", "Remove epochs not referenced by any selected event", tag="deleteepochs", value=True
                ),
                ControlSpec("checkbox", "Invert epoch selection", tag="invertepochs", value=False),
            ]
        )
        geometry.extend([(1,), (0.1, 2, 0.5), (0.1, 2, 0.5)])
    return DialogSpec(
        title="Select events -- pop_selectevent()",
        function_name="pop_selectevent",
        eeglab_source="functions/popfunc/pop_selectevent.m",
        size=(760, max(430, 180 + 34 * len(fields))),
        content_margins=(42, 26, 42, 30),
        row_spacing=6,
        help_text="pophelp('pop_selectevent')",
        geometry=tuple(geometry),
        controls=tuple(controls),
    )


def _run_gui(EEG: dict[str, Any], *, renderer: Any | None = None) -> dict[str, Any] | None:
    result = inputgui(pop_selectevent_dialog_spec(EEG), renderer=renderer)
    if result is None:
        return None
    options: dict[str, Any] = {}
    indices = _parse_numeric_vector(result.get("indices"))
    if indices:
        options["omitevent" if result.get("not_indices") else "event"] = indices
    for field in _ordered_event_fields(events_as_list(EEG.get("event"))):
        if field in {"latency", "duration"}:
            minimum = str(result.get(f"min_{field}") or "").strip()
            maximum = str(result.get(f"max_{field}") or "").strip()
            if minimum and maximum:
                options[f"omit{field}" if result.get(f"not_{field}") else field] = f"{minimum}<={maximum}"
            continue
        text = str(result.get(field) or "").strip()
        if text:
            options[f"omit{field}" if result.get(f"not_{field}") else field] = _parse_field_text(text)
    if result.get("invertevent"):
        options["select"] = "inverse"
    if result.get("deleteevents"):
        options["deleteevents"] = "on"
    else:
        options["deleteevents"] = "off"
    if result.get("deleteepochs") is not None:
        options["deleteepochs"] = "on" if result.get("deleteepochs") else "off"
    if result.get("invertepochs"):
        options["invertepochs"] = "on"
    renametype = str(result.get("renametype") or "").strip()
    if renametype:
        options["renametype"] = renametype
    oldtypefield = str(result.get("oldtypefield") or "").strip()
    if oldtypefield:
        options["oldtypefield"] = oldtypefield
    return options


def _apply_selectevent_one(EEG: dict[str, Any], options: dict[str, Any]) -> tuple[dict[str, Any], list[int]]:
    events = events_as_list(EEG.get("event"))
    if not events:
        raise ValueError("pop_selectevent: cannot deal with empty event structure")
    output = deepcopy(EEG)
    fields = event_field_names(events)
    selected = set(normalize_event_indices(options.get("event"), len(events)))
    omitted = set(normalize_event_indices(options.get("omitevent"), len(events), allow_empty=True))

    for field in fields:
        selected &= _matching_indices(EEG, events, field, options.get(field), omit=False)
        omitted |= _matching_indices(EEG, events, field, options.get(f"omit{field}"), omit=True)

    selected -= omitted
    if str(options.get("select", "normal")).lower() in {"inverse", "remove"}:
        selected = set(range(len(events))) - selected
    boundary = {index for index, event in enumerate(events) if is_boundary_event(event)}
    if int(EEG.get("trials", 1) or 1) == 1 and str(options.get("deleteevents", "off")).lower() in {"on", "yes"}:
        selected |= boundary
    selected_indices = sorted(selected)

    renametype = str(options.get("renametype") or "")
    oldtypefield = str(options.get("oldtypefield") or "")
    if oldtypefield and not renametype:
        raise ValueError("A name for the new type must be defined")
    if renametype:
        for index in sorted(selected - boundary):
            if oldtypefield:
                events[index][oldtypefield] = events[index].get("type", "")
            events[index]["type"] = renametype

    deleteevents = str(options.get("deleteevents", "off")).lower() in {"on", "yes"}
    deleteepochs = str(options.get("deleteepochs", "on")).lower() in {"on", "yes"}
    output["event"] = events
    if deleteepochs and int(output.get("trials", 1) or 1) > 1:
        epochs = _selected_epochs(events, selected_indices, int(output.get("trials", 1) or 1))
        if str(options.get("invertepochs", "off")).lower() == "on":
            all_epochs = set(range(1, int(output.get("trials", 1) or 1) + 1))
            epochs = sorted(all_epochs - set(epochs))
        if not epochs:
            raise ValueError("Empty dataset: all epochs have been removed")
        if deleteevents:
            output["event"] = [events[index] for index in selected_indices]
        output = pop_select(output, "trial", epochs)
    elif deleteevents:
        output["event"] = [events[index] for index in selected_indices]
    else:
        output["event"] = events
    output["saved"] = "no"
    output = eeg_checkset(output, "eventconsistency")
    return output, [index + 1 for index in selected_indices]


def _matching_indices(
    EEG: dict[str, Any],
    events: list[dict[str, Any]],
    field: str,
    criterion: Any,
    *,
    omit: bool,
) -> set[int]:
    if _is_empty(criterion):
        return set() if omit else set(range(len(events)))
    if isinstance(criterion, str) and "<=" in criterion:
        lower, upper = [float(part.strip()) for part in criterion.split("<=", 1)]
        return {index for index, event in enumerate(events) if lower <= _comparison_value(EEG, event, field) <= upper}
    values = criterion if isinstance(criterion, (list, tuple, set, np.ndarray)) else [criterion]
    wanted = {str(value).strip() for value in values}
    numeric = set()
    for value in values:
        number = _to_float(value)
        if number is not None:
            numeric.add(number)
    return {
        index
        for index, event in enumerate(events)
        if str(event.get(field, "")).strip() in wanted or (_to_float(event.get(field)) in numeric if numeric else False)
    }


def _comparison_value(EEG: dict[str, Any], event: dict[str, Any], field: str) -> float:
    value = float(event.get(field, np.nan))
    if field == "latency":
        return float(
            eeg_point2lat(
                value, event.get("epoch", 1), float(EEG.get("srate", 1)), [EEG.get("xmin", 0), EEG.get("xmax", 0)]
            )
        )
    if field == "duration":
        scale = float(EEG.get("srate", 1)) / (1000 if int(EEG.get("trials", 1) or 1) > 1 else 1)
        return value / scale
    return value


def _selected_epochs(events: list[dict[str, Any]], selected_indices: list[int], trials: int) -> list[int]:
    epochs = sorted(
        {
            int(events[index].get("epoch", 0))
            for index in selected_indices
            if 1 <= int(events[index].get("epoch", 0) or 0) <= trials
        }
    )
    return epochs


def _ordered_event_fields(events: list[dict[str, Any]]) -> list[str]:
    fields = event_field_names(events)
    ordered = [field for field in ("latency", "duration", "type") if field in fields]
    ordered.extend(field for field in fields if field not in ordered)
    return ordered


def _display_field_label(EEG: dict[str, Any], field: str) -> str:
    if field == "latency":
        return "latency (ms)" if int(EEG.get("trials", 1) or 1) > 1 else "latency (s)"
    if field == "duration":
        return "duration (ms)" if int(EEG.get("trials", 1) or 1) > 1 else "duration (s)"
    return field


def _parse_field_text(text: str) -> Any:
    values = parse_text_tokens(text, parse_ints=True)
    return values if len(values) != 1 else values[0]


def _parse_numeric_vector(value: Any) -> list[int]:
    text = str(value or "").strip()
    if not text:
        return []
    return [int(float(token)) for token in text.strip("[]").replace(",", " ").split() if token]


def _history_command(options: dict[str, Any]) -> str:
    pieces = []
    for key, value in options.items():
        if _is_empty(value):
            continue
        pieces.append(format_history_value(key))
        pieces.append(format_history_value(value, cell_for_sequence="any_strings"))
    return f"EEG = pop_selectevent( EEG, {', '.join(pieces)});"


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, np.ndarray):
        return value.size == 0
    if isinstance(value, (list, tuple, set, dict, str)):
        return len(value) == 0
    return False


__all__ = ["pop_selectevent", "pop_selectevent_dialog_spec"]
