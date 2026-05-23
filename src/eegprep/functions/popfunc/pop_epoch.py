"""EEGLAB-style epoch extraction pop function."""

from __future__ import annotations

import copy
import logging
import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset
from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import (
    format_history_value,
    parse_key_value_args,
    parse_text_tokens,
)
from eegprep.functions.popfunc.eeg_findboundaries import eeg_findboundaries
from eegprep.functions.popfunc.pop_select import pop_select
from eegprep.functions.sigprocfunc.epoch import epoch


logger = logging.getLogger(__name__)

_UNSET = object()
_VALID_OPTIONS = {
    "epochfield",
    "timeunit",
    "verbose",
    "newname",
    "eventindices",
    "epochinfo",
    "valuelim",
}


def pop_epoch(
    EEG: Any = _UNSET,
    types: Any = _UNSET,
    lim: Any = _UNSET,
    *args: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
    **kwargs: Any,
):
    """Extract data epochs time locked to event types or event indices.

    This mirrors EEGLAB's ``pop_epoch`` workflow. Calling ``pop_epoch(EEG)``
    opens the EEGPrep GUI; passing ``types`` and ``lim`` runs directly.
    Returned ``indices`` are 0-based Python indices into the accepted
    time-locking event list. Use ``return_com=True`` for GUI/session callers
    that need ``(EEG, command)`` for shared history.
    """
    if EEG is _UNSET or EEG is None:
        raise ValueError("pop_epoch: EEG dataset is required")

    options = _parse_options(args, kwargs)
    if gui is None:
        gui = types is _UNSET and lim is _UNSET and not options

    if gui:
        gui_eeg = EEG[0] if isinstance(EEG, list) else EEG
        gui_result = _run_gui(gui_eeg, renderer=renderer, multiple=_is_multiple(EEG))
        if gui_result is None:
            return (EEG, "") if return_com else (EEG, [])
        types = gui_result["types"]
        lim = gui_result["lim"]
        options.update(gui_result["options"])
    else:
        types = [] if types is _UNSET or types is None else types
        lim = [-1, 2] if lim is _UNSET or lim is None else lim

    types = _normalise_types(types)
    lim = _normalise_lim(lim)

    if isinstance(EEG, list):
        if len(EEG) == 1:
            eeg_out, indices = _pop_epoch_one(EEG[0], types, lim, options)
            command = _history_command(types, lim, options)
            return (eeg_out, command) if return_com else (eeg_out, indices)
        outputs = []
        all_indices = []
        for dataset in EEG:
            eeg_out, indices = _pop_epoch_one(dataset, types, lim, options)
            outputs.append(eeg_out)
            all_indices.append(indices)
        command = _history_command(types, lim, options)
        return (outputs, command) if return_com else (outputs, all_indices)

    eeg_out, indices = _pop_epoch_one(EEG, types, lim, options)
    command = _history_command(types, lim, options)
    return (eeg_out, command) if return_com else (eeg_out, indices)


def pop_epoch_dialog_spec(EEG: dict[str, Any], *, multiple: bool = False) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_epoch``."""
    epoch_limits = _default_epoch_limits(EEG)
    event_types = tuple(_unique_event_types(_event_list(EEG.get("event"))))
    setname = _text_field(EEG.get("setname", ""))
    newname = "" if multiple or not setname else f"{setname} epochs"
    return DialogSpec(
        title="Extract data epochs - pop_epoch()",
        function_name="pop_epoch",
        eeglab_source="functions/popfunc/pop_epoch.m",
        geometry=((2, 1, 0.5), (2, 1, 0.5), (2, 1.5), (2, 1, 0.5)),
        size=(665, 264),
        content_margins=(42, 33, 42, 35),
        help_text="pophelp('pop_epoch')",
        controls=(
            ControlSpec("text", "Time-locking event type(s) ([]=all)"),
            ControlSpec("edit", tag="events", value=""),
            ControlSpec(
                "pushbutton",
                "...",
                tag="eventtypes_button",
                callback=CallbackSpec(
                    "select_event_types",
                    params={
                        "button": "eventtypes_button",
                        "target": "events",
                        "event_types": event_types,
                    },
                    matlab_callback="pop_chansel(unique({ EEG.event.type }))",
                ),
            ),
            ControlSpec("text", "Epoch limits [start, end] in seconds"),
            ControlSpec("edit", tag="limits", value=epoch_limits),
            ControlSpec("spacer"),
            ControlSpec("text", "Name for the new dataset"),
            ControlSpec("edit", tag="newname", value=newname, enabled=not multiple),
            ControlSpec("text", "Out-of-bounds EEG limits if any [min max]"),
            ControlSpec("edit", tag="valuelim", value=""),
            ControlSpec("spacer"),
        ),
    )


def _run_gui(EEG: dict[str, Any], *, renderer: Any | None = None, multiple: bool = False) -> dict[str, Any] | None:
    spec = pop_epoch_dialog_spec(EEG, multiple=multiple)
    result = inputgui(spec, renderer=renderer)
    if result is None:
        return None

    events_text = str(result.get("events", "")).strip()
    limits_text = str(result.get("limits", "")).strip()
    valuelim_text = str(result.get("valuelim", "")).strip()
    newname = str(result.get("newname", "")).strip()

    types = _parse_gui_types(events_text)
    lim = _parse_numeric_vector(limits_text, "Epoch limits", length=2)
    options: dict[str, Any] = {}
    if newname:
        options["newname"] = newname
    if valuelim_text:
        options["valuelim"] = _parse_numeric_vector(valuelim_text, "Out-of-bounds EEG limits", length=None)
    options["epochinfo"] = "yes"
    return {"types": types, "lim": lim, "options": options}


def _pop_epoch_one(
    EEG: dict[str, Any],
    types: Any,
    lim: list[float],
    options: dict[str, Any],
) -> tuple[dict[str, Any], list[int]]:
    if not isinstance(EEG, dict):
        raise ValueError("pop_epoch: EEG must be a dataset dictionary")

    events = _event_list(EEG.get("event"))
    if not events:
        if (
            int(EEG.get("trials", 1) or 1) > 1
            and float(EEG.get("xmin", 0) or 0) <= 0 <= float(EEG.get("xmax", 0) or 0)
        ):
            events = _time_locking_events(EEG)
        else:
            logger.info("Cannot epoch data with no events")
            return EEG, []

    if not any("latency" in event for event in events):
        raise ValueError("Absent latency field in event array/structure: must name one of the fields 'latency'")
    if isinstance(EEG.get("data"), str):
        raise NotImplementedError("Data loading from file not implemented")

    data = np.asarray(EEG.get("data"))
    if data.ndim not in {2, 3}:
        raise ValueError("pop_epoch supports continuous or epoched EEG data")

    g = _processing_options(EEG, events, options)
    sorted_event_indices = sorted(range(len(events)), key=lambda index: float(events[index].get("latency", 0)))
    sorted_events = [copy.deepcopy(events[index]) for index in sorted_event_indices]
    selected_event_indices = _selected_event_indices(sorted_events, types, g["eventindices"])
    alllatencies = [float(sorted_events[index]["latency"]) for index in selected_event_indices]
    selected_event_indices, alllatencies = _adjust_latencies_for_boundaries(
        sorted_events,
        selected_event_indices,
        alllatencies,
        lim,
        float(EEG.get("srate", 1)),
    )

    if not alllatencies:
        raise ValueError("pop_epoch(): empty epoch range (no epochs were found).")
    logger.info("pop_epoch(): %d epochs selected", len(alllatencies))

    tmpeventlatency = [float(event["latency"]) for event in sorted_events]
    if g["timeunit"].lower() == "points":
        epochdat, tmptime, indices, alleventout, _alllatencyout, _reallim = epoch(
            data,
            alllatencies,
            [lim[0] * float(EEG["srate"]), lim[1] * float(EEG["srate"])],
            valuelim=g["valuelim"],
            allevents=tmpeventlatency,
            verbose=g["verbose"],
        )
        tmptime = tmptime / float(EEG["srate"])
    elif g["timeunit"].lower() == "seconds":
        epochdat, tmptime, indices, alleventout, _alllatencyout, _reallim = epoch(
            data,
            alllatencies,
            lim,
            valuelim=g["valuelim"],
            srate=float(EEG["srate"]),
            allevents=tmpeventlatency,
            verbose=g["verbose"],
        )
    else:
        raise ValueError("pop_epoch: invalid event time format")

    accepted_positions = [int(index) for index in np.asarray(indices, dtype=int).tolist()]
    accepted_latencies = [alllatencies[index] for index in accepted_positions]
    logger.info("pop_epoch(): %d epochs generated", len(accepted_positions))

    output = copy.deepcopy(EEG)
    output["data"] = epochdat
    output["nbchan"] = int(epochdat.shape[0])
    output["xmin"] = float(tmptime[0])
    output["xmax"] = float(tmptime[1])
    output["pnts"] = int(epochdat.shape[1])
    output["trials"] = int(epochdat.shape[2])
    output["times"] = np.linspace(output["xmin"] * 1000, output["xmax"] * 1000, output["pnts"])
    output["icaact"] = np.array([])
    output["event"] = _epoch_events(
        sorted_events,
        alleventout,
        accepted_latencies,
        output["pnts"],
        output["srate"],
        output["xmin"],
    )
    output["epoch"] = []
    output["saved"] = "no"
    _update_dataset_text(output, EEG, g["newname"])
    output = eeg_checkset(output, "eventconsistency")

    output, accepted_positions = _remove_boundary_epochs(output, accepted_positions)
    return output, accepted_positions


def _parse_options(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    unknown = sorted(set(options) - _VALID_OPTIONS)
    if unknown:
        raise ValueError(f"pop_epoch: unsupported option(s): {', '.join(unknown)}")
    return options


def _processing_options(EEG: dict[str, Any], events: list[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
    setname = _text_field(EEG.get("setname", ""))
    return {
        "epochfield": str(options.get("epochfield", "type")),
        "timeunit": str(options.get("timeunit", "points")),
        "verbose": str(options.get("verbose", "on")),
        "newname": _text_field(options.get("newname", f"{setname} epochs" if setname else "")),
        "eventindices": _normalise_eventindices(options.get("eventindices", None), len(events)),
        "epochinfo": str(options.get("epochinfo", "yes")),
        "valuelim": _normalise_valuelim(options.get("valuelim", None)),
    }


def _normalise_types(types: Any) -> Any:
    if types is None:
        return []
    if isinstance(types, np.ndarray):
        types = types.tolist()
    if isinstance(types, bytes):
        return types.decode("utf-8")
    if isinstance(types, str):
        return [] if types.strip() == "[]" else types
    if isinstance(types, Sequence):
        return list(types)
    raise ValueError("pop_epoch(): multiple event types must be entered as a list/cell array or a string")


def _normalise_lim(lim: Any) -> list[float]:
    values = _as_float_list(lim)
    if len(values) != 2:
        raise ValueError("Epoch limits must contain exactly two values")
    if values[0] >= values[1]:
        raise ValueError("Epoch start must be lower than epoch end")
    return values


def _normalise_valuelim(value: Any) -> list[float]:
    if value is None or _is_empty_sequence(value):
        return [-np.inf, np.inf]
    values = _as_float_list(value)
    if len(values) == 1:
        limit = abs(values[0])
        return [-limit, limit]
    if len(values) != 2:
        raise ValueError("valuelim must contain one or two values")
    return values


def _normalise_eventindices(value: Any, event_count: int) -> list[int]:
    if value is None:
        return list(range(event_count))
    if _is_empty_sequence(value):
        return []
    indices = [int(item) for item in _as_list(value)]
    if any(index == 0 for index in indices):
        normalised = indices
    else:
        normalised = [index - 1 for index in indices]
    if any(index < 0 or index >= event_count for index in normalised):
        raise ValueError("eventindices are out of range")
    return sorted(set(normalised))


def _selected_event_indices(events: list[dict[str, Any]], types: Any, eventindices: list[int]) -> list[int]:
    if not types:
        return sorted(eventindices)
    eventindex_set = set(eventindices)
    if isinstance(types, str):
        types = [types]
    if not isinstance(types, list):
        raise ValueError("pop_epoch(): multiple event types must be entered as a list/cell array or a string")
    selected = []
    for index, event in enumerate(events):
        if index not in eventindex_set:
            continue
        if _event_type_matches(event.get("type"), types):
            selected.append(index)
    return sorted(selected)


def _event_type_matches(event_type: Any, types: list[Any]) -> bool:
    if event_type in types:
        return True
    if isinstance(event_type, bytes):
        event_type = event_type.decode("utf-8")
    for value in types:
        if isinstance(event_type, str) and str(value) == event_type:
            return True
        if not isinstance(event_type, str):
            try:
                if float(event_type) == float(value):
                    return True
            except (TypeError, ValueError):
                continue
    return False


def _adjust_latencies_for_boundaries(
    events: list[dict[str, Any]],
    selected_indices: list[int],
    latencies: list[float],
    lim: list[float],
    srate: float,
) -> tuple[list[int], list[float]]:
    pairs = list(zip(selected_indices, latencies))
    if lim[0] > 0:
        pairs = _adjust_boundary_side(events, pairs, lim[0] * srate, after=True)
    if lim[1] < 0:
        pairs = _adjust_boundary_side(events, pairs, lim[1] * srate, after=False)
    return [index for index, _latency in pairs], [latency for _index, latency in pairs]


def _adjust_boundary_side(
    events: list[dict[str, Any]],
    pairs: list[tuple[int, float]],
    offset: float,
    *,
    after: bool,
) -> list[tuple[int, float]]:
    adjusted = list(pairs)
    for pair_index in range(len(adjusted) - 1, -1, -1):
        _, latency = adjusted[pair_index]
        if after:
            low, high = latency, latency + offset
            boundary_events = [event for event in events if _latency_between(event, low, high)]
            sign = -1
            threshold = offset
        else:
            low, high = latency + offset, latency
            boundary_events = [event for event in events if _latency_between(event, low, high)]
            sign = 1
            threshold = -offset
        boundary_events = [event for event in boundary_events if _is_boundary_event(event)]
        if not boundary_events:
            continue
        duration = sum(float(event.get("duration", 0) or 0) for event in boundary_events)
        if duration > threshold:
            adjusted.pop(pair_index)
        else:
            adjusted[pair_index] = (adjusted[pair_index][0], latency + sign * duration)
    return adjusted


def _epoch_events(
    events: list[dict[str, Any]],
    alleventout: list[Any],
    accepted_latencies: list[float],
    pnts: int,
    srate: float,
    xmin: float,
) -> list[dict[str, Any]]:
    new_events = []
    if not alleventout:
        return new_events
    for epoch_index, epoch_events in enumerate(alleventout):
        for event_index in np.asarray(epoch_events, dtype=int).tolist():
            if event_index < 0 or event_index >= len(events):
                continue
            new_event = copy.deepcopy(events[event_index])
            new_event["epoch"] = epoch_index + 1
            new_event["latency"] = (
                float(new_event["latency"])
                - accepted_latencies[epoch_index]
                - xmin * srate
                + 1
                + pnts * epoch_index
            )
            new_events.append(new_event)
    return new_events


def _remove_boundary_epochs(output: dict[str, Any], accepted_positions: list[int]) -> tuple[dict[str, Any], list[int]]:
    events = _event_list(output.get("event"))
    if not events:
        return output, accepted_positions
    if not isinstance(events[0].get("type"), str):
        return output, accepted_positions
    boundary_indices = eeg_findboundaries(EEG=events)
    if not boundary_indices:
        return output, accepted_positions
    remove_epochs = {
        int(events[index].get("epoch", 1) or 1)
        for index in boundary_indices
    }
    if not remove_epochs:
        return output, accepted_positions
    selected = pop_select(output, notrial=sorted(remove_epochs))
    if isinstance(selected, tuple):
        selected = selected[0]
    kept_positions = [
        position for epoch_number, position in enumerate(accepted_positions, start=1)
        if epoch_number not in remove_epochs
    ]
    return selected, kept_positions


def _update_dataset_text(output: dict[str, Any], original: dict[str, Any], newname: str) -> None:
    setname = _text_field(original.get("setname", ""))
    if setname:
        comments = original.get("comments")
        if _has_comment_text(comments):
            if isinstance(comments, str):
                body = comments
            else:
                body = "\n".join(str(line) for line in comments)
            output["comments"] = (
                f'Parent dataset: {setname}\n\nParent dataset "{setname}": ----------\n{body}'
            )
        else:
            output["comments"] = f"Parent dataset: {setname}\n"
    output["setname"] = newname


def _history_command(types: Any, lim: list[float], options: dict[str, Any]) -> str:
    pieces = [
        f"EEG = pop_epoch( EEG, {_history_types(types)}, "
        f"{format_history_value(lim, cell_for_sequence=None)}"
    ]
    for key, value in options.items():
        if _is_empty_history_value(value):
            continue
        pieces.append(f", {format_history_value(key)}, {format_history_value(value)}")
    return "".join(pieces) + ");"


def _history_types(types: Any) -> str:
    if isinstance(types, str):
        return format_history_value(types)
    values = list(types or [])
    if not values:
        return "{ }"
    return "{ " + " ".join(format_history_value(value) for value in values) + " }"


def _parse_gui_types(text: str) -> Any:
    if not text or text == "[]":
        return []
    return parse_text_tokens(text, parse_ints=True)


def _parse_numeric_vector(text: str, label: str, *, length: int | None) -> list[float]:
    values = _as_float_list(text)
    if length is not None and len(values) != length:
        raise ValueError(f"{label} must contain {length} values")
    return values


def _default_epoch_limits(EEG: dict[str, Any]) -> str:
    data = EEG.get("data")
    is_epoched = False
    if isinstance(data, np.ndarray):
        is_epoched = data.ndim == 3 and data.shape[2] > 1
    is_epoched = is_epoched or int(EEG.get("trials", 1) or 1) > 1
    if is_epoched:
        return f"{round(float(EEG.get('xmin', 0) or 0)):g} {round(float(EEG.get('xmax', 0) or 0)):g}"
    return "-1 2"


def _time_locking_events(EEG: dict[str, Any]) -> list[dict[str, Any]]:
    trials = int(EEG.get("trials", 1))
    pnts = int(EEG.get("pnts", 0))
    xmin = float(EEG.get("xmin", 0) or 0)
    srate = float(EEG.get("srate", 1) or 1)
    return [
        {
            "epoch": trial + 1,
            "type": "TLE",
            "latency": -xmin * srate + 1 + trial * pnts,
        }
        for trial in range(trials)
    ]


def _event_list(events: Any) -> list[dict[str, Any]]:
    if events is None:
        return []
    if isinstance(events, np.ndarray):
        events = events.tolist()
    if isinstance(events, dict):
        return [dict(events)]
    return [dict(event) for event in events]


def _unique_event_types(events: list[dict[str, Any]]) -> list[str]:
    values = []
    seen = set()
    for event in events:
        text = _text_field(event.get("type", ""))
        if text and text not in seen:
            values.append(text)
            seen.add(text)
    return values


def _text_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        if value.size == 1:
            value = value.item()
        else:
            return ""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value if isinstance(value, str) else str(value)


def _as_float_list(value: Any) -> list[float]:
    if isinstance(value, str):
        value = [item for item in re.split(r"[\s,]+", value.strip().strip("[]")) if item]
    elif isinstance(value, np.ndarray):
        value = value.ravel().tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        value = list(value)
    else:
        value = [value]
    return [float(item) for item in value]


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.ravel().tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def _is_empty_sequence(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return value.size == 0
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 0


def _is_empty_history_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    return _is_empty_sequence(value)


def _has_comment_text(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.size > 0
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value) > 0
    return bool(value)


def _latency_between(event: dict[str, Any], low: float, high: float) -> bool:
    try:
        latency = float(event.get("latency"))
    except (TypeError, ValueError):
        return False
    return low < latency < high


def _is_boundary_event(event: dict[str, Any]) -> bool:
    return bool(eeg_findboundaries(EEG=[event]))


def _is_multiple(EEG: Any) -> bool:
    return isinstance(EEG, list) and len(EEG) > 1
