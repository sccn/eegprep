"""Remove or keep continuous data portions around events."""

from __future__ import annotations

from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc._event_utils import events_as_list, is_boundary_event
from eegprep.functions.popfunc._pop_utils import format_history_value, parse_text_tokens
from eegprep.functions.popfunc.pop_select import pop_select


def pop_rmdat(
    EEG: dict[str, Any] | list[dict[str, Any]],
    events: Any = None,
    timelims: Any = None,
    invertsel: int | bool = 1,
    *,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
):
    """Keep or remove continuous data around event types.

    ``invertsel=1`` removes selected windows, matching current EEGLAB. Passing
    ``invertsel=0`` keeps the selected windows.
    """
    if gui is None:
        gui = events is None and timelims is None
    if gui:
        result = _run_gui(EEG[0] if isinstance(EEG, list) else EEG, renderer=renderer)
        if result is None:
            return (EEG, "") if return_com else EEG
        events = result["events"]
        timelims = result["timelims"]
        invertsel = result["invertsel"]
    events = [] if events is None else _normalise_event_types(events)
    timelims = [-1, 1] if timelims is None else _normalise_timelims(timelims)
    if isinstance(EEG, list):
        outputs = [pop_rmdat(item, events, timelims, invertsel, gui=False) for item in EEG]
        command = _history_command(events, timelims, invertsel)
        return (outputs, command) if return_com else outputs
    output = _apply_rmdat_one(EEG, events, timelims, bool(invertsel))
    command = _history_command(events, timelims, invertsel)
    return (output, command) if return_com else output


def pop_rmdat_dialog_spec(EEG: dict[str, Any]) -> DialogSpec:
    """Return the EEGLAB-like ``pop_rmdat`` dialog spec."""
    event_types = tuple(
        sorted({str(event.get("type")) for event in events_as_list(EEG.get("event")) if event.get("type")})
    )
    return DialogSpec(
        title="Remove data portions around events - pop_rmdat()",
        function_name="pop_rmdat",
        eeglab_source="functions/popfunc/pop_rmdat.m",
        size=(616, 221),
        content_margins=(42, 30, 42, 30),
        row_spacing=8,
        help_text="pophelp('pop_rmdat')",
        geometry=((2, 1, 1.2), (2, 1, 1.2)),
        controls=(
            ControlSpec("text", "Event type(s) ([]=all)"),
            ControlSpec("edit", tag="events", value=""),
            ControlSpec(
                "pushbutton",
                "...",
                tag="events_button",
                callback=CallbackSpec(
                    "select_event_types",
                    params={"button": "events_button", "target": "events", "event_types": event_types},
                ),
            ),
            ControlSpec("text", "Time limits [start, end] in sec."),
            ControlSpec("edit", tag="timelims", value="-1 1"),
            ControlSpec("popupmenu", "Keep selected|Remove selected", tag="mode", value=2),
        ),
    )


def _run_gui(EEG: dict[str, Any], *, renderer: Any | None = None) -> dict[str, Any] | None:
    result = inputgui(pop_rmdat_dialog_spec(EEG), renderer=renderer)
    if result is None:
        return None
    events_text = str(result.get("events") or "").strip()
    return {
        "events": [] if events_text in {"", "[]"} else _normalise_event_types(parse_text_tokens(events_text)),
        "timelims": _normalise_timelims(result.get("timelims")),
        "invertsel": int(result.get("mode") or 2) - 1,
    }


def _apply_rmdat_one(EEG: dict[str, Any], events: list[Any], timelims: list[float], invertsel: bool) -> dict[str, Any]:
    event_list = events_as_list(EEG.get("event"))
    if not event_list:
        raise ValueError("No event. This function removes data based on event latencies")
    if int(EEG.get("trials", 1) or 1) != 1:
        raise ValueError("This function only works with continuous data")
    if not any("latency" in event for event in event_list):
        raise ValueError("Absent latency field in event array/structure: must name one of the fields 'latency'")
    wanted = {str(event) for event in events}
    selected = [index for index, event in enumerate(event_list) if not wanted or str(event.get("type", "")) in wanted]
    if not selected:
        return EEG
    windows = _event_windows(EEG, event_list, selected, timelims)
    if not windows:
        return EEG
    option = "notime" if invertsel else "time"
    return pop_select(EEG, option, windows)


def _event_windows(
    EEG: dict[str, Any],
    events: list[dict[str, Any]],
    selected: list[int],
    timelims: list[float],
) -> list[list[float]]:
    srate = float(EEG.get("srate", 1))
    pnts = int(EEG.get("pnts", 0))
    boundary_latencies = [float(event.get("latency")) for event in events if is_boundary_event(event)]
    windows: list[list[float]] = []
    selected = sorted(selected, key=lambda item: float(events[item].get("latency", np.inf)))
    for index in selected:
        latency = float(events[index]["latency"])
        start = latency + srate * float(timelims[0])
        stop = latency + srate * float(timelims[1])
        for boundary in boundary_latencies:
            if start < boundary < stop:
                if boundary < latency:
                    start = boundary
                elif boundary > latency:
                    stop = boundary
        start = max(1, start)
        stop = min(pnts, stop)
        if stop <= start:
            continue
        if windows and start < windows[-1][1]:
            windows[-1][1] = max(windows[-1][1], stop)
        else:
            windows.append([start, stop])
    return [[(start - 1) / srate, (stop - 1) / srate] for start, stop in windows]


def _normalise_event_types(events: Any) -> list[Any]:
    if isinstance(events, np.ndarray):
        events = events.tolist()
    if isinstance(events, (list, tuple, set)):
        return list(events)
    return [events]


def _normalise_timelims(value: Any) -> list[float]:
    if isinstance(value, np.ndarray):
        value = value.ravel().tolist()
    if isinstance(value, str):
        value = [float(token) for token in value.strip().strip("[]").replace(",", " ").split() if token]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Time limits must contain [start, end]")
    limits = [float(value[0]), float(value[1])]
    if limits[0] >= limits[1]:
        raise ValueError("Time limit start must be lower than end")
    return limits


def _history_command(events: list[Any], timelims: list[float], invertsel: int | bool) -> str:
    return (
        "EEG = pop_rmdat( EEG, "
        + ", ".join(
            (
                format_history_value(events, cell_for_sequence="any_strings"),
                format_history_value(timelims),
                format_history_value(int(bool(invertsel))),
            )
        )
        + ");"
    )


__all__ = ["pop_rmdat", "pop_rmdat_dialog_spec"]
