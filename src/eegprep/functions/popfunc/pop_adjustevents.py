"""Adjust EEGLAB event latencies."""

from __future__ import annotations

import copy
import logging
from collections.abc import Iterable
from typing import Any

import numpy as np

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset
from eegprep.functions.adminfunc.eeg_options import EEG_OPTIONS
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc._pop_utils import (
    format_history_value,
    parse_key_value_args,
    parse_text_tokens,
)

logger = logging.getLogger(__name__)


_VALID_FORCE = {"on", "off", "auto"}


def pop_adjustevents(
    EEG: dict,
    *args: Any,
    addms: float | None = None,
    addsamples: float | None = None,
    eventtypes: Any = None,
    eventtype: Any = None,
    force: str | None = None,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
):
    """Adjust event offset of all or selected events.

    This mirrors EEGLAB's ``pop_adjustevents`` behavior while using Python data
    structures. When no processing arguments are supplied, the GUI dialog is
    launched by default, matching the MATLAB pop-function convention.
    """
    kwargs_from_args = parse_key_value_args(args)
    addms = _pick_option("addms", addms, kwargs_from_args)
    addsamples = _pick_option("addsamples", addsamples, kwargs_from_args)
    eventtypes = _pick_option("eventtypes", eventtypes, kwargs_from_args)
    eventtype = _pick_option("eventtype", eventtype, kwargs_from_args)
    force = _pick_option("force", force, kwargs_from_args)

    if kwargs_from_args:
        unknown = ", ".join(sorted(kwargs_from_args))
        raise ValueError(f"Unknown pop_adjustevents option(s): {unknown}")

    if eventtypes is None and eventtype is not None:
        eventtypes = eventtype

    has_eventtypes = eventtypes is not None and not (
        isinstance(eventtypes, str) and eventtypes == ""
    )
    has_processing_args = addms is not None or addsamples is not None or has_eventtypes
    if gui is None:
        gui = not has_processing_args

    if gui:
        result = _run_gui(EEG, renderer=renderer)
        if result is None:
            return (EEG, "") if return_com else EEG
        addms = result.get("addms")
        addsamples = result.get("addsamples")
        eventtypes = result.get("eventtypes")
        force = result.get("force", "auto")

    options = _normalize_options(
        addms=addms,
        addsamples=addsamples,
        eventtypes=eventtypes,
        force=force,
        gui=gui,
    )

    EEG_out = copy.deepcopy(EEG)
    events = _events_as_list(EEG_out.get("event", []))
    if not events:
        raise ValueError("Unable to proceed. Event type(s) requested not found")

    _check_force(EEG_out, events, options["force"])
    indices = _event_indices(events, options["eventtypes"])
    if not indices:
        raise ValueError("Unable to proceed. Event type(s) requested not found")

    shift = options["addsamples"]
    if shift is None:
        shift = options["addms"] / 1000.0 * float(EEG_out["srate"])

    for index in indices:
        if "latency" not in events[index]:
            raise ValueError("Event is missing required latency field")
        events[index]["latency"] = float(events[index]["latency"]) + float(shift)

    EEG_out["event"] = _restore_event_container(EEG_out.get("event", []), events)
    EEG_out = eeg_checkset(EEG_out)
    com = _history_command(options)
    return (EEG_out, com) if return_com else EEG_out


def _pick_option(name: str, explicit: Any, options: dict[str, Any]) -> Any:
    if name in options:
        value = options.pop(name)
        if explicit is not None and explicit != value:
            raise ValueError(f"Option {name!r} was provided more than once")
        return value
    return explicit


def _normalize_options(
    *,
    addms: Any,
    addsamples: Any,
    eventtypes: Any,
    force: Any,
    gui: bool,
) -> dict[str, Any]:
    force = "auto" if force is None else force
    force_value = str(force).lower()
    if force_value not in _VALID_FORCE:
        raise ValueError("force must be one of 'on', 'off', or 'auto'")
    if force_value == "auto":
        force_value = "off" if gui else "on"

    addms_value = _empty_to_none(addms)
    addsamples_value = _empty_to_none(addsamples)
    if addms_value is not None and addsamples_value is not None:
        raise ValueError("Specify either addms or addsamples, not both")
    if addms_value is None and addsamples_value is None:
        raise ValueError(
            "To adjust event latencies, you need to specify a number of samples or ms"
        )

    return {
        "addms": None if addms_value is None else float(addms_value),
        "addsamples": None if addsamples_value is None else float(addsamples_value),
        "eventtypes": _normalize_eventtypes(eventtypes),
        "force": force_value,
    }


def _empty_to_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _normalize_eventtypes(eventtypes: Any) -> list[Any]:
    if eventtypes is None:
        return []
    if isinstance(eventtypes, str):
        text = eventtypes.strip()
        if not text or text == "{}":
            return []
        if text.startswith("{") and text.endswith("}"):
            text = text[1:-1]
        return parse_text_tokens(text)
    if isinstance(eventtypes, np.ndarray):
        return _normalize_eventtypes(eventtypes.tolist())
    if isinstance(eventtypes, Iterable) and not isinstance(eventtypes, (bytes, bytearray)):
        return list(eventtypes)
    return [eventtypes]


def _events_as_list(events: Any) -> list[dict[str, Any]]:
    if events is None:
        return []
    if isinstance(events, dict):
        return [events]
    if isinstance(events, np.ndarray):
        events = events.tolist()
    return list(events)


def _restore_event_container(original_events: Any, events: list[dict[str, Any]]) -> Any:
    if isinstance(original_events, np.ndarray):
        return np.asarray(events, dtype=object)
    if isinstance(original_events, dict):
        return events[0] if events else {}
    return events


def _event_indices(events: list[dict[str, Any]], eventtypes: list[Any]) -> list[int]:
    if not eventtypes:
        return list(range(len(events)))

    indices: list[int] = []
    for event_type in eventtypes:
        matches = [
            index for index, event in enumerate(events) if event.get("type") == event_type
        ]
        if not matches:
            logger.warning("Event type %r not found.", event_type)
        indices.extend(matches)
    return indices


def _check_force(EEG: dict, events: list[dict[str, Any]], force: str) -> None:
    if force != "off":
        return
    if int(EEG.get("trials", 1)) > 1:
        raise ValueError(
            "Be careful when adjusting latencies for data epochs. Use the 'force' option to do so."
        )
    if _has_boundary_event(events):
        raise ValueError(
            "Be careful when adjusting latencies when boundary events are present. Use the 'force' option to do so."
        )


def _has_boundary_event(events: list[dict[str, Any]]) -> bool:
    for event in events:
        event_type = event.get("type")
        if isinstance(event_type, str) and event_type.startswith("boundary"):
            return True
        if EEG_OPTIONS.get("option_boundary99") and event_type == -99:
            return True
    return False


def _unique_event_types(events: list[dict[str, Any]]) -> list[Any]:
    values: list[Any] = []
    for event in events:
        value = event.get("type")
        if value not in values:
            values.append(value)
    return values


def pop_adjustevents_dialog_spec(srate: float, event_types: Iterable[object] = ()) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_adjustevents``."""
    event_types = tuple(event_types)
    return DialogSpec(
        title="Adjust event latencies - pop_adjustevents()",
        function_name="pop_adjustevents",
        eeglab_source="functions/popfunc/pop_adjustevents.m",
        size=(784, 264),
        geometry=((1, 0.7, 0.5), (1, 0.7, 0.5), (1, 0.7, 0.5), 1),
        content_margins=(42, 36, 42, 13),
        row_spacing=14,
        known_differences=(
            "Python callback code is explicit; original MATLAB callback strings are kept as metadata.",
        ),
        controls=(
            ControlSpec("text", "Event type(s) to adjust (all by default): "),
            ControlSpec("edit", tag="events", value=""),
            ControlSpec(
                "pushbutton",
                "...",
                tag="events_button",
                callback=CallbackSpec(
                    "select_event_types",
                    params={
                        "button": "events_button",
                        "target": "events",
                        "event_types": event_types,
                    },
                    matlab_callback="pop_chansel(unique({ tmpevent.type }))",
                ),
            ),
            ControlSpec("text", "Add in milliseconds (can be negative)"),
            ControlSpec(
                "edit",
                tag="edit_time",
                value="",
                callback=CallbackSpec(
                    "sync_time_to_samples",
                    params={
                        "source": "edit_time",
                        "target": "edit_samples",
                        "srate": float(srate),
                    },
                    matlab_callback=(
                        "set(findobj('tag','edit_samples'),'string',"
                        "num2str(str2num(get(gcbo,'string'))*srate))"
                    ),
                ),
            ),
            ControlSpec("spacer"),
            ControlSpec("text", "Or add in samples"),
            ControlSpec(
                "edit",
                tag="edit_samples",
                value="",
                callback=CallbackSpec(
                    "sync_samples_to_time",
                    params={
                        "source": "edit_samples",
                        "target": "edit_time",
                        "srate": float(srate),
                    },
                    matlab_callback=(
                        "set(findobj('tag','edit_time'),'string',"
                        "num2str(str2num(get(gcbo,'string'))/srate))"
                    ),
                ),
            ),
            ControlSpec("spacer"),
            ControlSpec(
                "checkbox",
                "Force adjustment even when boundaries are present",
                tag="force",
                value=False,
            ),
        ),
    )


def _run_gui(EEG: dict, renderer: Any | None = None) -> dict[str, Any] | None:
    from eegprep.functions.guifunc.inputgui import inputgui

    events = _events_as_list(EEG.get("event", []))
    spec = pop_adjustevents_dialog_spec(float(EEG.get("srate", 1.0)), _unique_event_types(events))
    result = inputgui(spec, renderer=renderer)
    if result is None:
        return None
    addms = _empty_to_none(result.get("edit_time"))
    addsamples = _empty_to_none(result.get("edit_samples"))
    if addms is not None:
        addsamples = None
    if addms is None and addsamples is None:
        logger.info("Not enough parameters selected")
        return None
    eventtypes = result.get("events", "")
    force = "on" if result.get("force") else "off"
    return {"addms": addms, "addsamples": addsamples, "eventtypes": eventtypes, "force": force}


def _history_command(options: dict[str, Any]) -> str:
    parts: list[str] = []
    if options["addms"] is not None:
        parts.extend(["'addms'", _adjustevents_history_value(options["addms"])])
    if options["addsamples"] is not None:
        parts.extend(["'addsamples'", _adjustevents_history_value(options["addsamples"])])
    if options["eventtypes"]:
        parts.extend(["'eventtypes'", _adjustevents_history_value(options["eventtypes"])])
    if options["force"] != "auto":
        parts.extend(["'force'", _adjustevents_history_value(options["force"])])
    return f"[EEG,com] = pop_adjustevents(EEG, {', '.join(parts)});"


def _adjustevents_history_value(value: Any) -> str:
    return format_history_value(value, cell_for_sequence="always", empty_sequence="{}")


__all__ = ["pop_adjustevents"]
