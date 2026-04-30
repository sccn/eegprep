"""GUI spec for EEGLAB's pop_adjustevents dialog."""

from __future__ import annotations

from collections.abc import Iterable

from eegprep.gui.spec import CallbackSpec, ControlSpec, DialogSpec


def pop_adjustevents_dialog_spec(srate: float, event_types: Iterable[object] = ()) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_adjustevents``."""

    event_types = tuple(event_types)
    return DialogSpec(
        title="Adjust event latencies - pop_adjustevents()",
        function_name="pop_adjustevents",
        eeglab_source="functions/popfunc/pop_adjustevents.m",
        geometry=([1, 0.7, 0.5], [1, 0.7, 0.5], [1, 0.7, 0.5], 1),
        known_differences=(
            "Python callback code is explicit; original MATLAB callback strings are kept only as metadata.",
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
                    params={"button": "events_button", "target": "events", "event_types": event_types},
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
                    params={"source": "edit_time", "target": "edit_samples", "srate": float(srate)},
                    matlab_callback="set(findobj('tag','edit_samples'),'string',num2str(str2num(get(gcbo,'string'))*srate))",
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
                    params={"source": "edit_samples", "target": "edit_time", "srate": float(srate)},
                    matlab_callback="set(findobj('tag','edit_time'),'string',num2str(str2num(get(gcbo,'string'))/srate))",
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
