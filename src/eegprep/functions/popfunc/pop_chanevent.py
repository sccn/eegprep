"""Extract event latencies from one or more data channels."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset, strict_mode
from eegprep.functions.popfunc._file_io import events_to_records
from eegprep.functions.popfunc._pop_utils import format_history_value, parse_key_value_args, parse_numeric_sequence


def _channel_indices(chan: Any) -> list[int]:
    if isinstance(chan, bool):
        raise TypeError("chan indices must be integers")
    if isinstance(chan, (int, np.integer)):
        return [int(chan)]
    if isinstance(chan, (float, np.floating)):
        if not float(chan).is_integer():
            raise ValueError("chan indices must be integers")
        return [int(chan)]
    values = chan.reshape(-1) if isinstance(chan, np.ndarray) else chan
    return [int(item) for item in values]


def pop_chanevent(
    EEG: dict[str, Any],
    chan: int | list[int] | tuple[int, ...] | None = None,
    *args: Any,
    return_com: bool = False,
    gui: bool | None = None,
    renderer: Any = None,
    **kwargs: Any,
) -> dict[str, Any] | tuple[dict[str, Any], str]:
    """Import events from rising, falling, or both edges of data channels.

    ``chan`` uses 1-based channel numbers. A whole number selects one channel.
    """
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    if gui is True or (gui is None and chan is None):
        result = inputgui(pop_chanevent_dialog_spec(), renderer=renderer)
        if result is None:
            return (EEG, '') if return_com else EEG
        chan = parse_numeric_sequence(result['chan'], dtype=int)
        options = {'edge': ('leading', 'both', 'trailing')[int(result['edge']) - 1], 'edgelen': int(result['edgelen'])}
        for key in ('duration', 'delchan', 'delevent'):
            options[key] = 'on' if result[key] else 'off'
        if result.get('oper'):
            options['oper'] = result['oper']
        if result['nbtype']:
            options['nbtype'] = 1
    if chan is None or (isinstance(chan, list) and not chan):
        raise ValueError('pop_chanevent: no channel selected. Enter one or more 1-based channel indices.')
    edge = str(options.get("edge", "both")).lower()
    if edge not in {"both", "leading", "trailing"}:
        raise ValueError("edge must be 'both', 'leading', or 'trailing'")
    duration = str(options.get("duration", "off")).lower() in {"on", "yes", "true", "1"}
    if duration and edge != "leading":
        raise ValueError("duration extraction requires leading edges")
    channels = _channel_indices(chan)
    data = np.asarray(EEG["data"])
    if data.ndim != 2:
        raise ValueError("pop_chanevent currently supports continuous 2-D data")
    if any(channel < 1 or channel > data.shape[0] for channel in channels):
        raise ValueError("chan indices must be 1-based and within EEG.nbchan")
    events = []
    for channel in channels:
        x = data[channel - 1, :]
        nbtype = options.get("nbtype", np.nan)
        named_type = not _is_nan(nbtype) or np.unique(x).size == 2
        if "oper" in options and options["oper"]:
            x = _apply_oper(x, str(options["oper"]))
        events.extend(
            _events_from_channel(
                x,
                edge=edge,
                duration=duration,
                edgelen=int(options.get("edgelen", 1)),
                typename=str(options.get("typename", f"chan{channel}")),
                named_type=named_type,
            )
        )
    out = deepcopy(EEG)
    delete_events = str(options.get("delevent", "on")).lower() in {"on", "yes", "true", "1"}
    if delete_events:
        imported_events, imported_urevents = _events_with_new_urevents(events, 0)
        out["event"] = imported_events
        out["urevent"] = imported_urevents
    else:
        original_events, original_urevents = _events_with_existing_urevents(
            events_to_records(out.get("event")),
            events_to_records(out.get("urevent")),
        )
        imported_events, imported_urevents = _events_with_new_urevents(events, len(original_urevents))
        out["event"] = original_events + imported_events
        out["event"].sort(key=lambda item: float(item.get("latency", np.inf)))
        out["urevent"] = original_urevents + imported_urevents
    if str(options.get("delchan", "on")).lower() in {"on", "yes", "true", "1"}:
        keep = [index for index in range(data.shape[0]) if index + 1 not in channels]
        out["data"] = data[keep, :]
        out["nbchan"] = len(keep)
        out["chanlocs"] = [
            loc for index, loc in enumerate(list(out.get("chanlocs", [])), start=1) if index not in channels
        ]
    out["saved"] = "no"
    with strict_mode(False):
        out = eeg_checkset(out)
    command = _history_command(channels, options)
    out["history"] = command if not out.get("history") else f"{out['history'].rstrip()}\n{command}"
    return (out, command) if return_com else out


def _events_from_channel(
    x: np.ndarray,
    *,
    edge: str,
    duration: bool,
    edgelen: int,
    typename: str,
    named_type: bool,
) -> list[dict[str, Any]]:
    values = np.asarray(x)
    diff = np.diff(np.abs(np.r_[values, values[-1]]))
    leading = _keep_first_of_close(np.flatnonzero(diff > 0) + 1, edgelen)
    trailing = _keep_last_of_close(np.flatnonzero(diff < 0) + 2, edgelen)
    if edge == "leading":
        latencies, value_indices = leading, leading
    elif edge == "trailing":
        latencies, value_indices = trailing, trailing - 2
    else:
        # pop_chanevent.m sorts edge positions and sampled-value positions
        # separately, retaining both types when falling and rising edges coincide.
        latencies = np.sort(np.r_[leading, trailing])
        value_indices = np.sort(np.r_[leading, trailing - 2])
    events = []
    for latency, value_index in zip(latencies, value_indices):
        event_type = typename if named_type else values[int(value_index)].item()
        event = {"type": event_type, "latency": int(latency)}
        if duration:
            falling_boundaries = trailing - 1
            next_trailing = falling_boundaries[falling_boundaries >= latency]
            event["duration"] = int(next_trailing[0] - latency) if next_trailing.size else int(values.size - latency)
        events.append(event)
    return events


def _events_with_existing_urevents(
    events: list[dict[str, Any]],
    urevents: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized_urevents = [dict(event) for event in urevents]
    normalized_events = []
    for event in events:
        normalized = dict(event)
        urevent_index = _valid_urevent_index(normalized.get("urevent"), len(normalized_urevents))
        if urevent_index is None:
            urevent_index = len(normalized_urevents)
            normalized_urevents.append(_urevent_record(normalized))
        normalized["urevent"] = urevent_index
        normalized_events.append(normalized)
    return normalized_events, normalized_urevents


def _events_with_new_urevents(
    events: list[dict[str, Any]],
    offset: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized_events = []
    urevents = []
    for index, event in enumerate(events):
        urevent = _urevent_record(event)
        event_with_ref = dict(urevent)
        event_with_ref["urevent"] = offset + index
        normalized_events.append(event_with_ref)
        urevents.append(urevent)
    return normalized_events, urevents


def _valid_urevent_index(value: Any, count: int) -> int | None:
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    return index if 0 <= index < count else None


def _urevent_record(event: dict[str, Any]) -> dict[str, Any]:
    record = dict(event)
    record.pop("urevent", None)
    return record


def _keep_first_of_close(values: np.ndarray, edgelen: int) -> np.ndarray:
    if values.size < 2 or edgelen <= 1:
        return values
    return values[np.r_[True, np.diff(values) >= edgelen]]


def _keep_last_of_close(values: np.ndarray, edgelen: int) -> np.ndarray:
    if values.size < 2 or edgelen <= 1:
        return values
    return values[np.r_[np.diff(values) >= edgelen, True]]


def _is_nan(value: Any) -> bool:
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


def _apply_oper(x: np.ndarray, oper: str) -> np.ndarray:
    if oper.strip() == "X>0":
        return x > 0
    if oper.strip().startswith("X>"):
        return x > float(oper.strip()[2:])
    if oper.strip().startswith("X<"):
        return x < float(oper.strip()[2:])
    raise ValueError("Only simple X>threshold or X<threshold preprocessing expressions are supported")


def _history_command(channels: list[int], options: dict[str, Any]) -> str:
    channel_value: int | list[int] = channels[0] if len(channels) == 1 else channels
    pieces = [format_history_value(channel_value)]
    for key in ["oper", "edge", "edgelen", "duration", "delchan", "delevent", "nbtype", "typename"]:
        if key in options:
            pieces.extend([format_history_value(key), format_history_value(options[key])])
    return f"EEG = pop_chanevent(EEG, {', '.join(pieces)});"


def pop_chanevent_dialog_spec() -> DialogSpec:
    """Return the channel-transition options from pop_chanevent.m."""
    controls = [
        ControlSpec('text', 'Event channel(s)'),
        ControlSpec('edit', tag='chan', value='1'),
        ControlSpec('spacer'),
        ControlSpec('text', 'Preprocessing transform (Ex: X > 0)'),
        ControlSpec('edit', tag='oper'),
        ControlSpec('spacer'),
        ControlSpec('text', 'Transitions to extract'),
        ControlSpec(
            'popupmenu',
            'Up|Both|Down',
            tag='edge',
            value=2,
            callback=CallbackSpec(
                'toggle_index_enabled', {'source': 'edge', 'targets': ['duration'], 'enabled_index': 1}
            ),
        ),
        ControlSpec('spacer'),
        ControlSpec('text', 'Transition length (samples)'),
        ControlSpec('edit', tag='edgelen', value='1'),
        ControlSpec('spacer'),
    ]
    for label, tag, default in [
        ('Assign duration to each event?', 'duration', False),
        ('Delete event channel(s)?', 'delchan', True),
        ('Delete old events if any?', 'delevent', True),
        ('All events of same type?', 'nbtype', False),
    ]:
        controls.extend(
            [ControlSpec('text', label), ControlSpec('checkbox', tag=tag, value=default), ControlSpec('spacer')]
        )
    return DialogSpec(
        title='Extract event from channel(s) - pop_chanevent()',
        function_name='pop_chanevent',
        help_text='pop_chanevent',
        eeglab_source='functions/popfunc/pop_chanevent.m',
        geometry=tuple([[2, 1, 0.5]] * 8),
        controls=tuple(controls),
        weighted_columns=True,
        size=(640, 276),
    )
