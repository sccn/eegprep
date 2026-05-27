"""EEGLAB-style baseline removal pop function."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any

import numpy as np

from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.miscfunc.misc import round_mat
from eegprep.functions.popfunc._pop_utils import format_history_value, parse_text_tokens
from eegprep.functions.popfunc.eeg_findboundaries import eeg_findboundaries
from eegprep.functions.sigprocfunc.rmbase import rmbase


_UNSET = object()
_RANGE_TOKEN = re.compile(r"^(-?\d+(?:\.\d+)?)(?::(-?\d+(?:\.\d+)?))?(?::(-?\d+(?:\.\d+)?))?$")


def pop_rmbase(
    EEG: Any = _UNSET,
    timerange: Any = None,
    pointrange: Any = None,
    chanlist: Any = None,
    *,
    gui: bool | None = None,
    renderer: Any | None = None,
    return_com: bool = False,
):
    """Remove baseline means from an epoched or continuous EEG dataset.

    ``timerange`` uses the units stored in ``EEG["times"]``. ``pointrange``
    and numeric ``chanlist`` values use EEGLAB-facing 1-based indices.
    Calling ``pop_rmbase(EEG)`` opens the GUI; pass explicit ranges or
    ``gui=False`` for direct command-line use.
    """
    if EEG is _UNSET or EEG is None:
        raise ValueError("pop_rmbase(): EEG dataset is required")
    if gui is None:
        gui = timerange is None and pointrange is None and chanlist is None

    if isinstance(EEG, list):
        if not EEG:
            return ([], "") if return_com else []
        if gui:
            result = _run_gui(EEG[0], renderer=renderer, multiple=True)
            if result is None:
                return (EEG, "") if return_com else EEG
            timerange = result["timerange"]
            pointrange = result["pointrange"]
            chanlist = None
        outputs = [pop_rmbase(item, timerange, pointrange, chanlist, gui=False) for item in EEG]
        command = _history_command(timerange, pointrange, None, None)
        return (outputs, command) if return_com else outputs

    if gui:
        result = _run_gui(EEG, renderer=renderer, multiple=False)
        if result is None:
            return (EEG, "") if return_com else EEG
        timerange = result["timerange"]
        pointrange = result["pointrange"]
        chanlist = result["chanlist"]

    output, history_pointrange, history_chanlist = _apply_pop_rmbase_one(EEG, timerange, pointrange, chanlist)
    history_timerange = timerange if _has_values(timerange) else []
    command = _history_command(history_timerange, history_pointrange, history_chanlist, int(output["nbchan"]))
    return (output, command) if return_com else output


def pop_rmbase_dialog_spec(EEG: dict[str, Any], *, multiple: bool = False) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_rmbase``."""
    trials = int(EEG.get("trials", 1) or 1)
    labels = _channel_field_values(EEG, "labels")
    types = _channel_field_values(EEG, "type", unique=True)
    channel_enabled = not multiple
    controls: list[ControlSpec] = []
    geometry: list[tuple[float, ...]] = []
    if trials > 1:
        controls.extend(
            [
                ControlSpec("text", "Baseline latency range ([min max] in ms) ([ ] = whole epoch):"),
                ControlSpec("edit", tag="timerange", value=_default_baseline_timerange(EEG)),
                ControlSpec("text", "Or remove baseline points vector (ex:1:56):"),
                ControlSpec("edit", tag="pointrange", value=""),
                ControlSpec("text", "Note: press Cancel if you do not want to remove the baseline"),
            ]
        )
        geometry.extend([(3, 1), (3, 1), (1,)])
        size = (716, 299)
    else:
        controls.append(ControlSpec("text", "Removing the mean of each data channel (press cancel to skip)"))
        geometry.append((1,))
        size = (626, 198)

    controls.extend(
        [
            ControlSpec("text", "Channel type(s)", enabled=channel_enabled),
            ControlSpec("edit", tag="chantypes", value="", enabled=channel_enabled),
            ControlSpec(
                "pushbutton",
                "...",
                tag="chantypes_button",
                enabled=channel_enabled and bool(types),
                callback=CallbackSpec(
                    "select_channels",
                    params={"button": "chantypes_button", "target": "chantypes", "channels": types},
                    matlab_callback="pop_chansel({tmpchanlocs.type}, 'withindex', 'off')",
                ),
            ),
            ControlSpec("text", "OR channel(s) (default all)", enabled=channel_enabled),
            ControlSpec("edit", tag="channels", value="", enabled=channel_enabled),
            ControlSpec(
                "pushbutton",
                "...",
                tag="channels_button",
                enabled=channel_enabled and bool(labels),
                callback=CallbackSpec(
                    "select_channels",
                    params={"button": "channels_button", "target": "channels", "channels": labels},
                    matlab_callback="pop_chansel({tmpchanlocs.labels}, 'withindex', 'on')",
                ),
            ),
        ]
    )
    geometry.extend([(2, 1.5, 0.5), (2, 1.5, 0.5)])
    return DialogSpec(
        title="Baseline removal - pop_rmbase()",
        function_name="pop_rmbase",
        eeglab_source="functions/popfunc/pop_rmbase.m",
        geometry=tuple(geometry),
        size=size,
        help_text="pophelp('pop_rmbase')",
        controls=tuple(controls),
        content_margins=(42, 35, 42, 35),
        row_spacing=18,
    )


def _run_gui(EEG: dict[str, Any], *, renderer: Any | None = None, multiple: bool = False) -> dict[str, Any] | None:
    spec = pop_rmbase_dialog_spec(EEG, multiple=multiple)
    result = inputgui(spec, renderer=renderer)
    if result is None:
        return None
    trials = int(EEG.get("trials", 1) or 1)
    timerange: Any = []
    pointrange: Any = []
    if trials > 1:
        timerange_text = str(result.get("timerange", "")).strip()
        pointrange_text = str(result.get("pointrange", "")).strip()
        if _is_blank_text(timerange_text) and _is_blank_text(pointrange_text):
            times = np.asarray(EEG["times"], dtype=float).ravel()
            timerange = [float(times[0]), float(times[-1])]
        else:
            timerange = _parse_numeric_vector(timerange_text)
            pointrange = _parse_numeric_vector(pointrange_text)
    chanlist = None
    if not multiple:
        chantypes = str(result.get("chantypes", "")).strip()
        channels = str(result.get("channels", "")).strip()
        if not _is_blank_text(chantypes):
            chanlist = _resolve_channel_types(EEG, chantypes)
        elif not _is_blank_text(channels):
            chanlist = _resolve_channel_indices(EEG, channels)[1]
    return {"timerange": timerange, "pointrange": pointrange, "chanlist": chanlist}


def _apply_pop_rmbase_one(
    EEG: dict[str, Any],
    timerange: Any,
    pointrange: Any,
    chanlist: Any,
) -> tuple[dict[str, Any], list[int], list[int] | None]:
    _validate_eeg(EEG)
    output = deepcopy(EEG)
    data = np.asarray(output["data"])
    nbchan = int(output.get("nbchan", data.shape[0]))
    pnts = int(output.get("pnts", data.shape[1]))
    trials = int(output.get("trials", data.shape[2] if data.ndim == 3 else 1))
    if data.ndim == 2 and trials > 1:
        data = data.reshape(nbchan, pnts, trials)
    elif data.ndim == 3 and trials <= 1:
        data = data[:, :, 0]
        trials = 1
    elif data.ndim not in {2, 3}:
        raise ValueError("pop_rmbase(): EEG data must be 2D or 3D")

    baseline_indices, history_pointrange = _baseline_indices(output, timerange, pointrange, pnts)
    channel_indices, history_chanlist = _resolve_channel_indices(output, chanlist)
    if data.ndim == 2:
        data = _remove_continuous_baseline(output, data, channel_indices, baseline_indices)
    else:
        data[channel_indices, :, :] = rmbase(data[channel_indices, :, :], pnts, baseline_indices + 1)

    output["data"] = data
    output["nbchan"] = nbchan
    output["pnts"] = pnts
    output["trials"] = trials
    output["icaact"] = np.array([])
    output["saved"] = "no"
    return output, history_pointrange, history_chanlist


def _remove_continuous_baseline(
    EEG: dict[str, Any],
    data: np.ndarray,
    channel_indices: list[int],
    baseline_indices: np.ndarray,
) -> np.ndarray:
    boundaries = _continuous_boundary_starts(EEG, data.shape[1])
    if not boundaries:
        data[channel_indices, :] = rmbase(data[channel_indices, :], data.shape[1], baseline_indices + 1)
        return data
    segment_bounds = [0, *boundaries, data.shape[1]]
    for start, stop in zip(segment_bounds[:-1], segment_bounds[1:]):
        if stop <= start:
            continue
        segment_baseline = baseline_indices[(baseline_indices >= start) & (baseline_indices < stop)]
        if segment_baseline.size == 0:
            continue
        # EEGLAB's boundary path only writes the selected baseline window.
        window = np.ix_(channel_indices, segment_baseline)
        data[window] = rmbase(data[window])
    return data


def _baseline_indices(
    EEG: dict[str, Any],
    timerange: Any,
    pointrange: Any,
    pnts: int,
) -> tuple[np.ndarray, list[int]]:
    if _has_values(timerange):
        times = np.asarray(EEG.get("times"), dtype=float).ravel()
        if times.size == 0:
            raise ValueError('EEG["times"] is required when using timerange')
        values = _parse_numeric_vector(timerange, dtype=float)
        if len(values) != 2:
            raise ValueError("timerange must contain [min max]")
        if values[0] < float(times[0]) or values[-1] > float(times[-1]):
            raise ValueError("pop_rmbase(): Bad time range")
        indices = np.where((times >= values[0]) & (times <= values[-1]))[0]
        if indices.size == 0:
            raise ValueError("pop_rmbase(): time range does not contain samples")
        return indices.astype(int), []
    if _has_values(pointrange):
        values = _parse_numeric_vector(pointrange, dtype=float)
        indices = _point_indices_from_values(values, pnts)
        return indices, (indices + 1).astype(int).tolist()
    indices = np.arange(pnts, dtype=int)
    return indices, []


def _point_indices_from_values(values: list[float], pnts: int) -> np.ndarray:
    if not values:
        return np.arange(pnts, dtype=int)
    indices = np.asarray(values, dtype=int)
    indices = indices[(indices >= 1) & (indices <= pnts)]
    if indices.size == 0:
        raise ValueError("pointrange does not overlap the data")
    return np.unique(indices - 1)


def _continuous_boundary_starts(EEG: dict[str, Any], pnts: int) -> list[int]:
    events = EEG.get("event", [])
    if isinstance(events, np.ndarray):
        events = events.tolist()
    if not isinstance(events, list) or not events:
        return []
    starts = []
    for event_index in eeg_findboundaries(EEG=EEG):
        event = events[event_index]
        try:
            latency = float(event.get("latency"))
        except (AttributeError, TypeError, ValueError):
            continue
        start = int(round_mat(latency - 0.5))
        if 0 < start < pnts:
            starts.append(start)
    return sorted(set(starts))


def _resolve_channel_indices(EEG: dict[str, Any], chanlist: Any) -> tuple[list[int], list[int] | None]:
    nbchan = int(EEG.get("nbchan", np.asarray(EEG.get("data")).shape[0]))
    if not _has_values(chanlist):
        return list(range(nbchan)), None
    tokens = _channel_tokens(chanlist)
    numeric = _numeric_channel_tokens(tokens)
    if numeric is not None:
        indices = [index - 1 for index in numeric]
        history = numeric
        if any(index < 0 or index >= nbchan for index in indices):
            raise ValueError("chanlist indices must be 1-based and within EEG.nbchan")
        return list(dict.fromkeys(indices)), list(dict.fromkeys(history))
    labels = _channel_field_values(EEG, "labels")
    lookup = {label.lower(): index for index, label in enumerate(labels)}
    indices = []
    for token in tokens:
        key = str(token).strip().lower()
        if key not in lookup:
            raise ValueError(f"Channel '{token}' not found")
        indices.append(lookup[key])
    indices = list(dict.fromkeys(indices))
    return indices, [index + 1 for index in indices]


def _resolve_channel_types(EEG: dict[str, Any], text: str) -> list[int]:
    requested = {str(token).strip().lower() for token in parse_text_tokens(text) if str(token).strip()}
    chanlocs = _chanlocs(EEG)
    indices = [
        index + 1 for index, chan in enumerate(chanlocs) if str(chan.get("type", "")).strip().lower() in requested
    ]
    if not indices:
        raise ValueError("No channels matched the selected channel type(s)")
    return indices


def _numeric_channel_tokens(tokens: list[Any]) -> list[int] | None:
    values = []
    for token in tokens:
        if isinstance(token, (int, float, np.integer, np.floating)) and float(token).is_integer():
            values.append(int(token))
            continue
        text = str(token).strip()
        if not text.isdigit():
            return None
        values.append(int(text))
    return values


def _channel_tokens(value: Any) -> list[Any]:
    if isinstance(value, str):
        return parse_text_tokens(value, parse_ints=True)
    if isinstance(value, np.ndarray):
        return value.ravel().tolist()
    if isinstance(value, (int, float, np.integer, np.floating)):
        return [value]
    return list(value)


def _parse_numeric_vector(value: Any, *, dtype: type = float) -> list[Any]:
    if not _has_values(value):
        return []
    if isinstance(value, str):
        text = value.strip().strip("[]")
        if not text:
            return []
        values = []
        for token in text.replace(",", " ").split():
            values.extend(_parse_range_token(token))
        return [dtype(item) for item in values]
    if isinstance(value, np.ndarray):
        values = value.ravel().tolist()
    elif isinstance(value, (int, float, np.integer, np.floating)):
        values = [value]
    else:
        values = list(np.asarray(value).ravel())
    return [dtype(item) for item in values]


def _parse_range_token(token: str) -> list[float]:
    match = _RANGE_TOKEN.match(token)
    if match is None or match.group(2) is None:
        return [float(token)]
    first = float(match.group(1))
    if match.group(3) is None:
        step = 1.0 if float(match.group(2)) >= first else -1.0
        last = float(match.group(2))
    else:
        step = float(match.group(2))
        last = float(match.group(3))
    if step == 0:
        raise ValueError("range step cannot be zero")
    values = []
    current = first
    compare = (lambda left, right: left <= right) if step > 0 else (lambda left, right: left >= right)
    while compare(current, last):
        values.append(current)
        current += step
    return values


def _history_command(timerange: Any, pointrange: Any, chanlist: Any, nbchan: int | None) -> str:
    timerange_text = format_history_value(_history_sequence(timerange), cell_for_sequence=None)
    pointrange_text = format_history_value(_history_sequence(pointrange), cell_for_sequence=None)
    parts = [timerange_text, pointrange_text]
    if _has_values(chanlist):
        chan_values = _history_sequence(chanlist)
        if nbchan is None or chan_values != list(range(1, nbchan + 1)):
            parts.append(format_history_value(chan_values, cell_for_sequence=None))
    return f"EEG = pop_rmbase( EEG, {', '.join(parts)});"


def _history_sequence(value: Any) -> list[Any]:
    if not _has_values(value):
        return []
    if isinstance(value, np.ndarray):
        return value.ravel().tolist()
    if isinstance(value, (int, float, np.integer, np.floating)):
        return [value]
    return list(value)


def _channel_field_values(EEG: dict[str, Any], field: str, *, unique: bool = False) -> list[str]:
    values = [str(chan.get(field, "")) for chan in _chanlocs(EEG)]
    if not unique:
        return values
    deduped = []
    for value in values:
        if value and value not in deduped:
            deduped.append(value)
    return deduped


def _chanlocs(EEG: dict[str, Any]) -> list[dict[str, Any]]:
    chanlocs = EEG.get("chanlocs", [])
    if isinstance(chanlocs, np.ndarray):
        chanlocs = chanlocs.tolist()
    return [chan if isinstance(chan, dict) else {} for chan in list(chanlocs or [])]


def _default_baseline_timerange(EEG: dict[str, Any]) -> str:
    times = np.asarray(EEG.get("times", []), dtype=float).ravel()
    if times.size == 0 or times[0] >= 0:
        return "[ ]"
    return f"{times[0]:g} 0"


def _validate_eeg(EEG: dict[str, Any]) -> None:
    if not isinstance(EEG, dict):
        raise ValueError("pop_rmbase(): EEG must be a dataset dictionary")
    data = EEG.get("data")
    if data is None or np.asarray(data).size == 0:
        raise ValueError("pop_rmbase(): cannot remove baseline of an empty dataset")


def _has_values(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return not _is_blank_text(value)
    if isinstance(value, np.ndarray):
        return value.size > 0
    if isinstance(value, (list, tuple)):
        return len(value) > 0
    return True


def _is_blank_text(text: str) -> bool:
    stripped = text.strip()
    return stripped in {"", "[]", "[ ]"}
