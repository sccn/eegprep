"""EEGLAB ``pop_erpimage``: ERP image of a channel or component, with its dialog."""

from __future__ import annotations

import ast
import inspect
import logging
from typing import Any

import numpy as np

from eegprep.functions.adminfunc.eeg_getdatact import eeg_getdatact
from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.miscfunc.value_parsing import is_empty_value, is_on, parse_text_tokens
from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.popfunc._event_utils import events_as_list
from eegprep.functions.popfunc.eeg_decodechan import eeg_decodechan
from eegprep.functions.popfunc.eeg_getepochevent import eeg_getepochevent, event_type_text
from eegprep.functions.popfunc.plot_utils import (
    channel_labels,
    component_map_data,
    eeg_epoch_data,
    numeric_vector,
    parse_plot_options_text,
    python_literal,
    show_figures,
)
from eegprep.functions.sigprocfunc.erpimage import ErpimageResult, erpimage
from eegprep.functions.sigprocfunc.erpimage_engine import erpimage_option_items

logger = logging.getLogger(__name__)

DEFAULT_SMOOTH = 5
DEFAULT_DECIMATE = 0
DIALOG_DECIMATE = 1
MAX_DIALOG_SMOOTH = 10
MICROVOLT_LABEL = "\\muV"
NO_RENORM = "no"
COLOR_LIMITS_HELP = "Color axis limits [low high], or the fraction of the data range to use"
ARGUMENT_NAMES = (
    "channel",
    "projchan",
    "titleplot",
    "smooth",
    "decimate",
    "sortingtype",
    "sortingwin",
    "sortingeventfield",
)
CHECKBOX_OPTIONS = ("erp", "cbar", "nosort", "noplot", "plotamps")
# The dialog always sets these, so More options cannot carry them.
DIALOG_SET_OPTIONS = ("erp", "cbar", "yerplabel")


def pop_erpimage(
    EEG: dict[str, Any] | None = None,
    typeplot: int = 1,
    channel: Any = None,
    projchan: Any = None,
    titleplot: str = "",
    smooth: Any = None,
    decimate: Any = None,
    sortingtype: Any = None,
    sortingwin: Any = None,
    sortingeventfield: str = "",
    *options: Any,
    gui: bool | None = None,
    renderer: Any | None = None,
    plot: str | bool = "on",
    target: Any = None,
    rng: Any = None,
    lastcom: str = "",
    return_com: bool = False,
    **kwargs: Any,
) -> ErpimageResult | tuple[ErpimageResult | None, str] | None:
    """Plot an ERP image of channel or component activity, as EEGLAB ``pop_erpimage``.

    The positional arguments follow EEGLAB: ``typeplot`` is 1 for channels and
    0 for components, ``channel`` the 1-based channel(s) (numbers or labels)
    or component(s), ``projchan`` the channel(s) to back-project components
    onto, then the title, the moving-average width ``smooth`` (trials; 5 by
    default), the window advance ``decimate`` (0 by default), and the events
    that sort the trials:
    ``sortingtype`` (event types), ``sortingwin`` ([start end] in ms) and
    ``sortingeventfield`` (for example ``'latency'``). Remaining arguments are
    :func:`~eegprep.erpimage` options, as EEGLAB tokens or keywords
    (``'phasesort', [0, 25, 10]`` or ``phasesort=[0, 25, 10]``).

    Unlike a bare ``erpimage`` call, the ERP trace, the colorbar and the scalp
    map are on by default, as in the dialog; turn them off with ``erp=False``,
    ``cbar=False`` or ``plotmap=False``. Without ``channel`` (or with
    ``gui=True``) the dialog opens, filled with the given arguments and
    options; its values then replace them. Options no dialog field can show
    (for example ``topo`` with its own channel locations) are applied as given.

    Args:
        gui: open the dialog; defaults to an empty ``channel``.
        renderer: dialog renderer used instead of the Qt dialog.
        plot: ``'off'`` builds the figure without displaying it.
        target: Matplotlib Figure or SubFigure to draw into.
        rng: seed or NumPy random generator for the coherence bootstrap and
            the ``erpalpha`` permutations; only an integer seed is written to
            the history command.
        lastcom: an earlier ``pop_erpimage`` history command whose arguments
            fill the dialog, as EEGLAB's ``lastcom``; pass no other arguments
            with it.
        return_com: also return the Python history command.

    Returns:
        The :class:`~eegprep.functions.sigprocfunc.erpimage.ErpimageResult`, or
        ``(result, command)`` with ``return_com=True``; ``None`` (or
        ``(None, "")``) when the dialog is cancelled.
    """
    if EEG is None:
        return (None, "") if return_com else None
    typeplot = _plot_type(typeplot)
    _check_dataset(EEG, typeplot)
    show_figures(None, plot=plot)  # rejects an invalid 'plot' before anything is drawn
    arguments = {
        "channel": channel,
        "projchan": projchan,
        "titleplot": titleplot,
        "smooth": smooth,
        "decimate": decimate,
        "sortingtype": sortingtype,
        "sortingwin": sortingwin,
        "sortingeventfield": sortingeventfield,
    }
    if gui is None:
        gui = is_empty_value(channel)
    if lastcom:
        if not all(is_empty_value(value) for value in arguments.values()) or options or kwargs:
            raise ValueError("pop_erpimage(): lastcom holds the arguments; pass no others with it")
        arguments, options, kwargs = _previous_call(lastcom)
    plotmap = _plotmap_flag(kwargs.pop("plotmap", True))
    user_options = _ordered_options(options, kwargs)
    if gui:
        call = _run_gui(EEG, typeplot, arguments, user_options, plotmap=plotmap, renderer=renderer)
        if call is None:
            return (None, "") if return_com else None
        arguments, user_options, plotmap = call
    channel, projchan, titleplot = arguments["channel"], arguments["projchan"], arguments["titleplot"]
    smooth = DEFAULT_SMOOTH if arguments["smooth"] is None else arguments["smooth"]
    decimate = DEFAULT_DECIMATE if arguments["decimate"] is None else arguments["decimate"]
    sortingtype, sortingwin = arguments["sortingtype"], arguments["sortingwin"]
    sortingeventfield = arguments["sortingeventfield"]
    channels = _selected_items(EEG, channel, typeplot)
    projection = [] if typeplot else _items(projchan)
    projection_numbers = _channel_numbers(EEG, projection, "projection channel") if projection else []
    if not typeplot and len(channels) > 1 and not projection:
        raise ValueError("pop_erpimage(): select a channel to plot the sum of several component projections")
    data = _activity(EEG, typeplot, channels, projection_numbers)
    times = np.linspace(float(EEG["xmin"]) * 1000, float(EEG["xmax"]) * 1000, data.shape[0])
    sortvar = _sorting_variable(EEG, sortingtype, sortingwin, sortingeventfield)
    defaults = _default_options(EEG, typeplot, channels, projection, plotmap=plotmap)
    settings = {name: value for name, value in defaults.items() if name not in user_options} | user_options
    title = (
        settings.pop("title")
        if "title" in settings
        else titleplot or _default_title(EEG, typeplot, channels, projection)
    )
    command = _history_command(
        EEG,
        typeplot,
        channel,
        {
            "projchan": projchan,
            "titleplot": titleplot,
            "smooth": smooth,
            "decimate": decimate,
            "sortingtype": sortingtype,
            "sortingwin": sortingwin,
            "sortingeventfield": sortingeventfield,
        },
        user_options,
        plotmap=plotmap,
        rng=rng,
    )
    result = erpimage(data, sortvar, times, title, smooth, decimate, target=target, rng=rng, **settings)
    # A caller's target figure is the caller's to show or close.
    show_figures(None if target is not None else result.figure, plot=plot)
    return (result, command) if return_com else result


def _ordered_options(tokens: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """erpimage options keyed by name, in the order of their last occurrence (a later token wins, as in EEGLAB)."""
    ordered: dict[str, Any] = {}
    for name, value in erpimage_option_items(tokens, **kwargs):
        ordered.pop(name, None)
        ordered[name] = value
    return ordered


def _plot_type(typeplot: Any) -> int:
    if isinstance(typeplot, bool) or typeplot not in (0, 1):
        raise ValueError(f"pop_erpimage(): typeplot must be 1 (channels) or 0 (components), got {typeplot!r}")
    return int(typeplot)


def _check_dataset(EEG: dict[str, Any], typeplot: int) -> None:
    if not typeplot and not all(_is_matrix(EEG.get(name)) for name in ("icaweights", "icasphere")):
        raise ValueError("pop_erpimage(): no ICA data for this set, first run ICA")
    if int(EEG.get("trials", 1) or 1) <= 1:
        raise ValueError("pop_erpimage(): an ERP image needs epoched data with more than one trial")


def _is_matrix(value: Any) -> bool:
    return value is not None and np.asarray(value).ndim == 2 and np.asarray(value).size > 0


def _plotmap_flag(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"pop_erpimage(): plotmap must be True or False, got {value!r}")


def _selected_items(EEG: dict[str, Any], value: Any, typeplot: int) -> list[int]:
    """1-based channels (numbers or labels) or components to plot."""
    kind = "channel" if typeplot else "component"
    items = _items(value)
    if not items:
        raise ValueError(f"pop_erpimage(): select at least one {kind}")
    if typeplot:
        return _channel_numbers(EEG, items, "channel")
    labels = [item for item in items if isinstance(item, str)]
    if labels:
        raise ValueError(f"pop_erpimage(): components are given by number, got {labels[0]!r}")
    count = np.asarray(EEG["icaweights"]).shape[0]
    return _checked_numbers(items, count, kind)


def _items(value: Any) -> list[Any]:
    """Numbers and labels in a value; text splits at spaces and commas, keeps quoted labels and expands '3:5'."""
    if value is None:
        return []
    if isinstance(value, str):
        raw: list[Any] = parse_text_tokens(value.strip().strip("[]{}"))
    else:
        raw = list(np.ravel(np.asarray(value, dtype=object)))
    items: list[Any] = []
    for item in raw:
        if not isinstance(item, str):
            items.append(item)
            continue
        try:
            items.extend(numeric_vector(item).tolist())
        except ValueError:
            items.append(item)
    return items


def _channel_numbers(EEG: dict[str, Any], items: list[Any], kind: str) -> list[int]:
    """1-based channel numbers; labels are looked up as EEGLAB's eeg_decodechan does (ignoring case)."""
    chanlocs = chanlocs_as_list(EEG.get("chanlocs", []))
    numbers: list[Any] = []
    for item in items:
        if not isinstance(item, str):
            numbers.append(item)
            continue
        try:
            indices, _labels = eeg_decodechan(chanlocs, [item])
        except ValueError as error:
            raise ValueError(f"pop_erpimage(): unknown {kind} label {item!r}") from error
        numbers.extend(index + 1 for index in indices)
    return _checked_numbers(numbers, len(channel_labels(EEG)), kind)


def _checked_numbers(items: list[Any], count: int, kind: str) -> list[int]:
    numbers = []
    for item in items:
        try:
            number = float(item)
        except (TypeError, ValueError) as error:
            raise ValueError(f"pop_erpimage(): {kind}s are given by number or label, got {item!r}") from error
        if not number.is_integer():
            raise ValueError(f"pop_erpimage(): {kind} numbers must be whole numbers, got {item!r}")
        if not 1 <= number <= count:
            raise ValueError(f"pop_erpimage(): {kind} {int(number)} is outside 1..{count}")
        numbers.append(int(number))
    return numbers


def _activity(EEG: dict[str, Any], typeplot: int, channels: list[int], projection: list[Any]) -> np.ndarray:
    """(frames, trials) activity: the mean of the channels, or component activity or its projection."""
    if typeplot:
        return np.mean(eeg_epoch_data(EEG)[np.asarray(channels) - 1], axis=0)
    activity = eeg_getdatact(EEG, component=channels, projchan=projection or None)
    return np.mean(activity, axis=0) if projection else activity[0]


def _sorting_variable(EEG: dict[str, Any], sortingtype: Any, sortingwin: Any, field: Any) -> np.ndarray:
    """Per-trial sorting values from ``eeg_getepochevent``, or EEGLAB's constant ``xmax`` (ms) without a field."""
    trials = int(EEG["trials"])
    field = str(field or "").strip()
    if not field:
        if not (is_empty_value(sortingtype) and is_empty_value(sortingwin)):
            logger.warning(
                "pop_erpimage(): sortingtype and sortingwin need sortingeventfield; the trials are not sorted."
            )
        return np.full(trials, float(EEG["xmax"]) * 1000)
    events = events_as_list(EEG.get("event", []))
    if not events:
        raise ValueError("pop_erpimage(): the dataset has no events to sort the trials by")
    if not any(field in event for event in events):
        raise ValueError(f"pop_erpimage(): events have no '{field}' field to sort the trials by")
    if trials > 1 and not any("epoch" in event for event in events):
        raise ValueError("pop_erpimage(): events have no 'epoch' field; epoch the data before sorting by events")
    types = _event_types(sortingtype)
    window = _number_list(sortingwin, "sortingwin")
    if window.size not in (0, 2):
        raise ValueError(f"pop_erpimage(): sortingwin must be [start end] in ms, got {window.tolist()}")
    values, all_values = eeg_getepochevent(EEG, type=types, timewin=window.tolist(), fieldname=field)
    selection = _event_selection_text(types, window, field)
    if not isinstance(values, np.ndarray):
        unusable = [index for index, value in enumerate(values, start=1) if not _converts_to_float(value)]
        raise ValueError(f"pop_erpimage(): {selection}s that are not numbers or text in epochs {unusable[:10]}")
    missing = np.flatnonzero(np.isnan(values)) + 1
    if missing.size == trials:
        raise ValueError(f"pop_erpimage(): no epoch has an {selection}")
    if missing.size:
        logger.warning(
            "pop_erpimage(): %d of %d epochs have no %s (epochs %s, counted from 1); erpimage leaves them out.",
            missing.size,
            trials,
            selection,
            missing.tolist(),
        )
    repeated = [index for index, epoch_values in enumerate(all_values, start=1) if len(epoch_values) > 1]
    if field == "latency" and repeated:
        logger.warning("pop_erpimage(): multiple event latencies found in epochs %s; the first is used.", repeated)
    return values


def _event_types(sortingtype: Any) -> list[Any]:
    """Event types to sort by; a string is one type."""
    if sortingtype is None:
        return []
    if isinstance(sortingtype, str):
        return [sortingtype]
    return [
        item.item() if isinstance(item, np.generic) else item
        for item in np.ravel(np.asarray(sortingtype, dtype=object))
    ]


def _event_selection_text(types: list[Any], window: np.ndarray, field: str) -> str:
    text = f"event '{field}' value"
    if types:
        text += " for event type " + " ".join(repr(event_type_text(item)) for item in types)
    if window.size:
        text += f" between {window[0]:g} and {window[1]:g} ms"
    return text


def _converts_to_float(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _number_list(value: Any, name: str) -> np.ndarray:
    if value is None or (isinstance(value, str) and not value.strip()):
        return np.zeros(0)
    try:
        return np.asarray(value, dtype=float).ravel()
    except (TypeError, ValueError) as error:
        raise ValueError(f"pop_erpimage(): {name} must be numbers, got {value!r}") from error


def _default_options(
    EEG: dict[str, Any], typeplot: int, channels: list[int], projection: list[Any], *, plotmap: bool
) -> dict[str, Any]:
    """The dialog's decorations: ERP trace, colorbar, ERP label and scalp map."""
    options: dict[str, Any] = {
        "erp": True,
        "cbar": True,
        "yerplabel": MICROVOLT_LABEL if typeplot or projection else "",
    }
    scalp_map = _scalp_map(EEG, typeplot, channels) if plotmap else None
    if scalp_map is not None:
        options["topo"] = scalp_map
    return options


def _scalp_map(EEG: dict[str, Any], typeplot: int, channels: list[int]) -> tuple[Any, list[dict[str, Any]]] | None:
    """The channel's location or the (mean) component map, when the channels have locations."""
    chanlocs = chanlocs_as_list(EEG.get("chanlocs", []))
    if not any(_located(location) for location in chanlocs):
        return None
    if not typeplot:
        maps, map_chanlocs = component_map_data(EEG)
        if not any(_located(location) for location in map_chanlocs):
            logger.warning("pop_erpimage(): the ICA channels have no locations, so the scalp map is left out.")
            return None
        return np.mean(maps[:, np.asarray(channels) - 1], axis=1), map_chanlocs
    if len(channels) > 1:
        logger.warning(
            "pop_erpimage(): the scalp map marks one channel, so it is left out for %d channels.", len(channels)
        )
        return None
    if not _located(chanlocs[channels[0] - 1]):
        logger.warning("pop_erpimage(): channel %d has no location, so the scalp map is left out.", channels[0])
        return None
    return channels[0], chanlocs


def _located(location: dict[str, Any]) -> bool:
    return _converts_to_float(location.get("theta")) and bool(np.isfinite(float(location["theta"])))


def _default_title(EEG: dict[str, Any], typeplot: int, channels: list[int], projection: list[Any]) -> str:
    if typeplot:
        labels = channel_labels(EEG)
        return " ".join(labels[index - 1] for index in channels)
    title = "Comp. " + " ".join(str(index) for index in channels)
    if projection:
        title += " -> Chan. " + " ".join(str(item) for item in projection)
    return title


def _history_command(
    EEG: dict[str, Any],
    typeplot: int,
    channel: Any,
    arguments: dict[str, Any],
    options: dict[str, Any],
    *,
    plotmap: bool,
    rng: Any,
) -> str:
    """Replayable Python call; empty arguments and the default decorations are omitted."""
    pieces = ["EEG", str(typeplot), python_literal(channel)]
    for name, value in arguments.items():
        if name in {"smooth", "decimate"} or not is_empty_value(value):
            pieces.append(f"{name}={python_literal(value)}")
    for name, value in options.items():
        pieces.append(f"{name}={_option_literal(EEG, name, value)}")
    if not plotmap:
        pieces.append("plotmap=False")
    if isinstance(rng, (int, np.integer)):
        pieces.append(f"rng={int(rng)}")
    return f"pop_erpimage({', '.join(pieces)})"


def _option_literal(EEG: dict[str, Any], name: str, value: Any) -> str:
    """Python literal of an option; a scalp map drawn on the dataset's channels names ``EEG['chanlocs']``."""
    if name == "topo" and isinstance(value, (list, tuple)) and len(value) >= 2 and _same_channels(value[1], EEG):
        return f"({python_literal(value[0])}, EEG['chanlocs'])"
    return python_literal(value)


def _same_channels(chanlocs: Any, EEG: dict[str, Any]) -> bool:
    def labels(locations: Any) -> list[str]:
        return [str(location.get("labels")) for location in chanlocs_as_list(locations)]

    return chanlocs is EEG.get("chanlocs") or labels(chanlocs) == labels(EEG.get("chanlocs", []))


def pop_erpimage_dialog_spec(EEG: dict[str, Any], *, typeplot: int = 1) -> DialogSpec:
    """Return the EEGLAB ``pop_erpimage`` dialog spec (channel or component variant)."""
    is_channel = bool(int(typeplot))
    smooth = min(max(int(EEG.get("trials", 1) or 1) - 5, 0), MAX_DIALOG_SMOOTH)
    labels = channel_labels(EEG)
    events = events_as_list(EEG.get("event", []))
    field_names = list(dict.fromkeys(name for event in events for name in event))
    event_types = _unique_types(event["type"] for event in events if "type" in event)
    controls: list[ControlSpec] = [
        ControlSpec("text", "Channel" if is_channel else "Component(s)", font_weight="bold"),
        ControlSpec("edit", tag="chan", value="1"),
    ]
    if is_channel:
        controls.extend(
            [
                _channel_button("chan_button", "chan", labels, enabled=bool(chanlocs_as_list(EEG.get("chanlocs")))),
                ControlSpec("text", "Fig. title", font_weight="bold"),
                ControlSpec("edit", tag="title", value=""),
            ]
        )
    else:
        controls.extend(
            [
                ControlSpec("spacer"),
                ControlSpec("spacer"),
                ControlSpec("spacer"),
                ControlSpec(
                    "text",
                    "Project to channel #",
                    font_weight="bold",
                    tooltip=(
                        "Project component(s) to data channel\n"
                        "This allows plotting projected component activity at one channel in microvolts"
                    ),
                ),
                ControlSpec("edit", tag="projchan", value=""),
                _channel_button("projchan_button", "projchan", labels, enabled=bool(labels)),
                ControlSpec("text", "Fig. title", font_weight="bold"),
                ControlSpec("edit", tag="title", value=""),
            ]
        )
    controls.extend(
        [
            ControlSpec(
                "text",
                "Smoothing",
                font_weight="bold",
                tooltip="Number of trials smoothed (vertically) by the moving average",
            ),
            ControlSpec("edit", tag="smooth", value=str(smooth)),
            ControlSpec(
                "checkbox",
                "Plot scalp map",
                tag="plotmap",
                value=True,
                tooltip="plot a 2-d head map (vector) at upper left",
            ),
            ControlSpec("spacer"),
            ControlSpec("spacer"),
            ControlSpec(
                "text",
                "Downsampling",
                font_weight="bold",
                tooltip="Moving-window advance in trials; values above sqrt(trials) give that many image rows",
            ),
            ControlSpec("edit", tag="decimate", value=str(DIALOG_DECIMATE)),
            ControlSpec("checkbox", "Plot ERP", tag="erp", value=True, tooltip="Plot the ERP below the ERP image"),
            ControlSpec(
                "text",
                "ERP limits (uV)" if is_channel else "ERP limits",
                tooltip="Plotting limits for ERP trace [min_uV max_uV]\n{Default: ERP data limits}",
            ),
            ControlSpec("edit", tag="limerp", value=""),
            ControlSpec("text", "Time limits (ms)", font_weight="bold", tooltip="Select time subset in ms"),
            ControlSpec("edit", tag="limtime", value=_epoch_limits_text(EEG)),
            ControlSpec("checkbox", "Plot colorbar", tag="cbar", value=True, tooltip=COLOR_LIMITS_HELP),
            ControlSpec("text", "Color limits (see Help)", tooltip=COLOR_LIMITS_HELP),
            ControlSpec("edit", tag="caxis", value=""),
            ControlSpec("spacer"),
            ControlSpec("text", "Sort/align trials by epoch event values", font_weight="bold"),
            ControlSpec(
                "pushbutton",
                "Epoch-sorting field",
                tag="field_button",
                tooltip="Epoch-sorting event field name (Ex: latency; default: no sorting):",
                callback=CallbackSpec(
                    "select_event_field",
                    params={
                        "button": "field_button",
                        "target": "field",
                        "fields": field_names,
                        "reminder_source": "type",
                        "reminder": "Do not forget to select an event type in the next edit box",
                    },
                    matlab_callback="listdlg2('PromptString', 'Select fields', 'SelectionMode', 'single', ...)",
                ),
            ),
            ControlSpec(
                "pushbutton",
                "Event type(s)",
                tag="type_button",
                tooltip="Event type(s) subset (default: all):\n(See '/Edit/Edit event values' for event types)",
                callback=CallbackSpec(
                    "select_event_types",
                    params={
                        "button": "type_button",
                        "target": "type",
                        "event_types": event_types,
                        "selectionmode": "multiple",
                        "no_types_message": "No type field",
                        "reminder_source": "field",
                        "reminder": "Do not forget to select an epoch-sorting field in the previous edit box",
                    },
                    matlab_callback="pop_chansel(unique({tmpevent.type}))",
                ),
            ),
            ControlSpec(
                "text",
                "Event time range",
                tooltip=(
                    "Sorting event window [start, end] in milliseconds (default: whole epoch):\n"
                    "events are only selected within this time window (can be useful if several\n"
                    "events of the same type are in the same epoch, or for selecting trials with given response time)"
                ),
            ),
            ControlSpec("text", "Rescale", tooltip="Rescale sorting variable to plot window (yes|no|a*x+b)(Ex:3*x+2):"),
            ControlSpec(
                "text",
                "Align",
                tooltip="Align the trials on this latency (ms), or Inf for the median sorting value",
            ),
            ControlSpec(
                "checkbox",
                "Don't sort by value",
                tag="nosort",
                value=False,
                tooltip="Keep the trials in their order but plot the event values",
            ),
            ControlSpec("edit", tag="field", value=""),
            ControlSpec("edit", tag="type", value=""),
            ControlSpec("edit", tag="eventrange", value=""),
            ControlSpec("edit", tag="renorm", value=NO_RENORM),
            ControlSpec("edit", tag="align", value=""),
            ControlSpec(
                "checkbox",
                "Don't plot values",
                tag="noplot",
                value=False,
                tooltip="Sort the trials by the event values without drawing them",
            ),
            ControlSpec("spacer"),
            ControlSpec("text", "Sort trials by phase", font_weight="bold"),
            ControlSpec(
                "text",
                "Frequency (Hz | minHz maxHz)",
                tooltip=(
                    "sort by phase at maximum-power frequency\nin the data within the range [minHz,maxHz]\n"
                    "(overrides frequency specified in 'coher' flag)"
                ),
            ),
            ControlSpec(
                "text",
                "Percent low-amp. trials to ignore",
                tooltip=(
                    "percent of trials to reject for low amplitude. Else,\n"
                    "if prct is in the range [-100,0] -> percent to reject for high amplitude"
                ),
            ),
            ControlSpec("text", "Window center (ms)", tooltip="Center time of the n-cycle window"),
            ControlSpec("text", "Wavelet cycles", tooltip="cycles per wavelet window"),
            ControlSpec("spacer"),
            ControlSpec("edit", tag="phase", value="", callback=_mirror("phase", "coher")),
            ControlSpec("edit", tag="phase2", value=""),
            ControlSpec("edit", tag="phase3", value=""),
            ControlSpec("text", "        3"),
            ControlSpec("spacer"),
            ControlSpec("spacer"),
            ControlSpec("text", "Inter-trial coherence options", font_weight="bold"),
            ControlSpec(
                "text",
                "Frequency (Hz | minHz maxHz)",
                tooltip=(
                    "[freq] -> plot erp plus amp & coher at freq (Hz)\n[minHz maxHz] -> find max in frequency range\n"
                    "(or at phase freq above, if specified)"
                ),
            ),
            ControlSpec(
                "text",
                "Signif. level (<0.20)",
                tooltip="add coher. signif. level line at alpha (alpha range: (0,0.1])",
            ),
            ControlSpec("text", "Amplitude limits (dB)"),
            ControlSpec("text", "Coher limits (<=1)"),
            ControlSpec(
                "checkbox",
                "Image amps",
                tag="plotamps",
                value=False,
                tooltip="Image the spectral amplitude at the coherence frequency instead of the potential",
            ),
            ControlSpec("edit", tag="coher", value="", callback=_mirror("coher", "phase")),
            ControlSpec("edit", tag="coher2", value=""),
            ControlSpec("edit", tag="limamp", value=""),
            ControlSpec("edit", tag="limcoher", value=""),
            ControlSpec("text", "   (Requires signif.)"),
            ControlSpec("spacer"),
            ControlSpec("text", "Other options", font_weight="bold"),
            ControlSpec(
                "text",
                "Plot spectrum (minHz maxHz)",
                tooltip="Plot the data spectrum between these frequencies at upper right",
            ),
            ControlSpec("text", "Baseline ampl. (dB)", tooltip="Use it to fix baseline amplitude"),
            ControlSpec("text", "Mark times (ms)", tooltip="Draw dashed vertical lines at these latencies"),
            ControlSpec("text", "More options (see >> help erpimage)"),
            ControlSpec("edit", tag="spec", value=""),
            ControlSpec("edit", tag="limbaseamp", value=""),
            ControlSpec("edit", tag="vert", value=""),
            ControlSpec("edit", tag="others", value=""),
        ]
    )
    geometry = [
        (1, 1, 0.4, 0.5, 2.1),
        (1, 1, 1, 1, 1),
        (1, 1, 1, 1, 1),
        (1, 1, 1, 1, 1),
        (1,),
        (1,),
        (1, 1, 1, 0.8, 0.8, 1.2),
        (1, 1, 1, 0.8, 0.8, 1.2),
        (1,),
        (1,),
        (1.6, 1.7, 1.2, 1, 0.5),
        (1.6, 1.7, 1.2, 1, 0.5),
        (1,),
        (1,),
        (1.5, 1, 1, 1, 1),
        (1.5, 1, 1, 1, 1),
        (1,),
        (1,),
        (1.5, 1, 1, 2.2),
        (1.5, 1, 1, 2.2),
    ]
    if not is_channel:
        geometry.insert(0, (1, 1, 0.1, 0.8, 2.1))
    return DialogSpec(
        title=f"{'Channel' if is_channel else 'Component'} ERP image -- pop_erpimage()",
        controls=tuple(controls),
        geometry=tuple(geometry),
        function_name="pop_erpimage",
        eeglab_source="functions/popfunc/pop_erpimage.m",
        help_text="pophelp('pop_erpimage')",
        size=(1113, 831),
        row_spacing=4,
        known_differences=(
            "The component dialog's projection-channel picker is enabled (EEGLAB disables it).",
            "The event-type reminder names the epoch-sorting field (EEGLAB's text says 'event type').",
            "More options is filled from the call or the restored command (EEGLAB leaves it empty).",
        ),
        extra_stylesheet="""
            QDialog#pop_erpimage QLabel,
            QDialog#pop_erpimage QCheckBox,
            QDialog#pop_erpimage QLineEdit,
            QDialog#pop_erpimage QPushButton {
                font-size: 11px;
            }
            QDialog#pop_erpimage QLineEdit,
            QDialog#pop_erpimage QPushButton {
                min-height: 15px;
                max-height: 15px;
            }
            QDialog#pop_erpimage QCheckBox::indicator {
                width: 11px;
                height: 11px;
            }
        """,
    )


def _unique_types(types: Any) -> list[str]:
    """Event types sorted like EEGLAB's unique(): numerically for numeric types, else as text."""
    values = list(types)
    numeric = all(isinstance(value, (int, float, np.number)) and not isinstance(value, bool) for value in values)
    ordered = sorted(set(values), key=float) if numeric else sorted({str(value) for value in values})
    return [event_type_text(value) for value in ordered]


def _channel_button(tag: str, target: str, labels: list[str], *, enabled: bool) -> ControlSpec:
    """EEGLAB's commandchan: a single-channel list opened without a preselection."""
    return ControlSpec(
        "pushbutton",
        "...",
        tag=tag,
        enabled=enabled,
        callback=CallbackSpec(
            "select_channels",
            params={
                "button": tag,
                "target": target,
                "channels": labels,
                "selectionmode": "single",
                "preselect": False,
            },
            matlab_callback="pop_chansel({tmpchanlocs.labels}, 'withindex', 'on', 'selectionmode', 'single')",
        ),
    )


def _mirror(source: str, target: str) -> CallbackSpec:
    """EEGLAB's commandphase and commandcoher: copy the frequency when both boxes are filled."""
    return CallbackSpec("mirror_text", params={"source": source, "target": target})


def _epoch_limits_text(EEG: dict[str, Any]) -> str:
    return " ".join(f"{value:.10g}" for value in 1000 * np.array([float(EEG["xmin"]), float(EEG["xmax"])]))


def _run_gui(
    EEG: dict[str, Any],
    typeplot: int,
    arguments: dict[str, Any],
    options: dict[str, Any],
    *,
    plotmap: bool,
    renderer: Any | None,
) -> tuple[dict[str, Any], dict[str, Any], bool] | None:
    """Arguments, options and scalp-map flag after the dialog; options it cannot show are kept as given."""
    values, hidden = _dialog_values(arguments, options, plotmap=plotmap)
    result = inputgui(pop_erpimage_dialog_spec(EEG, typeplot=typeplot), initial_values=values, renderer=renderer)
    if result is None:
        return None
    call = _dialog_call(EEG, typeplot, result)
    return call["arguments"], hidden | call["options"], call["plotmap"]


def _dialog_values(
    arguments: dict[str, Any], options: dict[str, Any], *, plotmap: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Dialog values showing a call (fields as ``pop_erpimage.m`` restores them), and the options no field shows."""
    values: dict[str, Any] = {"plotmap": plotmap}
    items = {
        "chan": _items(arguments["channel"]),
        "projchan": _items(arguments["projchan"]),
        "type": _event_types(arguments["sortingtype"]),
    }
    values.update({tag: " ".join(_token_text(item) for item in found) for tag, found in items.items() if found})
    for tag, name in (("title", "titleplot"), ("field", "sortingeventfield")):
        if arguments[name]:
            values[tag] = str(arguments[name])
    for tag, name in (("smooth", "smooth"), ("decimate", "decimate"), ("eventrange", "sortingwin")):
        if not is_empty_value(arguments[name]):
            numbers = _number_array(arguments[name])
            values[tag] = str(arguments[name]) if numbers is None else _numbers_text(numbers)
    hidden: dict[str, Any] = {}
    others = []
    for name, value in options.items():
        fields = _option_fields(name, value)
        if fields is not None:
            values.update(fields)
            continue
        text = None if name in DIALOG_SET_OPTIONS else _option_text(name, value)
        if text is None:
            hidden[name] = value
        else:
            others.append(text)
    if others:
        values["others"] = ", ".join(others)
    return values, hidden


def _token_text(item: Any) -> str:
    """A channel, component or event type as dialog text; quoted when it holds spaces, commas or quotes."""
    text = event_type_text(item)
    if text and not any(char in text for char in " ,'\""):
        return text
    quote = '"' if "'" in text else "'"
    return f"{quote}{text}{quote}"


def _option_fields(name: str, value: Any) -> dict[str, Any] | None:
    """The dialog fields that show an erpimage option, or None when none can."""
    if name in CHECKBOX_OPTIONS:
        if isinstance(value, (bool, np.bool_)) or (isinstance(value, str) and value.lower() in {"on", "off"}):
            return {name: is_on(value)}
        return None
    if name == "renorm":
        return {"renorm": value} if isinstance(value, str) else None
    numbers = _number_array(value)
    if numbers is None:
        return None
    if name in ("caxis", "spec", "vert", "align"):
        return {name: _numbers_text(numbers)}
    if name == "phasesort" and numbers.size in (3, 4):
        return {
            "phase3": _numbers_text(numbers[:1]),
            "phase2": _numbers_text(numbers[1:2]),
            "phase": _numbers_text(numbers[2:]),
        }
    if name == "coher" and numbers.size <= 3:
        fields = {"coher": _numbers_text(numbers[:2])}
        if numbers.size == 3:
            fields["coher2"] = _numbers_text(numbers[2:])
        return fields
    if name == "limits" and numbers.size <= 9:
        return _limits_fields(numbers)
    return None


def _limits_fields(limits: np.ndarray) -> dict[str, Any] | None:
    """EEGLAB's limits(1:9) in the dialog; NaN pairs stay blank, a half-NaN pair cannot be shown."""
    padded = np.full(9, np.nan)
    padded[: limits.size] = limits
    fields = {}
    for tag, part in (
        ("limtime", padded[0:2]),
        ("limerp", padded[2:4]),
        ("limamp", padded[4:6]),
        ("limcoher", padded[6:8]),
        ("limbaseamp", padded[8:9]),
    ):
        if np.all(np.isnan(part)):
            continue
        if np.any(np.isnan(part)):
            return None
        fields[tag] = _numbers_text(part)
    return fields


def _option_text(name: str, value: Any) -> str | None:
    """``name=value`` for the More options box, when reading it back gives the same value."""
    text = f"{name}={python_literal(value)}"
    try:
        shown = parse_plot_options_text(text).get(name)
    except ValueError:
        return None
    return text if python_literal(shown) == python_literal(value) else None


def _number_array(value: Any) -> np.ndarray | None:
    """A non-empty 1-D vector of numbers, or None."""
    if value is None or isinstance(value, (str, bytes, bool, np.bool_)):
        return None
    try:
        numbers = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return numbers.ravel() if numbers.ndim <= 1 and numbers.size else None


def _numbers_text(numbers: np.ndarray) -> str:
    return " ".join(str(int(number)) if number.is_integer() else repr(number) for number in map(float, numbers))


def _previous_call(lastcom: str) -> tuple[dict[str, Any], tuple[Any, ...], dict[str, Any]]:
    """Arguments, EEGLAB tokens and keyword options of an earlier ``pop_erpimage`` command."""
    try:
        tree = ast.parse(lastcom.strip())
    except SyntaxError as error:
        raise ValueError(f"pop_erpimage(): lastcom is not a Python command: {lastcom!r}") from error
    call = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and ast.unparse(node.func).split(".")[-1] == "pop_erpimage"
        ),
        None,
    )
    if call is None:
        raise ValueError(f"pop_erpimage(): lastcom is not a pop_erpimage command: {lastcom!r}")
    positional = [None, *(_restored_value(node, f"argument {index}") for index, node in enumerate(call.args[1:], 2))]
    keywords = {}
    for keyword in call.keywords:
        value = _restored_value(keyword.value, str(keyword.arg))
        if keyword.arg is not None and value is not None:
            keywords[keyword.arg] = value
    bound = inspect.signature(pop_erpimage).bind_partial(*positional, **keywords)
    bound.apply_defaults()
    values = bound.arguments
    return {name: values[name] for name in ARGUMENT_NAMES}, tuple(values["options"]), dict(values["kwargs"])


def _restored_value(node: ast.expr, name: str) -> Any:
    """A literal of a history command, or None (logged) when it is an expression such as ``EEG['chanlocs']``."""
    try:
        return _literal(node)
    except (TypeError, ValueError):
        logger.info("pop_erpimage(): %s of the previous command is not a literal, so it is not restored.", name)
        return None


def _literal(node: ast.expr) -> Any:
    """``ast.literal_eval`` that also reads ``float('nan')`` and ``float('inf')``, as history commands write them."""
    if isinstance(node, ast.Call) and ast.unparse(node.func) == "float" and len(node.args) == 1:
        return float(_literal(node.args[0]))
    if isinstance(node, (ast.List, ast.Tuple)):
        items = [_literal(item) for item in node.elts]
        return items if isinstance(node, ast.List) else tuple(items)
    return ast.literal_eval(node)


def _dialog_call(EEG: dict[str, Any], typeplot: int, result: dict[str, Any]) -> dict[str, Any]:
    """``pop_erpimage`` arguments from the dialog values, mapped as ``pop_erpimage.m`` does.

    As in EEGLAB, More options are read first and every field the dialog sets
    replaces them; an empty field or unchecked box sets nothing, except that
    the ERP trace and colorbar, on by default here, are turned off.
    """
    options = _more_options(_text(result, "others"))
    if options.pop("yerplabel", None) is not None:
        logger.warning("pop_erpimage(): the dialog sets the ERP axis label, so yerplabel in More options is ignored.")
    for flag in ("erp", "cbar"):
        if bool(result.get(flag, True)):
            options.pop(flag, None)
        else:
            options.setdefault(flag, False)
    limits = _dialog_limits(EEG, result)
    if limits is not None:
        options["limits"] = limits
    caxis = _dialog_numbers(result, "caxis", "Color limits")
    if caxis.size:
        options["caxis"] = caxis.tolist()
    field = _text(result, "field").strip("'\"")
    renorm = _text(result, "renorm") or NO_RENORM
    if field and renorm.lower() != NO_RENORM:
        options["renorm"] = renorm
    align = _dialog_numbers(result, "align", "Align", sizes=(1,))
    if align.size:
        options["align"] = float(align[0])
    for flag in ("nosort", "noplot"):
        if bool(result.get(flag, False)):
            options[flag] = True
    phasesort = _dialog_phasesort(result)
    if phasesort:
        options["phasesort"] = phasesort
    if bool(result.get("plotamps", False)):
        options["plotamps"] = True
    coher = _dialog_coher(result)
    if coher:
        options["coher"] = coher
    for tag, label in (("spec", "Plot spectrum"), ("vert", "Mark times")):
        values = _dialog_numbers(result, tag, label)
        if values.size:
            options[tag] = values.tolist()
    window = _dialog_numbers(result, "eventrange", "Event time range", sizes=(2,))
    channel_label = "Channel" if typeplot else "Component(s)"
    arguments = {
        "channel": _dialog_items(result, "chan", channel_label, required=True),
        "projchan": _dialog_items(result, "projchan", "Project to channel #", required=False),
        "titleplot": _text(result, "title"),
        "smooth": _dialog_number(result, "smooth", "Smoothing"),
        "decimate": _dialog_number(result, "decimate", "Downsampling"),
        "sortingtype": parse_text_tokens(_text(result, "type")) or None,
        "sortingwin": window.tolist() if window.size else None,
        "sortingeventfield": field,
    }
    return {"arguments": arguments, "options": options, "plotmap": bool(result.get("plotmap", True))}


def _text(result: dict[str, Any], tag: str) -> str:
    return str(result.get(tag, "") or "").strip()


def _dialog_items(result: dict[str, Any], tag: str, label: str, *, required: bool) -> Any:
    """Typed channels or components: one number or label, or a list of them."""
    items = _items(_text(result, tag))
    if not items:
        if required:
            raise ValueError(f"pop_erpimage(): '{label}' is required")
        return None
    plain = [int(item) if isinstance(item, float) and item.is_integer() else item for item in items]
    return plain[0] if len(plain) == 1 else plain


def _dialog_numbers(
    result: dict[str, Any], tag: str, label: str, *, sizes: tuple[int, ...] | None = None
) -> np.ndarray:
    text = _text(result, tag)
    try:
        values = numeric_vector(text.strip("{}"))
    except ValueError as error:
        raise ValueError(f"pop_erpimage(): '{label}' must be numbers, got {text!r}") from error
    if sizes is not None and values.size and values.size not in sizes:
        expected = " or ".join(str(size) for size in sizes)
        raise ValueError(f"pop_erpimage(): '{label}' takes {expected} value(s), got {text!r}")
    return values


def _dialog_number(result: dict[str, Any], tag: str, label: str) -> float | int:
    values = _dialog_numbers(result, tag, label, sizes=(1,))
    if not values.size:
        raise ValueError(f"pop_erpimage(): '{label}' is empty")
    value = float(values[0])
    return int(value) if value.is_integer() else value


def _dialog_limits(EEG: dict[str, Any], result: dict[str, Any]) -> list[float] | None:
    """EEGLAB's limits(1:9); the time limits only when they were edited."""
    limits = [np.nan] * 8
    erp = _dialog_numbers(result, "limerp", "ERP limits", sizes=(2,))
    if erp.size:
        limits[2:4] = erp.tolist()
    time_text = _text(result, "limtime")
    if time_text and time_text != _epoch_limits_text(EEG):
        limits[0:2] = _dialog_numbers(result, "limtime", "Time limits (ms)", sizes=(2,)).tolist()
    amplitude = _dialog_numbers(result, "limamp", "Amplitude limits (dB)", sizes=(2,))
    if amplitude.size:
        limits[4:6] = amplitude.tolist()
    coherence = _dialog_numbers(result, "limcoher", "Coher limits", sizes=(2,))
    if coherence.size:
        limits[6:8] = coherence.tolist()
    baseline = _dialog_numbers(result, "limbaseamp", "Baseline ampl. (dB)", sizes=(1,))
    if baseline.size:
        limits.append(float(baseline[0]))
    return None if np.all(np.isnan(limits)) else limits


def _dialog_phasesort(result: dict[str, Any]) -> list[float]:
    """[center pct freq(s)] from the phase row, growing like a MATLAB array."""
    values: list[float] = []
    frequencies = _dialog_numbers(result, "phase", "Sort by phase frequency")
    if frequencies.size:
        values = [0.0, 0.0, *frequencies.tolist()]
    for tag, label, position in (("phase2", "Percent low-amp. trials", 1), ("phase3", "Window center", 0)):
        value = _dialog_numbers(result, tag, label, sizes=(1,))
        if value.size:
            values = _set_position(values, position, float(value[0]))
    return values


def _dialog_coher(result: dict[str, Any]) -> list[float]:
    """[freq(s) alpha]; a single frequency is repeated when a significance level is given."""
    values = _dialog_numbers(result, "coher", "Coherence frequency").tolist()
    alpha = _dialog_numbers(result, "coher2", "Signif. level", sizes=(1,))
    if alpha.size:
        if len(values) == 1:
            values.append(values[0])
        values = _set_position(values, 2, float(alpha[0]))
    return values


def _set_position(values: list[float], position: int, value: float) -> list[float]:
    padded = values + [0.0] * max(0, position + 1 - len(values))
    padded[position] = value
    return padded


def _more_options(text: str) -> dict[str, Any]:
    try:
        return parse_plot_options_text(text)
    except ValueError as error:
        raise ValueError(
            f"pop_erpimage(): More options needs 'key', value pairs (for example 'erpstd', 'on'): {error}"
        ) from error


__all__ = ["pop_erpimage", "pop_erpimage_dialog_spec"]
