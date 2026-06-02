"""EEGLAB-style scrolling EEG browser data model and public ``eegplot`` API."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.popfunc._plot_utils import component_activations


DEFAULT_SRATE = 256.0
DEFAULT_WINLENGTH = 5.0
DEFAULT_WINREJ_COLOR = (0.7, 1.0, 0.9)
DEFAULT_TRACE_COLORS = ((0.0, 0.0, 0.4),)
COLOR_ON_TRACE_COLORS = ((0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.5, 0.0))
DECIMATION_SAMPLES_PER_PIXEL = 2

_OPTION_NAMES = {
    "srate",
    "spacing",
    "limits",
    "winlength",
    "time",
    "dispchans",
    "title",
    "plottitle",
    "xgrid",
    "ygrid",
    "data2",
    "command",
    "command_callback",
    "butlabel",
    "winrej",
    "wincolor",
    "events",
    "ploteventdur",
    "submean",
    "eloc_file",
    "scale",
    "color",
    "freqs",
    "freqlimits",
    "component",
    "setelectrode",
    "children",
    "noui",
    "show",
}


@dataclass(frozen=True)
class BrowserEvent:
    """Event marker normalized for browser rendering."""

    type: str
    latency: float
    duration: float | None = None
    color_index: int = 0


@dataclass(frozen=True)
class WinRejRegion:
    """EEGLAB ``eegplot`` rejection-window row."""

    start: float
    end: float
    color: tuple[float, float, float]
    channel_mask: tuple[bool, ...]


@dataclass(frozen=True)
class BrowserData:
    """Channel-major browser data normalized from arrays or EEG dictionaries."""

    data: np.ndarray
    flat_data: np.ndarray
    pnts: int
    trials: int
    channel_labels: tuple[str, ...]
    mode: str
    data2: np.ndarray | None = None
    flat_data2: np.ndarray | None = None
    axis_srate: float | None = None
    axis_limits: tuple[float, float] | None = None
    x_values: np.ndarray | None = None

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[0])

    @property
    def total_samples(self) -> int:
        return int(self.flat_data.shape[1])

    @property
    def epoched(self) -> bool:
        return self.trials > 1


@dataclass
class BrowserState:
    """Mutable browser display state."""

    srate: float
    spacing: float
    limits: tuple[float, float]
    winlength: float
    time: float
    dispchans: int
    title: str
    plottitle: str
    xgrid: bool
    ygrid: bool
    submean: bool
    scale: bool
    colors: tuple[Any, ...]
    winrej: list[WinRejRegion]
    events: list[BrowserEvent]
    channel_offset: int = 0
    show_events: bool = True
    show_event_durations: bool = False
    show_marks: bool = True
    channel_label_mode: str = "labels"
    zoom_enabled: bool = False
    stacked: bool = False
    normalized: bool = False
    accept_label: str | None = None
    mark_color: tuple[float, float, float] = DEFAULT_WINREJ_COLOR
    setelectrode: bool = False
    noui: bool = False
    accepted: bool = False
    cancelled: bool = False

    def clamp_to_data(self, browser_data: BrowserData) -> None:
        """Clamp visible time and channel offsets to the available data."""
        self.dispchans = max(1, min(int(self.dispchans), browser_data.n_channels))
        self.channel_offset = max(0, min(int(self.channel_offset), browser_data.n_channels - self.dispchans))
        max_time = max(0.0, browser_window_duration(browser_data, self) - float(self.winlength))
        self.time = max(0.0, min(float(self.time), max_time))


@dataclass(frozen=True)
class BrowserModel:
    """Fully normalized data plus display state for the Qt browser."""

    data: BrowserData
    state: BrowserState


@dataclass(frozen=True)
class _FrequencySelection:
    data: np.ndarray
    sample_slice: slice
    axis_srate: float | None
    axis_limits: tuple[float, float] | None
    freq_values: np.ndarray | None


def eegplot(data: Any, *args: Any, **kwargs: Any) -> Any:
    """Open an EEGLAB-style scrolling browser for channel-major EEG data.

    Parameters use EEGLAB names where practical. ``data`` may be a NumPy-like
    array shaped ``channels x samples`` or ``channels x samples x trials``, or
    an EEG dictionary containing ``data``, ``srate``, ``chanlocs``, and
    ``event`` fields. Spectral inputs use ``freqs`` with ``freqlimits``; in
    that mode ``winlength`` is interpreted as a frequency span in Hz. The input
    is copied into the browser model and is not mutated by the viewer.
    """
    options = parse_eegplot_options(args, kwargs)
    model = build_eegplot_model(data, **options)
    if not bool(options.get("show", True)):
        return model
    from eegprep.functions.guifunc.eegbrowser import open_eegbrowser

    command_callback = options.get("command_callback")
    if command_callback is None and callable(options.get("command")):
        command_callback = options.get("command")
    window = open_eegbrowser(model, accept_callback=command_callback)
    children = options.get("children")
    if children:
        from eegprep.functions.guifunc.eegbrowser import link_eegbrowser_windows

        link_eegbrowser_windows(window, children)
    return window


def parse_eegplot_options(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Normalize Python keyword and EEGLAB-style key/value options."""
    if len(args) % 2:
        raise TypeError("eegplot options must be key/value pairs")
    options = dict(kwargs)
    for index in range(0, len(args), 2):
        key = args[index]
        if not isinstance(key, str):
            raise TypeError("eegplot option names must be strings")
        if key in options:
            raise TypeError(f"eegplot option {key!r} was supplied twice")
        options[key] = args[index + 1]
    unknown = sorted(set(options) - _OPTION_NAMES)
    if unknown:
        raise ValueError(f"eegplot: unrecognized option: {unknown[0]!r}")
    return options


def build_eegplot_model(data: Any, **kwargs: Any) -> BrowserModel:
    """Build a browser model without opening a Qt window."""
    source_eeg = data if isinstance(data, dict) else None
    time_supplied = "time" in kwargs and kwargs["time"] is not None
    if source_eeg is not None and bool(kwargs.get("component", False)):
        kwargs = dict(kwargs)
        kwargs["_component_data"] = component_activations(source_eeg)
    options = _model_options(source_eeg, kwargs)
    browser_data = normalize_browser_data(data, options)
    if options["dispchans"] is None:
        options["dispchans"] = browser_data.n_channels
    if browser_data.epoched:
        options["time"] = max(0.0, float(options["time"]) - 1.0) if time_supplied else 0.0
    elif options["time"] is None:
        options["time"] = 0.0
    state = BrowserState(
        srate=float(browser_data.axis_srate if browser_data.axis_srate is not None else options["srate"]),
        spacing=float(options["spacing"]),
        limits=tuple(float(value) for value in (browser_data.axis_limits or options["limits"])),
        winlength=float(options["winlength"]),
        time=float(options["time"]),
        dispchans=int(options["dispchans"]),
        title=str(options["title"]),
        plottitle=str(options["plottitle"]),
        xgrid=_on_off(options["xgrid"], "xgrid"),
        ygrid=_on_off(options["ygrid"], "ygrid"),
        submean=_on_off(options["submean"], "submean"),
        scale=_on_off(options["scale"], "scale"),
        colors=normalize_trace_colors(options["color"]),
        winrej=normalize_winrej(options["winrej"], browser_data.n_channels, browser_data.total_samples),
        events=normalize_events(options["events"]),
        show_event_durations=bool(options["ploteventdur"]),
        accept_label=(
            str(options["butlabel"])
            if not _is_empty(options["command"]) or options.get("command_callback") is not None
            else None
        ),
        mark_color=_normalize_rgb(options["wincolor"], "wincolor"),
        setelectrode=bool(options["setelectrode"]),
        noui=_on_off(options["noui"], "noui"),
    )
    state.clamp_to_data(browser_data)
    return BrowserModel(browser_data, state)


def normalize_browser_data(data: Any, options: dict[str, Any]) -> BrowserData:
    """Normalize array, EEG, component, spectral, and overlay inputs."""
    source_eeg = data if isinstance(data, dict) else None
    component = bool(options.get("component", False))
    if source_eeg is not None:
        raw_data = (
            options["_component_data"]
            if component and "_component_data" in options
            else component_activations(source_eeg)
            if component
            else source_eeg.get("data")
        )
    else:
        raw_data = data
    array = _as_channel_data(raw_data)
    frequency_selection = _select_frequency_range(array, options)
    array = frequency_selection.data
    flat = flatten_browser_data(array)
    labels = _channel_labels(source_eeg, array.shape[0], options["eloc_file"], component=component)
    mode = _browser_mode(array, component=component, freqs=options.get("freqs"))
    x_values = None
    if frequency_selection.freq_values is not None:
        x_values = np.tile(frequency_selection.freq_values, int(array.shape[2]))
    data2 = options.get("data2")
    flat_data2 = None
    data2_array = None
    if data2 is not None and not _is_empty(data2):
        data2_array = _as_channel_data(data2)
        data2_array = data2_array[:, frequency_selection.sample_slice, :]
        if data2_array.shape != array.shape:
            raise ValueError("data2 must have the same normalized shape as data")
        data2_array = np.array(data2_array, dtype=float, copy=True)
        flat_data2 = flatten_browser_data(data2_array)
    return BrowserData(
        data=np.array(array, dtype=float, copy=True),
        flat_data=flat,
        pnts=int(array.shape[1]),
        trials=int(array.shape[2]),
        channel_labels=labels,
        mode=mode,
        data2=data2_array,
        flat_data2=flat_data2,
        axis_srate=frequency_selection.axis_srate,
        axis_limits=frequency_selection.axis_limits,
        x_values=x_values,
    )


def flatten_browser_data(data: np.ndarray) -> np.ndarray:
    """Flatten ``channels x points x trials`` data into EEGLAB browser order."""
    if data.ndim != 3:
        raise ValueError("browser data must be 3-D before flattening")
    return np.array(data.transpose(0, 2, 1).reshape(data.shape[0], data.shape[1] * data.shape[2]), copy=True)


def browser_window_duration(browser_data: BrowserData, state: BrowserState) -> float:
    """Return total browser x-axis length in seconds or epochs."""
    if browser_data.epoched:
        return float(browser_data.trials)
    return float(browser_data.total_samples) / float(state.srate)


def visible_sample_bounds(browser_data: BrowserData, state: BrowserState) -> tuple[int, int]:
    """Return 0-based half-open sample bounds for the visible window."""
    multiplier = browser_data.pnts if browser_data.epoched else state.srate
    start = int(round(float(state.time) * float(multiplier)))
    stop = int(round((float(state.time) + float(state.winlength)) * float(multiplier)))
    start = max(0, min(start, browser_data.total_samples - 1))
    stop = max(start + 1, min(stop, browser_data.total_samples))
    return start, stop


def time_to_sample(time_value: float, browser_data: BrowserData, state: BrowserState) -> int:
    """Convert a browser time in seconds or epochs to a 0-based sample index."""
    multiplier = browser_data.pnts if browser_data.epoched else state.srate
    return max(0, min(int(round(float(time_value) * float(multiplier))), browser_data.total_samples - 1))


def event_latency_to_sample(latency: float, browser_data: BrowserData) -> int:
    """Convert an EEGLAB 1-based event latency to a 0-based browser sample."""
    return max(0, min(int(round(float(latency))) - 1, browser_data.total_samples - 1))


def decimate_minmax(x_values: np.ndarray, y_values: np.ndarray, pixel_width: int) -> tuple[np.ndarray, np.ndarray]:
    """Min/max decimate a trace while preserving extrema and endpoints."""
    x = np.asarray(x_values, dtype=float).ravel()
    y = np.asarray(y_values, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError("x_values and y_values must have matching lengths")
    if x.size == 0:
        return x, y
    max_points = max(1, int(pixel_width)) * DECIMATION_SAMPLES_PER_PIXEL
    if x.size <= max_points:
        return x, y
    edges = np.linspace(0, x.size, max(1, int(pixel_width)) + 1, dtype=int)
    keep: set[int] = {0, x.size - 1}
    for start, stop in zip(edges[:-1], edges[1:]):
        if stop <= start:
            continue
        segment = y[start:stop]
        finite_indices = np.flatnonzero(np.isfinite(segment))
        if finite_indices.size == 0:
            keep.add(start)
            keep.add(stop - 1)
            continue
        finite_values = segment[finite_indices]
        keep.add(start + int(finite_indices[int(np.argmin(finite_values))]))
        keep.add(start + int(finite_indices[int(np.argmax(finite_values))]))
    indices = np.fromiter(sorted(keep), dtype=int)
    return x[indices], y[indices]


def normalize_winrej(value: Any, n_channels: int, total_samples: int) -> list[WinRejRegion]:
    """Normalize EEGLAB ``winrej`` rows."""
    if value is None or _is_empty(value):
        return []
    rows = np.asarray(value, dtype=float)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.ndim != 2 or rows.shape[1] < 2:
        raise ValueError("winrej must be a 2-D array with at least start and end columns")
    regions = []
    for row in rows:
        start = float(row[0])
        end = float(row[1])
        if end < start:
            start, end = end, start
        if start < 0 or end > total_samples:
            raise ValueError("winrej start/end values must fall within the browser sample range")
        color = tuple(float(item) for item in (row[2:5] if row.size >= 5 else DEFAULT_WINREJ_COLOR))
        if len(color) != 3:
            raise ValueError("winrej color columns must contain RGB values")
        if row.size >= 5 + n_channels:
            mask = tuple(bool(item) for item in row[5 : 5 + n_channels])
        else:
            mask = (True,) * n_channels
        regions.append(WinRejRegion(start=start, end=end, color=color, channel_mask=mask))
    return regions


def winrej_to_array(regions: list[WinRejRegion], n_channels: int | None = None) -> np.ndarray:
    """Convert normalized ``winrej`` regions back to EEGLAB row format."""
    if n_channels is None:
        n_channels = max((len(region.channel_mask) for region in regions), default=0)
    rows = np.zeros((len(regions), 5 + int(n_channels)), dtype=float)
    for index, region in enumerate(regions):
        rows[index, 0] = float(region.start)
        rows[index, 1] = float(region.end)
        rows[index, 2:5] = np.asarray(region.color, dtype=float)
        mask = np.asarray(region.channel_mask, dtype=bool)
        rows[index, 5 : 5 + min(mask.size, int(n_channels))] = mask[: int(n_channels)]
    return rows


def trial2eegplot(rej: Any, rejE: Any, pnts: int, color: Any = DEFAULT_WINREJ_COLOR) -> np.ndarray:
    """Convert EEGLAB trial/electrode marks to ``eegplot`` ``winrej`` rows.

    EEGLAB's ``trial2eegplot.m`` stores epoch windows as 0-based browser
    offsets: ``[(trial - 1) * pnts, trial * pnts - 1]``.
    """
    trial_marks = np.asarray(rej if rej is not None else [], dtype=bool).ravel()
    row_marks = np.asarray(rejE if rejE is not None else [], dtype=bool)
    if trial_marks.size == 0:
        return np.zeros((0, 5 + (0 if row_marks.ndim == 0 else row_marks.shape[0])), dtype=float)
    if row_marks.size == 0:
        row_marks = np.ones((0, trial_marks.size), dtype=bool)
    if row_marks.ndim == 1:
        row_marks = row_marks.reshape(1, -1)
    if row_marks.shape[1] < trial_marks.size:
        padded = np.zeros((row_marks.shape[0], trial_marks.size), dtype=bool)
        padded[:, : row_marks.shape[1]] = row_marks
        row_marks = padded
    marked = np.flatnonzero(trial_marks)
    rows = np.zeros((marked.size, 5 + row_marks.shape[0]), dtype=float)
    if marked.size == 0:
        return rows
    rows[:, 0] = marked * int(pnts)
    rows[:, 1] = (marked + 1) * int(pnts) - 1
    rows[:, 2:5] = np.tile(np.asarray(_normalize_rgb(color, "color"), dtype=float), (marked.size, 1))
    rows[:, 5:] = row_marks[:, marked].T
    return rows


def eegplot2event(
    eegplotrej: Any,
    event_type: float = -1,
    color: Any = None,
    colorout: Any = None,
) -> np.ndarray:
    """Convert continuous ``eegplot`` marks to EEGLAB ``eeg_eegrej`` rows."""
    rows = _filtered_winrej_rows(eegplotrej, color=color, colorout=colorout, round_colors=False)
    if rows.size == 0:
        return np.zeros((0, 7), dtype=float)
    events = np.zeros((rows.shape[0], 7), dtype=float)
    events[:, 0] = float(event_type)
    events[:, 1] = 1.0
    events[:, 2:4] = np.rint(rows[:, 0:2])
    events[:, 4:7] = rows[:, 2:5]
    return events


def eegplot2trial(
    eegplotrej: Any,
    pnts: int,
    sweeps: int,
    color: Any = None,
    colorout: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert ``eegplot`` marks to EEGLAB trial and electrode rejection arrays."""
    rows = _filtered_winrej_rows(eegplotrej, color=color, colorout=colorout, round_colors=True)
    trialrej = np.zeros(int(sweeps), dtype=bool)
    nbchan = max(0, rows.shape[1] - 5) if rows.ndim == 2 else 0
    elecrej = np.zeros((nbchan, int(sweeps)), dtype=bool)
    if rows.size == 0:
        return trialrej, elecrej
    rows = rows[np.argsort(rows[:, 0])]
    rows[rows[:, 0] == 1, 0] = 0
    starts = rows[:, 0] / float(pnts) + 1.0
    aligned = np.isclose(starts, np.rint(starts))
    for row, trial_value in zip(rows[aligned], starts[aligned]):
        trial_index = int(round(float(trial_value))) - 1
        if 0 <= trial_index < int(sweeps):
            trialrej[trial_index] = True
            if nbchan:
                elecrej[:, trial_index] |= np.asarray(row[5 : 5 + nbchan], dtype=bool)
    return trialrej, elecrej


def toggle_winrej_at_sample(
    regions: list[WinRejRegion],
    sample: int | float,
    *,
    n_channels: int,
    channel_index: int | None = None,
) -> list[WinRejRegion]:
    """Remove a mark at ``sample`` or toggle one channel in that mark."""
    sample_value = float(sample)
    out = list(regions)
    for index, region in enumerate(out):
        if not (float(region.start) < sample_value < float(region.end)):
            continue
        if channel_index is None:
            del out[index]
            return out
        mask = _expanded_mask(region.channel_mask, n_channels)
        if 0 <= int(channel_index) < n_channels:
            mask[int(channel_index)] = not mask[int(channel_index)]
        if any(mask):
            out[index] = replace(region, channel_mask=tuple(mask))
        else:
            del out[index]
        return out
    return out


def add_winrej_region(
    regions: list[WinRejRegion],
    start: int | float,
    end: int | float,
    *,
    n_channels: int,
    total_samples: int,
    color: Any = DEFAULT_WINREJ_COLOR,
    channel_index: int | None = None,
    pnts: int | None = None,
) -> list[WinRejRegion]:
    """Add a continuous or epoched ``winrej`` mark with EEGLAB-style merging.

    Continuous browser marks use 1-based sample latencies because accepted
    regions flow to ``eeg_eegrej``. Epoched marks use EEGLAB's 0-based
    ``trial2eegplot`` browser offsets.
    """
    start_value = int(round(float(start)))
    end_value = int(round(float(end)))
    if end_value < start_value:
        start_value, end_value = end_value, start_value
    if start_value == end_value:
        return toggle_winrej_at_sample(regions, start_value, n_channels=n_channels, channel_index=channel_index)
    if pnts is not None:
        return _add_epoched_winrej(
            regions,
            start_value,
            end_value,
            n_channels,
            int(total_samples),
            int(pnts),
            color,
            channel_index,
        )
    start_value = max(1, min(start_value, int(total_samples)))
    end_value = max(1, min(end_value, int(total_samples)))
    mask = _new_mask(n_channels, channel_index)
    new_region = WinRejRegion(float(start_value), float(end_value), _normalize_rgb(color, "color"), tuple(mask))
    return _merge_continuous_regions([*regions, new_region], n_channels)


def normalize_events(events: Any) -> list[BrowserEvent]:
    """Normalize EEGLAB event structures for rendering."""
    if events is None or _is_empty(events):
        return []
    event_items = _event_items(events)
    event_rows: list[tuple[str, float, float | None]] = []
    for event in event_items:
        if "latency" not in event:
            continue
        label = str(_scalar(event.get("type", "")))
        duration = event.get("duration")
        event_rows.append(
            (
                label,
                float(_scalar(event["latency"])),
                None if duration is None or _is_empty(duration) else float(_scalar(duration)),
            )
        )
    labels = sorted({label for label, _latency, _duration in event_rows})
    label_indices = {label: index for index, label in enumerate(labels)}
    return [
        BrowserEvent(type=label, latency=latency, duration=duration, color_index=label_indices[label])
        for label, latency, duration in event_rows
    ]


def normalize_trace_colors(value: Any) -> tuple[Any, ...]:
    """Normalize EEGLAB ``color`` option values."""
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "on":
            return COLOR_ON_TRACE_COLORS
        if lowered == "off":
            return DEFAULT_TRACE_COLORS
        raise ValueError("color must be 'on', 'off', or a sequence of colors")
    if value is None:
        return DEFAULT_TRACE_COLORS
    colors = tuple(value)
    if not colors:
        return DEFAULT_TRACE_COLORS
    return colors


def _model_options(source_eeg: dict[str, Any] | None, kwargs: dict[str, Any]) -> dict[str, Any]:
    options = dict(kwargs)
    data = source_eeg.get("data") if source_eeg is not None else None
    shape = np.asarray(data).shape if data is not None else ()
    pnts = int(source_eeg.get("pnts", shape[1] if len(shape) > 1 else 0) or 0) if source_eeg is not None else 0
    srate = float(
        options.get("srate", source_eeg.get("srate", DEFAULT_SRATE) if source_eeg is not None else DEFAULT_SRATE)
    )
    options["srate"] = srate
    options.setdefault("spacing", None)
    options.setdefault("limits", _default_limits(source_eeg, pnts, srate))
    options.setdefault("winlength", DEFAULT_WINLENGTH)
    options.setdefault("time", None)
    options.setdefault("dispchans", None)
    options.setdefault("title", _default_title(source_eeg))
    options.setdefault("plottitle", "")
    options.setdefault("xgrid", "off")
    options.setdefault("ygrid", "off")
    options.setdefault("data2", None)
    options.setdefault("command", None)
    options.setdefault("command_callback", None)
    options.setdefault("butlabel", "REJECT")
    options.setdefault("winrej", None)
    options.setdefault("wincolor", DEFAULT_WINREJ_COLOR)
    options.setdefault("events", source_eeg.get("event", []) if source_eeg is not None else [])
    options.setdefault("ploteventdur", False)
    options.setdefault("submean", "off")
    options.setdefault("eloc_file", source_eeg.get("chanlocs", None) if source_eeg is not None else None)
    options.setdefault("scale", "on")
    options.setdefault("color", "off")
    options.setdefault("freqs", None)
    options.setdefault("freqlimits", None)
    options.setdefault("component", False)
    options.setdefault("setelectrode", False)
    options.setdefault("children", ())
    options.setdefault("noui", "off")
    if options["spacing"] is None or _is_empty(options["spacing"]) or float(options["spacing"]) == 0:
        options["spacing"] = _default_spacing(source_eeg, options)
    if options["time"] is None:
        trials = int(source_eeg.get("trials", 1) or 1) if source_eeg is not None else 1
        options["time"] = 1.0 if trials > 1 else 0.0
    if options["dispchans"] is None:
        if (
            bool(options.get("component", False))
            and source_eeg is not None
            and np.asarray(source_eeg.get("icaweights", [])).ndim == 2
        ):
            options["dispchans"] = int(np.asarray(source_eeg.get("icaweights")).shape[0])
        elif source_eeg is not None:
            options["dispchans"] = int(source_eeg.get("nbchan", 0) or 0)
    return options


def _as_channel_data(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 2:
        array = array[:, :, np.newaxis]
    if array.ndim != 3:
        raise ValueError("eegplot data must be 2-D or 3-D channel-major data")
    if array.shape[0] == 0 or array.shape[1] == 0 or array.shape[2] == 0:
        raise ValueError("eegplot data must contain at least one channel, sample, and trial")
    return array


def _select_frequency_range(array: np.ndarray, options: dict[str, Any]) -> _FrequencySelection:
    freqs = options.get("freqs")
    freqlimits = options.get("freqlimits")
    if freqs is None and freqlimits is None:
        return _FrequencySelection(array, slice(None), None, None, None)
    if freqs is None or freqlimits is None:
        raise ValueError("freqs and freqlimits must be supplied together")
    freq_values = np.asarray(freqs, dtype=float).ravel()
    bounds = np.asarray(freqlimits, dtype=float).ravel()
    if freq_values.size != array.shape[1] or bounds.size != 2:
        raise ValueError("freqs must match data samples and freqlimits must contain [start end]")
    start = int(np.argmin(np.abs(freq_values - bounds[0])))
    end = int(np.argmin(np.abs(freq_values - bounds[1])))
    if end < start:
        start, end = end, start
    selected_freqs = np.array(freq_values[start : end + 1], dtype=float, copy=True)
    axis_span = abs(float(selected_freqs[-1] - selected_freqs[0])) if selected_freqs.size > 1 else 1.0
    axis_srate = float(selected_freqs.size / max(axis_span, np.finfo(float).eps))
    return _FrequencySelection(
        data=array[:, start : end + 1, :],
        sample_slice=slice(start, end + 1),
        axis_srate=axis_srate,
        axis_limits=(float(selected_freqs[0]), float(selected_freqs[-1])),
        freq_values=selected_freqs,
    )


def _default_spacing(source_eeg: dict[str, Any] | None, options: dict[str, Any]) -> float:
    raw = None
    if source_eeg is not None and bool(options.get("component", False)):
        raw = options["_component_data"] if "_component_data" in options else component_activations(source_eeg)
    elif source_eeg is not None:
        raw = source_eeg.get("data")
    if raw is None:
        return 1.0
    array = _as_channel_data(raw)
    flat = flatten_browser_data(array)
    sample_count = min(1000, flat.shape[1])
    stds = np.nanstd(flat[:, :sample_count], axis=1)
    stds = np.sort(stds[np.isfinite(stds)])
    if stds.size > 2:
        spacing = float(np.mean(stds[1:-1]) * 3.0)
    elif stds.size:
        spacing = float(np.mean(stds) * 3.0)
    else:
        spacing = 1.0
    if spacing > 10:
        spacing = float(round(spacing))
    if spacing <= 0 or np.isnan(spacing):
        spacing = 1.0
    return spacing


def _default_limits(source_eeg: dict[str, Any] | None, pnts: int, srate: float) -> tuple[float, float]:
    if source_eeg is not None and ("xmin" in source_eeg or "xmax" in source_eeg):
        return (float(source_eeg.get("xmin", 0.0) or 0.0) * 1000.0, float(source_eeg.get("xmax", 0.0) or 0.0) * 1000.0)
    return (0.0, 1000.0 * float(max(pnts - 1, 0)) / float(srate))


def _default_title(source_eeg: dict[str, Any] | None) -> str:
    setname = str(source_eeg.get("setname", "") if source_eeg is not None else "").strip()
    suffix = f" -- {setname}" if setname else ""
    return f"Scroll activity -- eegplot(){suffix}"


def _channel_labels(
    source_eeg: dict[str, Any] | None,
    n_channels: int,
    eloc_file: Any,
    *,
    component: bool,
) -> tuple[str, ...]:
    if component:
        return tuple(str(index) for index in range(1, n_channels + 1))
    if isinstance(eloc_file, np.ndarray):
        eloc_file = eloc_file.tolist()
    if eloc_file is None or eloc_file == 0:
        chanlocs = chanlocs_as_list(source_eeg.get("chanlocs", [])) if source_eeg is not None else []
    elif _is_empty(eloc_file):
        return tuple("" for _index in range(n_channels))
    elif isinstance(eloc_file, (list, tuple)) and all(isinstance(item, dict) for item in eloc_file):
        chanlocs = chanlocs_as_list(eloc_file)
    elif isinstance(eloc_file, (list, tuple)):
        return tuple(str(int(item)) for item in eloc_file[:n_channels])
    else:
        chanlocs = []
    labels = []
    for index, chanloc in enumerate(chanlocs[:n_channels], start=1):
        label = str(chanloc.get("labels") or "").strip()
        labels.append(label or str(index))
    while len(labels) < n_channels:
        labels.append(str(len(labels) + 1))
    return tuple(labels)


def _browser_mode(array: np.ndarray, *, component: bool, freqs: Any) -> str:
    if freqs is not None:
        return "spectral"
    if component:
        return "component"
    if array.shape[2] > 1:
        return "epoched"
    return "continuous"


def _on_off(value: Any, name: str) -> bool:
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "on":
            return True
        if lowered == "off":
            return False
        raise ValueError(f"{name} must be either 'on' or 'off'")
    return bool(value)


def _normalize_rgb(value: Any, name: str) -> tuple[float, float, float]:
    if value is None or _is_empty(value):
        return DEFAULT_WINREJ_COLOR
    values = tuple(float(item) for item in value)
    if len(values) != 3:
        raise ValueError(f"{name} must contain three RGB values")
    if any(item < 0.0 or item > 1.0 for item in values):
        raise ValueError(f"{name} RGB values must be between 0 and 1")
    return values


def _filtered_winrej_rows(eegplotrej: Any, *, color: Any, colorout: Any, round_colors: bool) -> np.ndarray:
    if eegplotrej is None or _is_empty(eegplotrej):
        return np.zeros((0, 5), dtype=float)
    rows = np.asarray(eegplotrej, dtype=float)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.ndim != 2 or rows.shape[1] < 5:
        raise ValueError("eegplot rejection rows must contain start, end, and RGB columns")
    rows = np.array(rows, dtype=float, copy=True)
    if round_colors:
        rows[:, 2:5] = np.round(rows[:, 2:5] * 100.0) / 100.0
    if color is not None and not _is_empty(color):
        colors = np.asarray(color, dtype=float)
        if colors.ndim == 1:
            colors = colors.reshape(1, -1)
        if round_colors:
            colors = np.round(colors * 100.0) / 100.0
        selected = []
        keys = _color_keys(rows[:, 2:5])
        for row_color in colors[:, :3]:
            selected.extend(np.flatnonzero(keys == _color_key(row_color)).tolist())
        return rows[selected, :] if selected else rows[:0, :]
    if colorout is not None and not _is_empty(colorout):
        colors = np.asarray(colorout, dtype=float)
        if colors.ndim == 1:
            colors = colors.reshape(1, -1)
        if round_colors:
            colors = np.round(colors * 100.0) / 100.0
        keep = np.ones(rows.shape[0], dtype=bool)
        keys = _color_keys(rows[:, 2:5])
        for row_color in colors[:, :3]:
            keep &= keys != _color_key(row_color)
        rows = rows[keep, :]
    return rows


def _color_keys(colors: np.ndarray) -> np.ndarray:
    return colors[:, 0] + 255.0 * colors[:, 1] + 255.0 * 255.0 * colors[:, 2]


def _color_key(color: np.ndarray) -> float:
    values = np.asarray(color, dtype=float).ravel()
    return float(values[0] + 255.0 * values[1] + 255.0 * 255.0 * values[2])


def _new_mask(n_channels: int, channel_index: int | None) -> list[bool]:
    if channel_index is None:
        return [True] * int(n_channels)
    mask = [False] * int(n_channels)
    if 0 <= int(channel_index) < int(n_channels):
        mask[int(channel_index)] = True
    return mask


def _expanded_mask(mask: tuple[bool, ...], n_channels: int) -> list[bool]:
    out = list(mask[: int(n_channels)])
    out.extend([False] * (int(n_channels) - len(out)))
    return out


def _add_epoched_winrej(
    regions: list[WinRejRegion],
    start: int,
    end: int,
    n_channels: int,
    total_samples: int,
    pnts: int,
    color: Any,
    channel_index: int | None,
) -> list[WinRejRegion]:
    first_epoch = max(0, min(start, total_samples - 1)) // pnts
    last_epoch = max(0, min(end, total_samples - 1)) // pnts
    out = list(regions)
    new_mask = tuple(_new_mask(n_channels, channel_index))
    color_value = _normalize_rgb(color, "color")
    for epoch in range(int(first_epoch), int(last_epoch) + 1):
        epoch_start = float(epoch * pnts)
        epoch_end = float((epoch + 1) * pnts - 1)
        match_index = next(
            (
                index
                for index, region in enumerate(out)
                if int(region.start) // pnts == epoch and int(region.start) % pnts == 0
            ),
            None,
        )
        if match_index is None:
            out.append(WinRejRegion(epoch_start, epoch_end, color_value, new_mask))
            continue
        existing = out[match_index]
        existing_mask = tuple(_expanded_mask(existing.channel_mask, n_channels))
        out[match_index] = replace(
            existing,
            start=min(existing.start, epoch_start),
            end=max(existing.end, epoch_end),
            color=color_value,
            channel_mask=tuple(old or new for old, new in zip(existing_mask, new_mask)),
        )
    return sorted(out, key=lambda region: (region.start, region.end))


def _merge_continuous_regions(regions: list[WinRejRegion], n_channels: int) -> list[WinRejRegion]:
    sorted_regions = sorted(regions, key=lambda region: (region.start, region.end))
    merged: list[WinRejRegion] = []
    for region in sorted_regions:
        mask = tuple(_expanded_mask(region.channel_mask, n_channels))
        region = replace(region, channel_mask=mask)
        if not merged or region.start > merged[-1].end:
            merged.append(region)
            continue
        previous = merged[-1]
        merged[-1] = WinRejRegion(
            start=previous.start,
            end=max(previous.end, region.end),
            color=region.color,
            channel_mask=tuple(a or b for a, b in zip(previous.channel_mask, region.channel_mask)),
        )
    return merged


def _event_items(events: Any) -> list[dict[str, Any]]:
    if isinstance(events, dict):
        if "latency" in events:
            latencies = np.asarray(events["latency"]).ravel()
            types = np.asarray(events.get("type", [""] * latencies.size), dtype=object).ravel()
            durations = np.asarray(events.get("duration", [None] * latencies.size), dtype=object).ravel()
            return [
                {
                    "type": types[min(index, types.size - 1)],
                    "latency": latencies[index],
                    "duration": durations[min(index, durations.size - 1)],
                }
                for index in range(latencies.size)
            ]
        return [copy.deepcopy(events)]
    if isinstance(events, np.ndarray):
        events = events.tolist()
    return [copy.deepcopy(event) for event in list(events) if isinstance(event, dict)]


def _scalar(value: Any) -> Any:
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    if array.size == 1:
        return array.ravel()[0].item() if hasattr(array.ravel()[0], "item") else array.ravel()[0]
    return value


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, np.ndarray):
        return value.size == 0
    if isinstance(value, (list, tuple, dict, str)):
        return len(value) == 0
    return False


def copy_model_with_state(model: BrowserModel, **state_updates: Any) -> BrowserModel:
    """Return a model sharing immutable data with a copied display state."""
    return BrowserModel(model.data, replace(model.state, **state_updates))


__all__ = [
    "BrowserData",
    "BrowserEvent",
    "BrowserModel",
    "BrowserState",
    "WinRejRegion",
    "add_winrej_region",
    "build_eegplot_model",
    "browser_window_duration",
    "copy_model_with_state",
    "decimate_minmax",
    "eegplot",
    "eegplot2event",
    "eegplot2trial",
    "event_latency_to_sample",
    "flatten_browser_data",
    "normalize_browser_data",
    "normalize_events",
    "normalize_trace_colors",
    "normalize_winrej",
    "parse_eegplot_options",
    "time_to_sample",
    "toggle_winrej_at_sample",
    "trial2eegplot",
    "visible_sample_bounds",
    "winrej_to_array",
]
