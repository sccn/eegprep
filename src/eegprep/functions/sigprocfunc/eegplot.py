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
    "butlabel",
    "winrej",
    "wincolor",
    "events",
    "submean",
    "eloc_file",
    "scale",
    "color",
    "freqs",
    "freqlimits",
    "component",
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
    show_marks: bool = True
    channel_label_mode: str = "labels"
    zoom_enabled: bool = False
    stacked: bool = False
    normalized: bool = False
    accept_label: str | None = None
    mark_color: tuple[float, float, float] = DEFAULT_WINREJ_COLOR
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

    return open_eegbrowser(model)


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
        accept_label=str(options["butlabel"]) if not _is_empty(options["command"]) else None,
        mark_color=_normalize_rgb(options["wincolor"], "wincolor"),
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


def normalize_events(events: Any) -> list[BrowserEvent]:
    """Normalize EEGLAB event structures for rendering."""
    if events is None or _is_empty(events):
        return []
    event_items = _event_items(events)
    labels: list[str] = []
    normalized = []
    for event in event_items:
        if "latency" not in event:
            continue
        label = str(_scalar(event.get("type", "")))
        if label not in labels:
            labels.append(label)
        duration = event.get("duration")
        normalized.append(
            BrowserEvent(
                type=label,
                latency=float(_scalar(event["latency"])),
                duration=None if duration is None or _is_empty(duration) else float(_scalar(duration)),
                color_index=labels.index(label),
            )
        )
    return normalized


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
    options.setdefault("butlabel", "REJECT")
    options.setdefault("winrej", None)
    options.setdefault("wincolor", DEFAULT_WINREJ_COLOR)
    options.setdefault("events", source_eeg.get("event", []) if source_eeg is not None else [])
    options.setdefault("submean", "off")
    options.setdefault("eloc_file", source_eeg.get("chanlocs", None) if source_eeg is not None else None)
    options.setdefault("scale", "on")
    options.setdefault("color", "off")
    options.setdefault("freqs", None)
    options.setdefault("freqlimits", None)
    options.setdefault("component", False)
    if options["spacing"] is None or float(options["spacing"]) == 0:
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
        return tuple(f"Comp {index}" for index in range(1, n_channels + 1))
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
    "build_eegplot_model",
    "browser_window_duration",
    "copy_model_with_state",
    "decimate_minmax",
    "eegplot",
    "event_latency_to_sample",
    "flatten_browser_data",
    "normalize_browser_data",
    "normalize_events",
    "normalize_trace_colors",
    "normalize_winrej",
    "parse_eegplot_options",
    "time_to_sample",
    "visible_sample_bounds",
]
