"""Render event timing across epoched EEG data."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from eegprep.functions.miscfunc.unique_cell_string import unique_cell_string
from eegprep.functions.popfunc._pop_utils import parse_key_value_args
from eegprep.functions.popfunc.plot_utils import show_figures


_MATLAB_LINE_COLORS = np.asarray(
    [
        [0.0000, 0.4470, 0.7410],
        [0.8500, 0.3250, 0.0980],
        [0.9290, 0.6940, 0.1250],
        [0.4940, 0.1840, 0.5560],
        [0.4660, 0.6740, 0.1880],
        [0.3010, 0.7450, 0.9330],
        [0.6350, 0.0780, 0.1840],
    ],
    dtype=float,
)


def show_events(
    EEG: Mapping[str, Any],
    *args: Any,
    event_thickness_coef: float = 1.0,
    event_names: Any = None,
    time_warp: Mapping[str, Any] | None = None,
    ax: Any = None,
    plot: str | bool = "on",
    image_shape: tuple[int, int] = (960, 1200),
    **kwargs: Any,
) -> np.ndarray:
    """Return an RGB raster showing epoch-relative event latencies.

    Rows represent epochs, columns span ``EEG.xmin`` to ``EEG.xmax``, and
    event types receive stable categorical colors. Events excluded by a
    ``make_timewarp`` result are dimmed. A figure is also drawn unless
    ``plot="off"`` is requested.
    """
    options = parse_key_value_args(args, kwargs, lowercase_kwargs=True)
    event_thickness_coef = float(
        options.pop("eventthicknesscoef", options.pop("event_thickness_coef", event_thickness_coef))
    )
    event_names = options.pop("eventnames", options.pop("event_names", event_names))
    time_warp = options.pop("timewarp", options.pop("time_warp", time_warp))
    if options:
        raise ValueError(f"show_events: unrecognized option: {next(iter(options))!r}")
    if not np.isfinite(event_thickness_coef) or event_thickness_coef < 0:
        raise ValueError("event_thickness_coef must be a non-negative finite number")
    height, width = (int(image_shape[0]), int(image_shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError("image_shape dimensions must be positive")

    epochs = list(EEG.get("epoch", []))
    if not epochs:
        raise ValueError("show_events requires an epoched EEG dataset")
    epoch_events = [_epoch_events(epoch) for epoch in epochs]
    names = _event_names(event_names, time_warp, epoch_events)
    if not names:
        raise ValueError("show_events found no named events to display")
    xmin_ms = float(EEG.get("xmin", 0.0)) * 1000.0
    xmax_ms = float(EEG.get("xmax", 0.0)) * 1000.0
    if not np.isfinite(xmin_ms) or not np.isfinite(xmax_ms) or xmax_ms <= xmin_ms:
        raise ValueError("EEG.xmin and EEG.xmax must define a finite positive epoch duration")

    colors = _line_colors(len(names))
    marker_width = _event_marker_width(epoch_events, width, xmax_ms - xmin_ms, event_thickness_coef)
    image = np.zeros((height, width, 3), dtype=float)
    for epoch_index, events in enumerate(epoch_events):
        row_start = round(epoch_index * height / len(epochs))
        row_stop = round((epoch_index + 1) * height / len(epochs))
        for event_type, latency in events:
            if event_type not in names:
                continue
            event_index = names.index(event_type)
            center = round(width * (latency - xmin_ms) / (xmax_ms - xmin_ms))
            start = max(0, center - marker_width // 2)
            stop = min(width, start + marker_width)
            if stop <= 0 or start >= width:
                continue
            color = colors[event_index]
            if not _accepted(time_warp, epoch_index, latency):
                color = color * 0.3
            image[row_start:row_stop, start:stop] = np.maximum(image[row_start:row_stop, start:stop], color)

    own_figure = ax is None
    if own_figure:
        figure, ax = plt.subplots()
    else:
        figure = ax.figure
    ax.imshow(
        image,
        aspect="auto",
        origin="upper",
        extent=(xmin_ms, xmax_ms, len(epochs) + 0.5, 0.5),
    )
    handles = [
        Line2D([], [], color=color, linewidth=8, label=name.replace("_", "-")) for name, color in zip(names, colors)
    ]
    ax.legend(handles=handles, loc="upper left")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Epochs")
    if own_figure:
        figure.tight_layout()
        show_figures(figure, plot=plot)
    return image


def _epoch_events(epoch: Any) -> list[tuple[str, float]]:
    if not isinstance(epoch, Mapping):
        raise ValueError("EEG.epoch entries must be mappings")
    types = _values(epoch.get("eventtype", []))
    latencies = _values(epoch.get("eventlatency", []))
    if len(types) != len(latencies):
        raise ValueError("each epoch must contain one eventlatency per eventtype")
    result = []
    for event_type, latency in zip(types, latencies):
        numeric_latency = float(_scalar(latency))
        if not np.isfinite(numeric_latency):
            raise ValueError("epoch event latencies must be finite scalars")
        result.append((_event_string(event_type), numeric_latency))
    return result


def _values(value: Any) -> list[Any]:
    array = np.asarray(value, dtype=object)
    if array.size == 0:
        return []
    return array.reshape(-1).tolist()


def _scalar(value: Any) -> Any:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("epoch event latencies must be finite scalars")
    return array.reshape(-1)[0].item()


def _event_string(value: Any) -> str:
    value = _scalar(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return format(float(value), "g")
    return str(value)


def _event_names(
    requested: Any,
    time_warp: Mapping[str, Any] | None,
    epoch_events: list[list[tuple[str, float]]],
) -> list[str]:
    if requested is not None and np.asarray(requested, dtype=object).size:
        values = [_event_string(value) for value in _values(requested)]
    elif time_warp:
        sequence = time_warp.get("event_sequence", time_warp.get("eventSequence", []))
        values = []
        for value in _values(sequence):
            if isinstance(value, (list, tuple, np.ndarray)):
                values.extend(_event_string(item) for item in _values(value))
            else:
                values.append(_event_string(value))
    else:
        values = [event_type for events in epoch_events for event_type, _latency in events]
    return unique_cell_string(values)


def _event_marker_width(
    epoch_events: list[list[tuple[str, float]]],
    image_width: int,
    duration_ms: float,
    coefficient: float,
) -> int:
    intervals = []
    for events in epoch_events:
        latencies = sorted(latency for _event_type, latency in events)
        intervals.extend(np.diff(latencies).tolist())
    if not intervals:
        return max(1, round(0.05 * coefficient))
    quantile = float(np.quantile(np.asarray(intervals, dtype=float), 0.2))
    return max(1, round(coefficient * 0.5 * image_width * quantile / duration_ms))


def _line_colors(count: int) -> np.ndarray:
    return np.vstack([_MATLAB_LINE_COLORS[index % len(_MATLAB_LINE_COLORS)] for index in range(count)])


def _accepted(time_warp: Mapping[str, Any] | None, epoch_index: int, latency: float) -> bool:
    if not time_warp:
        return True
    epochs = np.asarray(time_warp.get("epochs", []), dtype=int).ravel()
    python_style = "event_sequence" in time_warp
    target = epoch_index if python_style else epoch_index + 1
    matches = np.flatnonzero(epochs == target)
    if matches.size == 0:
        return False
    latencies = np.asarray(time_warp.get("latencies", []), dtype=float)
    if latencies.size == 0:
        return True
    row = latencies.reshape(len(epochs), -1)[matches[0]]
    return bool(np.any(np.isclose(row, latency, rtol=0.0, atol=1e-9)))


__all__ = ["show_events"]
