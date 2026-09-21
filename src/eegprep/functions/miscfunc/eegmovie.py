"""Create headless-friendly movies of evolving EEG scalp maps."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.popfunc.plot_utils import show_figures
from eegprep.functions.sigprocfunc.headplot import headplot, headplot_setup
from eegprep.functions.sigprocfunc.readlocs import readlocs
from eegprep.functions.sigprocfunc.topoplot import topoplot

DEFAULT_MOVIE_SRATE = 256.0
DEFAULT_CAMERA_PATH = np.asarray([-127.0, 0.0, 30.0, 0.0])


def eegmovie(
    data: Any,
    srate: float = 0,
    elec_locs: Any = None,
    *args: Any,
    plot: str | bool = "on",
    spline_file: str | Path | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Render channel-by-frame data as a sequence of RGB scalp-map images.

    EEGLAB name/value options are accepted as positional pairs or keywords.
    ``movieframes`` remains 1-based at this public compatibility boundary.
    The returned movie is an unsigned-byte array shaped
    ``(frames, height, width, 3)`` and can be replayed with :func:`seemovie`.
    """
    options = _movie_options(args, kwargs)
    values = np.asarray(data, dtype=float)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] == 0:
        raise ValueError("data must be a non-empty channels x frames matrix with at least three channels")
    if not np.all(np.isfinite(values)):
        raise ValueError("data must contain only finite values")
    sample_rate = DEFAULT_MOVIE_SRATE if not srate else float(srate)
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("srate must be positive, or zero to use 256 Hz")
    locs = _channel_locations(elec_locs, values.shape[0])
    frame_indices = _frame_indices(options.pop("movieframes", None), values.shape[1])
    limits = movie_limits(values, options.pop("minmax", None))
    mode = str(options.pop("mode", "2D")).upper()
    if mode not in {"2D", "3D"}:
        raise ValueError("mode must be '2D' or '3D'")

    title = str(options.pop("title", "") or "")
    start_seconds = float(options.pop("startsec", 0.0) or 0.0)
    timecourse = _on(options.pop("timecourse", "on"))
    frame_number = _on(options.pop("framenum", "on"))
    show_time = _on(options.pop("time", "off"))
    if show_time:
        frame_number = False
    vertical_times = np.asarray(options.pop("vert", []), dtype=float).ravel()
    camera_path = _camera_path(options.pop("camerapath", None)) if mode == "3D" else DEFAULT_CAMERA_PATH[np.newaxis, :]
    topoplot_options = _plot_options(options.pop("topoplotopt", None))
    headplot_options = _plot_options(options.pop("headplotopt", None))
    if options:
        raise ValueError(f"eegmovie: unrecognized option: {next(iter(options))!r}")

    figure = plt.figure(figsize=(7.5, 4.5), dpi=80)
    if timecourse:
        layout = figure.add_gridspec(1, 2, width_ratios=(3.0, 1.0))
        map_axis = figure.add_subplot(layout[0, 0], projection="3d" if mode == "3D" else None)
        trace_axis = figure.add_subplot(layout[0, 1])
        _draw_timecourse(trace_axis, values, sample_rate, start_seconds, vertical_times)
    else:
        map_axis = figure.add_subplot(111, projection="3d" if mode == "3D" else None)
        trace_axis = None
    figure.patch.set_facecolor("white")

    temporary = (
        TemporaryDirectory(prefix="eegprep-eegmovie-") if mode == "3D" and spline_file is None else nullcontext()
    )
    with temporary as temporary_directory:
        active_spline = spline_file
        if mode == "3D" and active_spline is None:
            active_spline = Path(str(temporary_directory)) / "eegmovie.spl"
            headplot_setup(locs, active_spline)
        views = _camera_views(camera_path, frame_indices)
        rendered = []
        for output_index, data_index in enumerate(frame_indices):
            map_axis.clear()
            if mode == "2D":
                topoplot(
                    values[:, data_index],
                    locs,
                    axes=map_axis,
                    maplimits=limits,
                    **topoplot_options,
                )
            else:
                headplot(
                    values[:, data_index],
                    active_spline,
                    ax=map_axis,
                    maplimits=limits,
                    view=views[output_index],
                    tight_layout=False,
                    **headplot_options,
                )
            if title:
                map_axis.set_title(title)
            if frame_number:
                _frame_text(map_axis, mode, str(output_index + 1))
            elif show_time:
                seconds = start_seconds + data_index / sample_rate
                _frame_text(map_axis, mode, f"{seconds:.3f} s")
            cursor = None
            if trace_axis is not None:
                cursor = trace_axis.axvline(start_seconds + data_index / sample_rate, color="b")
            rendered.append(_capture_rgb(figure))
            if cursor is not None:
                cursor.remove()

    movie = np.stack(rendered)
    colormap = np.vstack((plt.get_cmap("turbo")(np.linspace(0, 1, 64))[:, :3], [1.0, 1.0, 1.0]))
    show_figures(figure, plot=plot)
    return movie, colormap


def movie_limits(data: np.ndarray, minmax: Any = None) -> tuple[float, float]:
    """Return EEGLAB's symmetric movie color limits."""
    if minmax is not None and np.asarray(minmax).size:
        requested = np.asarray(minmax, dtype=float).ravel()
        if requested.size == 1 and requested[0] != 0:
            requested = np.asarray([-abs(requested[0]), abs(requested[0])])
        if requested.size == 2 and requested[0] < requested[1] and np.all(np.isfinite(requested)):
            return float(requested[0]), float(requested[1])
        if requested.size != 1 or requested[0] != 0:
            raise ValueError("minmax must be zero, a positive scalar, or two increasing finite values")
    data_min = float(np.min(data))
    data_max = float(np.max(data))
    absolute_maximum = max(abs(data_min), abs(data_max))
    padding = 0.05 * (data_max - data_min)
    limit = absolute_maximum + padding
    if limit == 0:
        limit = 1.0
    return -limit, limit


def _movie_options(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    if args and not isinstance(args[0], str):
        names = ("title", "movieframes", "minmax", "startsec")
        legacy_count = min(len(args), len(names))
        options = {names[index]: args[index] for index in range(legacy_count)}
        remainder = args[legacy_count:]
        if remainder:
            options["topoplotopt"] = remainder
    else:
        if len(args) % 2:
            raise TypeError("eegmovie options must be name/value pairs")
        options = {}
        for index in range(0, len(args), 2):
            if not isinstance(args[index], str):
                raise TypeError("eegmovie option names must be strings")
            options[args[index].lower()] = args[index + 1]
    for name, value in kwargs.items():
        key = name.lower()
        if key in options:
            raise TypeError(f"eegmovie option {name!r} was supplied twice")
        options[key] = value
    return options


def _channel_locations(elec_locs: Any, channel_count: int) -> list[dict[str, Any]]:
    if elec_locs is None or (np.isscalar(elec_locs) and elec_locs == 0):
        raise ValueError("elec_locs must provide one channel position per data row")
    if isinstance(elec_locs, (str, Path)):
        locs = readlocs(elec_locs)
    else:
        positions = np.asarray(elec_locs)
        if np.iscomplexobj(positions) and positions.ndim == 1:
            locs = []
            for index, position in enumerate(positions):
                locs.append(
                    {
                        "labels": str(index + 1),
                        "theta": float(np.rad2deg(np.angle(position))),
                        "radius": float(abs(position)),
                    }
                )
        else:
            locs = chanlocs_as_list(elec_locs)
    if len(locs) != channel_count:
        raise ValueError("elec_locs must provide one channel position per data row")
    return locs


def _frame_indices(movieframes: Any, frame_count: int) -> np.ndarray:
    if movieframes is None or np.asarray(movieframes).size == 0:
        return np.arange(frame_count)
    requested = np.asarray(movieframes)
    if requested.size == 1 and requested.reshape(-1)[0] == 0:
        return np.arange(frame_count)
    numeric = np.asarray(movieframes, dtype=float).ravel()
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
        raise ValueError("movieframes must contain integer frame numbers")
    indices = numeric.astype(int) - 1
    if np.any(indices < 0) or np.any(indices >= frame_count):
        raise ValueError("movieframes contains a frame outside the data")
    return indices


def _plot_options(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    sequence = list(value)
    if len(sequence) % 2:
        raise TypeError("plot options must be name/value pairs")
    return {str(sequence[index]): sequence[index + 1] for index in range(0, len(sequence), 2)}


def _camera_path(value: Any) -> np.ndarray:
    if value is None or (np.asarray(value).size == 1 and float(np.asarray(value).item()) == 0):
        return DEFAULT_CAMERA_PATH[np.newaxis, :].copy()
    path = np.asarray(value, dtype=float)
    if path.ndim == 1:
        if not 1 <= path.size <= 4:
            raise ValueError("camerapath must have one to four columns")
        defaults = DEFAULT_CAMERA_PATH.copy()
        defaults[: path.size] = path
        path = defaults[np.newaxis, :]
    if path.ndim != 2 or path.shape[1] != 4 or not np.all(np.isfinite(path)):
        raise ValueError("camerapath must be a finite matrix with four columns")
    return path


def _camera_views(path: np.ndarray, frame_indices: np.ndarray) -> list[tuple[float, float]]:
    azimuth, azimuth_step, elevation, elevation_step = path[0]
    next_row = 1
    views = []
    for frame_index in frame_indices:
        frame_number = int(frame_index) + 1
        if next_row < len(path) and frame_number == int(path[next_row, 0]):
            azimuth_step = path[next_row, 1]
            elevation_step = path[next_row, 3]
            next_row += 1
        views.append((float(azimuth), float(np.clip(elevation, -89.99, 89.99))))
        azimuth += azimuth_step
        elevation += elevation_step
    return views


def _draw_timecourse(
    axis: Any,
    data: np.ndarray,
    srate: float,
    start_seconds: float,
    vertical_times: np.ndarray,
) -> None:
    times = start_seconds + np.arange(data.shape[1]) / srate
    channel_range = np.ptp(data, axis=1)
    spacing = float(np.max(channel_range)) if channel_range.size else 1.0
    if spacing == 0:
        spacing = 1.0
    offsets = spacing * np.arange(data.shape[0])
    axis.plot(times, -data.T + offsets, color="r", linewidth=0.6)
    for latency in vertical_times:
        axis.axvline(float(latency), color="k", linewidth=0.8)
    axis.set_xlim(times[0], times[-1] if len(times) > 1 else times[0] + 1 / srate)
    axis.set_yticks(offsets)
    axis.set_yticklabels([str(index + 1) for index in range(data.shape[0])])
    axis.grid(axis="y")
    axis.set_xlabel("Time (s)")


def _capture_rgb(figure: Figure) -> np.ndarray:
    original_canvas = figure.canvas
    if isinstance(original_canvas, FigureCanvasAgg):
        original_canvas.draw()
        return np.asarray(original_canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    raster_canvas = FigureCanvasAgg(figure)
    raster_canvas.draw()
    frame = np.asarray(raster_canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    figure.set_canvas(original_canvas)
    return frame


def _frame_text(axis: Any, mode: str, value: str) -> None:
    if mode == "3D":
        axis.text2D(0.03, 0.03, value, transform=axis.transAxes)
    else:
        axis.text(0.03, 0.03, value, transform=axis.transAxes)


def _on(value: Any) -> bool:
    if isinstance(value, str):
        if value.lower() == "on":
            return True
        if value.lower() == "off":
            return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError("on/off options must be 'on', 'off', True, or False")


__all__ = ["eegmovie"]
