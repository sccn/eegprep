"""Shared implementation for logarithmic image plots."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh
import numpy as np

from eegprep.functions.popfunc.plot_utils import show_figures


def log_image(
    times: Any,
    freqs: Any,
    data: Any,
    clim: Any,
    xticks: Any,
    yticks: Any,
    properties: dict[str, Any],
    *,
    log_x: bool,
    ax: Any,
) -> QuadMesh:
    x = _axis_values(times, "times", positive=log_x)
    y = _axis_values(freqs, "freqs", positive=True)
    values = np.asarray(data)
    if values.ndim != 2 or values.shape != (y.size, x.size):
        raise ValueError(f"data must have shape ({y.size}, {x.size}) for the supplied freqs and times")

    own_figure = ax is None
    if own_figure:
        figure, ax = plt.subplots()
    else:
        figure = ax.figure
    limits = _color_limits(clim)
    mesh = ax.pcolormesh(
        _cell_edges(x, logarithmic=log_x),
        _cell_edges(y, logarithmic=True),
        values,
        shading="flat",
        vmin=None if limits is None else limits[0],
        vmax=None if limits is None else limits[1],
    )
    ax.set_xscale("log" if log_x else "linear")
    ax.set_yscale("log")
    x_tick_values = _ticks(x, xticks, logarithmic=log_x)
    if x_tick_values is not None:
        ax.set_xticks(x_tick_values)
    ax.set_yticks(_ticks(y, yticks, logarithmic=True))
    ax.minorticks_off()
    ax.tick_params(direction="out")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("none")
    _apply_axes_properties(ax, properties)
    if own_figure:
        show_figures(figure)
    return mesh


def _axis_values(value: Any, name: str, *, positive: bool) -> np.ndarray:
    values = np.asarray(value, dtype=float).ravel()
    if values.size < 2:
        raise ValueError(f"{name} must contain at least two values")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    if positive and np.any(values <= 0):
        raise ValueError(f"{name} must be positive for a logarithmic axis")
    if np.any(np.diff(values) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return values


def _cell_edges(values: np.ndarray, *, logarithmic: bool) -> np.ndarray:
    if logarithmic:
        interior = np.sqrt(values[:-1] * values[1:])
        return np.concatenate(([values[0] ** 2 / interior[0]], interior, [values[-1] ** 2 / interior[-1]]))
    interior = (values[:-1] + values[1:]) / 2.0
    return np.concatenate(
        (
            [values[0] - (interior[0] - values[0])],
            interior,
            [values[-1] + values[-1] - interior[-1]],
        )
    )


def _ticks(values: np.ndarray, requested: Any, *, logarithmic: bool) -> np.ndarray | None:
    if requested is not None and np.asarray(requested).size:
        ticks = np.asarray(requested, dtype=float).ravel()
        if not np.all(np.isfinite(ticks)):
            raise ValueError("tick values must be finite")
        if logarithmic and np.any(ticks <= 0):
            raise ValueError("tick values must be positive for a logarithmic axis")
        return ticks
    if not logarithmic:
        return None
    raw = np.exp(np.linspace(np.log(values[0]), np.log(values[-1]), 10))
    rounded = np.ceil(raw - np.finfo(float).eps * np.maximum(1.0, np.abs(raw)))
    return np.unique(rounded)


def _color_limits(clim: Any) -> tuple[float, float] | None:
    if clim is None or np.asarray(clim).size == 0:
        return None
    values = np.asarray(clim, dtype=float).ravel()
    if values.size != 2 or not np.all(np.isfinite(values)) or values[0] >= values[1]:
        raise ValueError("clim must contain two increasing finite values")
    return float(values[0]), float(values[1])


def _apply_axes_properties(ax: Any, properties: dict[str, Any]) -> None:
    for name, value in properties.items():
        normalized = name.replace("_", "").lower()
        if normalized == "xgrid":
            ax.xaxis.grid(_on(value))
        elif normalized == "ygrid":
            ax.yaxis.grid(_on(value))
        elif normalized == "grid":
            ax.grid(_on(value))
        elif normalized == "xlim":
            ax.set_xlim(value)
        elif normalized == "ylim":
            ax.set_ylim(value)
        elif normalized == "xlabel":
            ax.set_xlabel(value)
        elif normalized == "ylabel":
            ax.set_ylabel(value)
        elif normalized == "title":
            ax.set_title(value)
        else:
            raise ValueError(f"unsupported axes property: {name}")


def _on(value: Any) -> bool:
    if isinstance(value, str):
        if value.lower() in {"on", "true"}:
            return True
        if value.lower() in {"off", "false"}:
            return False
    return bool(value)
