"""Spatial gradients of EEG scalp maps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.popfunc._chanutils import chanlocs_as_list
from eegprep.functions.popfunc.plot_utils import show_figures
from eegprep.functions.sigprocfunc.readlocs import readlocs
from eegprep.functions.sigprocfunc.topoplot import griddata_v4


def gradmap(maps: Any, locations: Any, draw: bool | int = False) -> tuple[np.ndarray, np.ndarray]:
    """Compute x/y gradients of one or more scalp maps at electrode sites.

    Args:
        maps: Values shaped ``channels x maps`` or a single channel vector.
        locations: Channel-location records, a location filename, complex
            ``x + yj`` positions, or an ``(channels, 2)`` coordinate array.
        draw: Draw interpolated contours and gradient arrows when true.

    Returns:
        ``(grad_x, grad_y)`` arrays shaped ``channels x maps``.
    """
    values = np.asarray(maps, dtype=float)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] == 0:
        raise ValueError("maps must contain at least three channels and one map")
    if not np.all(np.isfinite(values)):
        raise ValueError("maps must contain only finite values")
    x, y = _coordinates(locations)
    if x.size != values.shape[0]:
        raise ValueError("locations must contain one position per map channel")
    if np.unique(np.column_stack((x, y)), axis=0).shape[0] < 3:
        raise ValueError("at least three distinct electrode positions are required")

    grid_scale = 2 * values.shape[0] + 5
    axis = np.linspace(-0.5, 0.5, grid_scale)
    grid_x, grid_y = np.meshgrid(axis, axis)
    nearest_x = np.abs(axis[:, np.newaxis] - x).argmin(axis=0)
    nearest_y = np.abs(axis[:, np.newaxis] - y).argmin(axis=0)
    grad_x = np.empty_like(values, dtype=float)
    grad_y = np.empty_like(values, dtype=float)
    plot_data = []

    for map_index in range(values.shape[1]):
        interpolated = griddata_v4(x, y, values[:, map_index], grid_x, grid_y)
        grid_grad_y, grid_grad_x = np.gradient(interpolated)
        grad_x[:, map_index] = grid_grad_x[nearest_y, nearest_x]
        grad_y[:, map_index] = grid_grad_y[nearest_y, nearest_x]
        plot_data.append((interpolated, grid_grad_x, grid_grad_y))

    if bool(draw):
        _draw_gradients(axis, plot_data)
    return grad_x, grad_y


def _coordinates(locations: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(locations, (str, Path)):
        locations = readlocs(locations)
    array = np.asarray(locations)
    if np.iscomplexobj(array) and array.ndim == 1:
        x = np.real(array).astype(float)
        y = np.imag(array).astype(float)
    elif array.ndim == 2 and array.shape[1] == 2 and np.issubdtype(array.dtype, np.number):
        x = np.asarray(array[:, 0], dtype=float)
        y = np.asarray(array[:, 1], dtype=float)
    else:
        records = chanlocs_as_list(locations)
        try:
            theta = np.deg2rad(np.asarray([record["theta"] for record in records], dtype=float))
            radius = np.asarray([record["radius"] for record in records], dtype=float)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("channel locations require finite theta and radius values") from error
        x = np.cos(theta) * radius
        y = np.sin(theta) * radius
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("electrode coordinates must be finite")
    return x, y


def _draw_gradients(axis: np.ndarray, maps: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
    columns = int(np.ceil(np.sqrt(len(maps))))
    rows = int(np.ceil(len(maps) / columns))
    figure, axes = plt.subplots(rows, columns, squeeze=False)
    grid_x, grid_y = np.meshgrid(axis, axis)
    mask = np.hypot(grid_x, grid_y) > 0.5
    for index, (interpolated, grad_x, grad_y) in enumerate(maps):
        target = axes.flat[index]
        display = np.where(mask, np.nan, interpolated)
        target.contour(grid_x, grid_y, display)
        step = max(1, axis.size // 20)
        target.quiver(
            grid_x[::step, ::step],
            grid_y[::step, ::step],
            np.where(mask, np.nan, grad_x)[::step, ::step],
            np.where(mask, np.nan, grad_y)[::step, ::step],
        )
        target.set_title(f"Map {index + 1}")
        target.set_aspect("equal")
        target.set_axis_off()
    for target in axes.flat[len(maps) :]:
        target.set_visible(False)
    figure.tight_layout()
    show_figures(figure)


__all__ = ["gradmap"]
