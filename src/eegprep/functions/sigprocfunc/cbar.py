"""EEGLAB-compatible full and partial color bars."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap


def cbar(
    arg: str | Axes | int | None = "vert",
    colors: int | Sequence[int] = 0,
    minmax: Sequence[float] | None = None,
    grad: int = 5,
    *,
    ax: Axes | None = None,
    cmap: str | Colormap | None = None,
) -> Axes:
    """Display a full or partial color bar, following EEGLAB ``cbar``.

    Args:
        arg: ``"vert"``, ``"horiz"``, ``"pos"``, or an existing axes.
            Passing an axes draws into it and infers orientation from its shape.
        colors: One-based colormap indices, or an integer count of colors to
            remove from the upper end. Zero displays the full colormap.
        minmax: Optional values represented by the two ends of the color bar.
        grad: Number of tick labels when ``minmax`` is supplied.
        ax: Existing axes to draw into. This is the keyword equivalent of
            passing an axes as ``arg``.
        cmap: Matplotlib colormap name or object. The source axes' colormap is
            used when possible, otherwise Matplotlib's default is used.

    Returns:
        The axes containing the color strip.
    """
    orientation, target, positive = _resolve_target(arg, ax)
    parent = target if target is not None else plt.gca()
    colormap = _resolve_colormap(parent, cmap)
    rgba = _selected_colors(colormap, colors, positive=positive)
    target = target or _create_colorbar_axes(parent, orientation)

    if orientation == "vert":
        target.imshow(rgba[:, np.newaxis, :], origin="lower", aspect="auto", extent=(0.0, 1.0, 0.0, 1.0))
        target.set_xticks([])
        target.yaxis.tick_right()
    else:
        target.imshow(rgba[np.newaxis, :, :], origin="lower", aspect="auto", extent=(0.0, 1.0, 0.0, 1.0))
        target.set_yticks([])

    if minmax is not None:
        _set_value_ticks(target, orientation, minmax, grad)
    target.set_gid("cbar")
    return target


def _resolve_target(arg: str | Axes | int | None, ax: Axes | None) -> tuple[str, Axes | None, bool]:
    if isinstance(arg, Axes):
        if ax is not None and ax is not arg:
            raise ValueError("cbar received two different target axes")
        target = arg
        bounds = target.get_position().bounds
        return ("horiz" if bounds[2] > bounds[3] else "vert"), target, False
    if arg == 0:
        arg = "vert"
    if ax is not None:
        bounds = ax.get_position().bounds
        orientation = str(arg or ("horiz" if bounds[2] > bounds[3] else "vert")).lower()
        if orientation not in {"vert", "horiz", "pos"}:
            raise ValueError("cbar orientation must be 'vert', 'horiz', or 'pos'")
        return ("vert" if orientation == "pos" else orientation), ax, orientation == "pos"
    orientation = str(arg or "vert").lower()
    if orientation not in {"vert", "horiz", "pos"}:
        raise ValueError("cbar orientation must be 'vert', 'horiz', or 'pos'")
    return ("vert" if orientation == "pos" else orientation), None, orientation == "pos"


def _resolve_colormap(ax: Axes, value: str | Colormap | None) -> Colormap:
    if isinstance(value, Colormap):
        return value
    if value is not None:
        return plt.get_cmap(value)
    for artist in (*ax.images, *ax.collections):
        artist_cmap = getattr(artist, "cmap", None)
        if artist_cmap is not None:
            return artist_cmap
    return plt.get_cmap()


def _selected_colors(cmap: Colormap, colors: int | Sequence[int], *, positive: bool) -> np.ndarray:
    samples = np.asarray(cmap(np.linspace(0.0, 1.0, cmap.N)), dtype=float)
    raw = np.asarray(colors)
    if raw.ndim == 0:
        truncate = int(raw.item())
        if truncate < 0 or truncate >= samples.shape[0]:
            raise ValueError("cbar color truncation must leave at least one colormap color")
        first = int(np.ceil(samples.shape[0] / 2.0)) - 1 if positive else 0
        return samples[first : samples.shape[0] - truncate]
    indices = np.asarray(colors, dtype=int).ravel()
    if indices.size == 0:
        raise ValueError("cbar colors must not be empty")
    if np.any(indices < 1) or np.any(indices > samples.shape[0]):
        raise ValueError("cbar color indices exceed the active colormap")
    return samples[indices - 1]


def _create_colorbar_axes(parent: Axes, orientation: str) -> Axes:
    figure = parent.figure
    left, bottom, width, height = parent.get_position().bounds
    if orientation == "vert":
        return figure.add_axes([left + width + 0.02, bottom, 0.04 * width, height])
    parent.set_position([left, bottom + 0.175 * height, width, 0.825 * height])
    return figure.add_axes([left, bottom, width, 0.075 * height])


def _set_value_ticks(ax: Axes, orientation: str, minmax: Sequence[float], grad: int) -> None:
    limits = np.asarray(minmax, dtype=float).ravel()
    if limits.size != 2 or not np.isfinite(limits).all():
        raise ValueError("cbar minmax must contain two finite values")
    if grad < 2:
        raise ValueError("cbar grad must be at least 2")
    positions = np.linspace(0.0, 1.0, int(grad))
    labels = _rounded_labels(limits, int(grad))
    if orientation == "vert":
        ax.set_yticks(positions, labels)
    else:
        ax.set_xticks(positions, labels)


def _rounded_labels(limits: np.ndarray, grad: int) -> np.ndarray:
    labels = np.linspace(float(limits[0]), float(limits[1]), grad)
    maximum = float(np.max(np.abs(limits)))
    if maximum == 0:
        return labels
    decade = int(np.floor(np.log10(maximum)))
    if decade < 1:
        scale = 10.0 ** (1 - decade)
        return np.round(labels * scale) / scale
    if decade == 1:
        scale = 10.0 ** (2 - decade)
        return np.round(labels * scale) / scale
    return np.round(labels)


__all__ = ["cbar"]
