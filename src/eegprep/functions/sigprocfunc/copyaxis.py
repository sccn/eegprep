"""Copy a Matplotlib axes into a new figure, like EEGLAB ``copyaxis``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def copyaxis(command: Callable[[Axes], Any] | None = None, *, source: Axes | None = None) -> Figure:
    """Copy the current axes into an enlarged standalone figure.

    Common scientific plot content (lines and images), labels, scales, limits,
    and legends are reproduced. A Python callback may customize the copied
    axes; MATLAB command strings are intentionally not evaluated.

    Args:
        command: Optional callback receiving the copied axes.
        source: Axes to copy. The current axes is used by default.

    Returns:
        The new figure containing the copied axes.
    """
    if command is not None and not callable(command):
        raise TypeError("copyaxis command must be a callable receiving the copied axes")
    source = source or plt.gca()
    source_figure = source.get_figure(root=True)
    if source_figure is None:
        raise ValueError("copyaxis source must belong to a figure")
    figure = plt.figure(figsize=source_figure.get_size_inches(), facecolor=source_figure.get_facecolor())
    target = figure.add_axes([0.13, 0.11, 0.775, 0.815])
    _copy_axes_content(source, target)
    if command is not None:
        command(target)
    return figure


def _copy_axes_content(source: Axes, target: Axes) -> None:
    for line in source.lines:
        (copied,) = target.plot(
            line.get_xdata(),
            line.get_ydata(),
            color=line.get_color(),
            linestyle=line.get_linestyle(),
            linewidth=line.get_linewidth(),
            marker=line.get_marker(),
            markersize=line.get_markersize(),
            markerfacecolor=line.get_markerfacecolor(),
            markeredgecolor=line.get_markeredgecolor(),
            alpha=line.get_alpha(),
            label=line.get_label(),
        )
        copied.set_drawstyle(line.get_drawstyle())
    for image in source.images:
        target.imshow(
            image.get_array(),
            cmap=image.get_cmap(),
            norm=image.norm,
            aspect=source.get_aspect(),
            interpolation=image.get_interpolation(),
            origin=image.origin,
            extent=image.get_extent(),
            alpha=image.get_alpha(),
        )
    for text in source.texts:
        target.text(
            *text.get_position(),
            text.get_text(),
            color=text.get_color(),
            fontsize=text.get_fontsize(),
            horizontalalignment=text.get_horizontalalignment(),
            verticalalignment=text.get_verticalalignment(),
            transform=target.transAxes if text.get_transform() is source.transAxes else target.transData,
        )

    target.set(
        xlim=source.get_xlim(),
        ylim=source.get_ylim(),
        xscale=source.get_xscale(),
        yscale=source.get_yscale(),
        xlabel=source.get_xlabel(),
        ylabel=source.get_ylabel(),
        title=source.get_title(),
        facecolor=source.get_facecolor(),
    )
    target.tick_params(labelsize=14)
    target.xaxis.label.set_fontsize(16)
    target.yaxis.label.set_fontsize(16)
    target.title.set_fontsize(16)
    handles, labels = target.get_legend_handles_labels()
    if source.get_legend() is not None and handles:
        target.legend(handles, labels)


__all__ = ["copyaxis"]
