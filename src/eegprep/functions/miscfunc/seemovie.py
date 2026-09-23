"""Replay RGB or indexed image sequences as Matplotlib animations."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
import numpy as np

from eegprep.functions.popfunc.plot_utils import show_figures


def seemovie(
    movie: Any,
    ntimes: int = -10,
    colormap: Any = None,
    *,
    fps: float = 10,
    ax: Any = None,
    plot: str | bool = "on",
) -> FuncAnimation:
    """Return an animation for an EEGPrep movie.

    Positive ``ntimes`` values replay forward. Zero selects EEGLAB's default
    of ten forward/backward repetitions, and negative values request that many
    forward/backward repetitions.
    """
    frames = np.asarray(movie)
    if frames.ndim not in {3, 4} or frames.shape[0] == 0:
        raise ValueError("movie must be a non-empty frames x height x width image sequence")
    if frames.ndim == 4 and frames.shape[-1] not in {3, 4}:
        raise ValueError("RGB movie frames must have three or four color channels")
    frame_rate = float(fps)
    if not np.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError("fps must be positive")
    if ntimes == 0:
        ntimes = -10
    repeats = abs(int(ntimes))
    if int(ntimes) != ntimes:
        raise ValueError("ntimes must be an integer")
    forward = list(range(frames.shape[0]))
    sequence = forward * repeats
    if ntimes <= 0:
        sequence = (forward + forward[-2:0:-1]) * repeats

    own_figure = ax is None
    if own_figure:
        figure, ax = plt.subplots()
    else:
        figure = ax.figure
    image_options: dict[str, Any] = {"animated": True}
    if frames.ndim == 3:
        if colormap is None:
            image_options["cmap"] = "gray"
        else:
            colors = np.asarray(colormap, dtype=float)
            if colors.ndim != 2 or colors.shape[1] not in {3, 4}:
                raise ValueError("colormap must contain RGB or RGBA rows")
            image_options["cmap"] = ListedColormap(colors)
    image = ax.imshow(frames[sequence[0]], **image_options)
    ax.set_axis_off()

    def update(frame_index: int) -> tuple[Any]:
        image.set_data(frames[frame_index])
        return (image,)

    animation = FuncAnimation(
        figure,
        update,
        frames=sequence,
        interval=1000.0 / frame_rate,
        blit=True,
        repeat=False,
    )
    if own_figure:
        show_figures(figure, plot=plot)
    return animation


__all__ = ["seemovie"]
