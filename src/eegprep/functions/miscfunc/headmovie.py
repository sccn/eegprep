"""Legacy 3-D scalp-movie compatibility wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from eegprep.functions.miscfunc.eegmovie import eegmovie, movie_limits


def headmovie(
    data: Any,
    elec_loc: Any,
    spline_file: str | Path | None = None,
    srate: float = 0,
    title: str = "",
    camerapath: Any = None,
    movieframes: Any = None,
    minmax: Any = None,
    startsec: float = 0,
    *args: Any,
    plot: str | bool = "on",
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Render a 3-D scalp movie using EEGPrep's maintained ``eegmovie`` path.

    ``headmovie`` is deprecated in EEGLAB, but keeping this thin wrapper makes
    historical scripts usable while sharing the tested renderer.
    """
    values = np.asarray(data, dtype=float)
    if np.isscalar(spline_file) and spline_file == 0:
        spline_file = None
    headplot_options = list(args)
    for name, value in kwargs.items():
        headplot_options.extend((name, value))
    movie, colormap = eegmovie(
        values,
        srate,
        elec_loc,
        mode="3D",
        headplotopt=headplot_options,
        title=title,
        camerapath=camerapath,
        movieframes=movieframes,
        minmax=minmax,
        startsec=startsec,
        spline_file=spline_file,
        plot=plot,
    )
    lower, upper = movie_limits(values, minmax)
    return movie, colormap, lower, upper


__all__ = ["headmovie"]
