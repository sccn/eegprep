"""Display a matrix with logarithmic time and frequency axes."""

from __future__ import annotations

from typing import Any

from matplotlib.collections import QuadMesh

from eegprep.functions.miscfunc._log_image import log_image
from eegprep.functions.miscfunc.imagesclogy import _properties


def imagescloglog(
    times: Any,
    freqs: Any,
    data: Any,
    clim: Any = None,
    xticks: Any = None,
    yticks: Any = None,
    *args: Any,
    ax: Any = None,
    **kwargs: Any,
) -> QuadMesh:
    """Plot frequency-by-time values on logarithmic x- and y-axes."""
    properties = _properties(args, kwargs)
    return log_image(times, freqs, data, clim, xticks, yticks, properties, log_x=True, ax=ax)


__all__ = ["imagescloglog"]
