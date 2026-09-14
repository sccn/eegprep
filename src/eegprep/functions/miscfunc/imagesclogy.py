"""Display a matrix with a logarithmic frequency axis."""

from __future__ import annotations

from typing import Any

from matplotlib.collections import QuadMesh

from eegprep.functions.miscfunc._log_image import log_image


def imagesclogy(
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
    """Plot frequency-by-time values on a logarithmic y-axis.

    Extra positional arguments are interpreted as EEGLAB-style axes property
    name/value pairs. The returned ``QuadMesh`` exposes the plotted data and
    color limits for headless workflows.
    """
    properties = _properties(args, kwargs)
    return log_image(times, freqs, data, clim, xticks, yticks, properties, log_x=False, ax=ax)


def _properties(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    if len(args) % 2:
        raise TypeError("additional plot properties must be name/value pairs")
    properties = dict(kwargs)
    for index in range(0, len(args), 2):
        name = args[index]
        if not isinstance(name, str):
            raise TypeError("plot property names must be strings")
        if name in properties:
            raise TypeError(f"plot property {name!r} was supplied twice")
        properties[name] = args[index + 1]
    return properties


__all__ = ["imagesclogy"]
