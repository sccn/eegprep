"""Simple 3-D scalp map helper for EEGLAB ``headplot``-style outputs."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.popfunc._chanutils import chanlocs_as_list


def headplot(values: Any, chanlocs: Any, *, title: str = "3-D scalp map"):
    """Plot channel or component weights on a 3-D head sphere.

    EEGLAB's full ``headplot`` uses spline/mesh resources. EEGPrep currently
    provides a standalone static 3-D map over channel coordinates and fails
    clearly when usable 3-D channel locations are missing.
    """
    locs = chanlocs_as_list(chanlocs)
    coordinates = np.asarray([_xyz(loc) for loc in locs], dtype=float)
    if coordinates.size == 0 or not np.isfinite(coordinates).all():
        raise ValueError("headplot requires channel locations with X/Y/Z coordinates")
    data = np.asarray(values, dtype=float).ravel()
    if data.size != coordinates.shape[0]:
        raise ValueError("headplot values must have one value per channel")

    fig = plt.figure(figsize=(5.5, 5.0))
    ax = fig.add_subplot(111, projection="3d")
    _draw_head_sphere(ax)
    scatter = ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        coordinates[:, 2],
        c=data,
        cmap="RdBu_r",
        s=70,
        depthshade=False,
    )
    for point, loc in zip(coordinates, locs):
        label = str(loc.get("labels") or "")
        if label:
            ax.text(point[0], point[1], point[2], label, fontsize=8)
    fig.colorbar(scatter, ax=ax, shrink=0.7)
    ax.set_title(title)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()
    return fig


def _xyz(chanloc: dict[str, Any]) -> tuple[float, float, float]:
    if all(key in chanloc and chanloc[key] not in (None, "") for key in ("X", "Y", "Z")):
        return float(chanloc["X"]), float(chanloc["Y"]), float(chanloc["Z"])
    theta = np.deg2rad(float(chanloc.get("theta", 0) or 0))
    radius = float(chanloc.get("radius", 0.5) or 0.5)
    return float(radius * np.sin(theta)), float(radius * np.cos(theta)), 0.0


def _draw_head_sphere(ax: Any) -> None:
    u = np.linspace(0, 2 * np.pi, 32)
    v = np.linspace(0, np.pi, 16)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color="lightgray", linewidth=0.4, alpha=0.45)


__all__ = ["headplot"]
