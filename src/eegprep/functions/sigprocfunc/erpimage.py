"""ERP image plotting helper."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def erpimage(data: Any, *, times: Any = None, title: str = "", sort_values: Any = None):
    """Plot trials as an EEGLAB-style ERP image plus the average ERP."""
    values = np.asarray(data, dtype=float)
    if values.ndim != 2:
        raise ValueError("erpimage data must be points x trials")
    points, trials = values.shape
    x_values = (
        np.asarray(times, dtype=float).ravel()
        if times is not None and len(np.asarray(times).ravel())
        else np.arange(points)
    )
    if x_values.size != points:
        raise ValueError("times must match the number of data points")
    order = _trial_order(values, sort_values)
    image = values[:, order].T
    fig, (image_ax, erp_ax) = plt.subplots(
        2,
        1,
        figsize=(7.5, 5.0),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    extent = [float(x_values[0]), float(x_values[-1]), 1, trials]
    im = image_ax.imshow(image, aspect="auto", origin="lower", extent=extent, cmap="RdBu_r")
    image_ax.set_ylabel("Trials")
    image_ax.set_title(title or "ERP image")
    fig.colorbar(im, ax=image_ax, shrink=0.85)
    erp_ax.plot(x_values, np.nanmean(values, axis=1), color="black")
    erp_ax.axhline(0, color="0.7", linewidth=0.6)
    erp_ax.set_xlabel("Time (ms)")
    erp_ax.set_ylabel("ERP")
    fig.tight_layout()
    return fig, image


def _trial_order(values: np.ndarray, sort_values: Any) -> np.ndarray:
    if sort_values is None or (isinstance(sort_values, (list, tuple)) and not sort_values):
        return np.arange(values.shape[1])
    sort_array = np.asarray(sort_values, dtype=float).ravel()
    if sort_array.size != values.shape[1]:
        raise ValueError("sort_values must contain one value per trial")
    return np.argsort(sort_array)


__all__ = ["erpimage"]
