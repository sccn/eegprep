"""ERP image plotting helper."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def erpimage(
    data: Any,
    *,
    times: Any = None,
    title: str = "",
    sort_values: Any = None,
    smooth: Any = None,
    decimate: int = 1,
    caxis: Any = None,
    cbar: bool = True,
    plot_erp: bool = True,
    vert: Any = None,
):
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
    image = _decimate_trials(image, decimate)
    image = _smooth_trials(image, smooth)
    if plot_erp:
        fig, (image_ax, erp_ax) = plt.subplots(
            2,
            1,
            figsize=(7.5, 5.0),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )
    else:
        fig, image_ax = plt.subplots(figsize=(7.5, 4.2))
        erp_ax = None
    extent = [float(x_values[0]), float(x_values[-1]), 1, image.shape[0]]
    im = image_ax.imshow(image, aspect="auto", origin="lower", extent=extent, cmap="RdBu_r")
    limits = _limits(caxis)
    if limits is not None:
        im.set_clim(*limits)
    image_ax.set_ylabel("Trials")
    image_ax.set_title(title or "ERP image")
    for latency in _numeric_values(vert):
        image_ax.axvline(latency, color="black", linestyle=":", linewidth=0.8)
    if cbar:
        fig.colorbar(im, ax=image_ax, shrink=0.85)
    if erp_ax is not None:
        erp_ax.plot(x_values, np.nanmean(values, axis=1), color="black")
        erp_ax.axhline(0, color="0.7", linewidth=0.6)
        for latency in _numeric_values(vert):
            erp_ax.axvline(latency, color="black", linestyle=":", linewidth=0.8)
        erp_ax.set_xlabel("Time (ms)")
        erp_ax.set_ylabel("ERP")
    else:
        image_ax.set_xlabel("Time (ms)")
    fig.tight_layout()
    return fig, image


def _trial_order(values: np.ndarray, sort_values: Any) -> np.ndarray:
    if sort_values is None or (isinstance(sort_values, (list, tuple)) and not sort_values):
        return np.arange(values.shape[1])
    sort_array = np.asarray(sort_values, dtype=float).ravel()
    if sort_array.size != values.shape[1]:
        raise ValueError("sort_values must contain one value per trial")
    return np.argsort(sort_array)


def _decimate_trials(image: np.ndarray, decimate: int) -> np.ndarray:
    step = max(1, int(decimate or 1))
    return image[::step, :]


def _smooth_trials(image: np.ndarray, smooth: Any) -> np.ndarray:
    values = _numeric_values(smooth)
    if values.size == 0 or values[0] <= 1:
        return image
    window = min(int(values[0]), image.shape[0])
    kernel = np.ones(window, dtype=float) / window
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 0, image)


def _limits(value: Any) -> tuple[float, float] | None:
    values = _numeric_values(value)
    if values.size != 2 or np.all(values == 0):
        return None
    return float(values[0]), float(values[1])


def _numeric_values(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=float)
    values = np.asarray(value, dtype=float).ravel()
    return values[np.isfinite(values)]


__all__ = ["erpimage"]
