"""Signal/event statistics helper matching EEGLAB ``signalstat`` basics."""

from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from eegprep.functions.sigprocfunc.topoplot import topoplot


@dataclass(frozen=True)
class SignalStatResult:
    """Statistics returned by ``signalstat``."""

    mean: float
    std: float
    skewness: float
    kurtosis: float
    median: float
    zlow: float
    zhigh: float
    trimmed_mean: float
    trimmed_std: float
    trimmed_indices: np.ndarray
    kstest_h: int
    figure: Any | None = None

    def matlab_tuple(self) -> tuple[Any, ...]:
        """Return the EEGLAB output tuple order."""
        return (
            self.mean,
            self.std,
            self.skewness,
            self.kurtosis,
            self.median,
            self.zlow,
            self.zhigh,
            self.trimmed_mean,
            self.trimmed_std,
            self.trimmed_indices,
            self.kstest_h,
        )


def signalstat(
    data: Any,
    plotlab: int = 1,
    dlabel: str | None = None,
    percent: float = 5,
    dlabel2: str = "",
    map: Any = None,
    chan_locs: Any = None,
) -> SignalStatResult:
    """Compute EEGLAB-style summary statistics for a real-valued signal."""
    values = np.asarray(data, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("signalstat requires at least one finite data value")
    if percent < 0 or percent > 100:
        raise ValueError("signalstat percent must be between 0 and 100")
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    skewness = float(stats.skew(values, bias=False)) if values.size > 2 and std else 0.0
    kurtosis = float(stats.kurtosis(values, fisher=True, bias=False)) if values.size > 3 and std else 0.0
    median = float(np.median(values))
    trim_fraction = float(percent) / 100.0 / 2.0
    zlow = float(_matlab_quantile(values, trim_fraction))
    zhigh = float(_matlab_quantile(values, 1.0 - trim_fraction))
    trimmed_indices = np.flatnonzero((values >= zlow) & (values <= zhigh))
    trimmed = values[trimmed_indices]
    trimmed_mean = float(np.mean(trimmed)) if trimmed.size else float("nan")
    trimmed_std = float(np.std(trimmed, ddof=1)) if trimmed.size > 1 else 0.0
    kstest_h = _kstest_h(values, mean, std)
    figure = None
    if int(plotlab) == 1:
        figure = _plot_signalstat(values, mean, std, dlabel or "Potential [V]", dlabel2, map, chan_locs)
    elif int(plotlab) != 0:
        raise ValueError("signalstat plotlab must be 0 or 1")
    return SignalStatResult(
        mean,
        std,
        skewness,
        kurtosis,
        median,
        zlow,
        zhigh,
        trimmed_mean,
        trimmed_std,
        trimmed_indices,
        kstest_h,
        figure,
    )


def _kstest_h(values: np.ndarray, mean: float, std: float) -> int:
    if values.size < 2 or std <= 0:
        return -1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _statistic, pvalue = stats.kstest(values, "norm", args=(mean, std))
    return int(pvalue < 0.05)


def _plot_signalstat(
    values: np.ndarray,
    mean: float,
    std: float,
    dlabel: str,
    dlabel2: str,
    map_values: Any,
    chan_locs: Any,
):
    has_map = _has_topomap(map_values, chan_locs)
    fig, axes = plt.subplots(2, 3 if has_map else 2, figsize=(9.5 if has_map else 8.0, 6.0))
    axes = axes.ravel()
    axes[0].hist(values, bins=max(10, min(80, round(values.size / 100))), color=(0.56, 0.66, 0.9), edgecolor="black")
    axes[0].set_title("Data histogram")
    axes[0].set_xlabel(dlabel)
    axes[0].set_ylabel("Count")
    if std > 0:
        xs = np.linspace(float(np.min(values)), float(np.max(values)), 200)
        density = stats.norm.pdf(xs, mean, std) * values.size * (xs[1] - xs[0])
        axes[0].plot(xs, density, color="black")
    axes[1].boxplot(values)
    axes[1].set_title("Boxplot")
    stats.probplot(values, dist="norm", plot=axes[2])
    axes[2].set_title("QQ plot")
    if has_map:
        _plot_stat_topomap(axes[3], map_values, chan_locs)
        text_axis = axes[4]
        axes[5].axis("off")
    else:
        text_axis = axes[3]
    text_axis.axis("off")
    text = (
        f"Mean: {mean:.6g}\n"
        f"Std: {std:.6g}\n"
        f"Median: {np.median(values):.6g}\n"
        f"Skewness: {stats.skew(values, bias=False) if values.size > 2 and std else 0:.6g}\n"
        f"Kurtosis: {stats.kurtosis(values, fisher=True, bias=False) if values.size > 3 and std else 0:.6g}"
    )
    text_axis.text(0.02, 0.98, text, va="top", family="monospace")
    if dlabel2:
        fig.suptitle(dlabel2)
    fig.tight_layout()
    return fig


def _has_topomap(map_values: Any, chan_locs: Any) -> bool:
    if chan_locs is None:
        return False
    return (
        np.asarray([] if map_values is None else map_values).size > 0
        and len(np.asarray(chan_locs, dtype=object).ravel()) > 0
    )


def _plot_stat_topomap(axis: Any, map_values: Any, chan_locs: Any) -> None:
    values = np.asarray(map_values, dtype=float).ravel()
    if values.size == 1:
        topoplot([], chan_locs, style="blank", electrodes="off", axes=axis)
        axis.set_title(f"Channel {int(values[0])}")
        return
    topoplot(values, chan_locs, electrodes="off", axes=axis, colorbar=False)
    axis.set_title("Topographic map")


def _matlab_quantile(values: np.ndarray, probability: float) -> float:
    sorted_values = np.sort(np.asarray(values, dtype=float).ravel())
    if sorted_values.size == 0:
        return float("nan")
    position = sorted_values.size * float(probability) + 0.5
    if position <= 1:
        return float(sorted_values[0])
    if position >= sorted_values.size:
        return float(sorted_values[-1])
    lower = int(np.floor(position))
    fraction = position - lower
    return float(sorted_values[lower - 1] + fraction * (sorted_values[lower] - sorted_values[lower - 1]))


__all__ = ["SignalStatResult", "signalstat"]
