"""Morlet wavelet cycle calculator used by ``pop_newtimef``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from eegprep.functions.miscfunc.value_parsing import parse_numeric_sequence

SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))
WIDTH_COLUMNS = ("freq", "cycles", "fwhm_f", "fwhm_t", "2_sigma_f", "2_sigma_t", "sigma_f", "sigma_t")
WIDTH_UNITS = ("fwhm_t", "fwhm_f", "2_sigma_t", "2_sigma_f", "sigma_t", "sigma_f", "cycles")


@dataclass(frozen=True)
class TFCycleCalcResult:
    """Calculated cycles and equivalent temporal/spectral widths."""

    cycles: np.ndarray
    widths_table: np.ndarray
    columns: tuple[str, ...] = WIDTH_COLUMNS
    tf_data: np.ndarray | None = None
    cmorlf: np.ndarray | None = None
    figure: Any | None = None


def tf_cycle_calc(
    freqs: Any = None,
    width: Any = None,
    width_unit: str | None = None,
    *,
    log_spaced: bool = False,
    plot: bool = False,
    data: Any = None,
    fs: Any = None,
    norm: bool = False,
) -> TFCycleCalcResult:
    """Calculate Morlet wavelet cycles from temporal or spectral width."""
    freq_values = _positive_vector(freqs, "freqs")
    width_values = _width_vector(width, freq_values.size, bool(log_spaced))
    if not width_unit:
        raise ValueError("Width unit input argument is required")
    unit = str(width_unit).lower()
    if unit not in WIDTH_UNITS:
        raise ValueError("Unknown wavelet width parameter")

    if unit == "fwhm_f":
        sigma_f = width_values / SIGMA_TO_FWHM
    elif unit == "fwhm_t":
        sigma_t = width_values / SIGMA_TO_FWHM
        sigma_f = 1.0 / (2.0 * np.pi * sigma_t)
    elif unit == "2_sigma_f":
        sigma_f = width_values / 2.0
    elif unit == "2_sigma_t":
        sigma_t = width_values / 2.0
        sigma_f = 1.0 / (2.0 * np.pi * sigma_t)
    elif unit == "sigma_f":
        sigma_f = width_values
    elif unit == "sigma_t":
        sigma_f = 1.0 / (2.0 * np.pi * width_values)
    else:
        sigma_f = freq_values / width_values

    sigma_t = 1.0 / (2.0 * np.pi * sigma_f)
    fwhm_f = sigma_f * SIGMA_TO_FWHM
    fwhm_t = sigma_t * SIGMA_TO_FWHM
    cycles = freq_values / sigma_f
    widths_table = np.column_stack(
        [freq_values, cycles, fwhm_f, fwhm_t, 2.0 * sigma_f, 2.0 * sigma_t, sigma_f, sigma_t]
    )

    tf_data = None
    cmorlf = None
    if data is not None:
        if fs is None:
            raise ValueError("Requested sampling frequency input argument required")
        tf_data, cmorlf = _time_frequency_demo_transform(data, float(fs), freq_values, sigma_f, bool(norm))

    figure = _plot_widths(freq_values, widths_table, tf_data=tf_data) if plot else None
    return TFCycleCalcResult(cycles=cycles, widths_table=widths_table, tf_data=tf_data, cmorlf=cmorlf, figure=figure)


def _positive_vector(value: Any, name: str) -> np.ndarray:
    values = _numeric_vector(value)
    if values.size == 0:
        raise ValueError(f"{name} vector input argument required")
    if np.any(values <= 0):
        raise ValueError(f"{name} values must be positive")
    return values.astype(float)


def _width_vector(value: Any, count: int, log_spaced: bool) -> np.ndarray:
    values = _positive_vector(value, "width")
    if values.size == 1:
        return np.full(count, float(values[0]), dtype=float)
    if values.size == 2 and count > 2:
        if log_spaced:
            return np.logspace(np.log10(values[0]), np.log10(values[1]), count)
        return np.linspace(values[0], values[1], count)
    if values.size > 2 and values.size != count:
        raise ValueError("Length width must equal length freqs if greater than 2")
    return values.astype(float)


def _numeric_vector(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=float)
    if isinstance(value, str) and value.strip() == "":
        return np.asarray([], dtype=float)
    return np.asarray(parse_numeric_sequence(value, dtype=float), dtype=float).ravel()


def _time_frequency_demo_transform(
    data: Any, fs: float, freqs: np.ndarray, sigma_f: np.ndarray, norm: bool
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(data, dtype=float)
    if values.ndim != 1:
        raise ValueError("Data should be a vector")
    sample_count = values.size
    scaled_freqs = freqs * (sample_count / fs)
    scaled_sigma = sigma_f * (sample_count / fs)
    frequency_grid = np.tile(np.arange(sample_count, dtype=float)[:, np.newaxis], (1, freqs.size)) - scaled_freqs
    sigma_grid = np.tile(scaled_sigma, (sample_count, 1))
    if norm:
        amplitude = np.sqrt(sample_count / (np.sqrt(np.pi) * sigma_grid))
    else:
        amplitude = 1.0
    cmorlf = amplitude * np.exp(-(frequency_grid**2) / (2.0 * sigma_grid**2))
    tf_data = np.fft.ifft(np.tile(np.fft.fft(values)[:, np.newaxis], (1, freqs.size)) * cmorlf, axis=0)
    return tf_data, cmorlf.T


def _plot_widths(freqs: np.ndarray, widths_table: np.ndarray, *, tf_data: np.ndarray | None):
    figure, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes[0, 0].plot(freqs, widths_table[:, 2], ".", markersize=10)
    axes[0, 0].set_xlabel("Frequency [Hz]")
    axes[0, 0].set_ylabel("FWHM spectral bandwidth [Hz]")
    axes[0, 0].set_title("Spectral bandwidth")
    axes[0, 1].plot(freqs, widths_table[:, 3], ".", markersize=10)
    axes[0, 1].set_xlabel("Frequency [Hz]")
    axes[0, 1].set_ylabel("FWHM width [s]")
    axes[0, 1].set_title("Temporal width")
    axes[1, 0].plot(freqs, widths_table[:, 1], ".", markersize=10)
    axes[1, 0].set_xlabel("Frequency [Hz]")
    axes[1, 0].set_ylabel("Cycles")
    axes[1, 0].set_title("Cycles")
    if tf_data is not None:
        image = axes[1, 1].imshow(2.0 * np.abs(tf_data.T) ** 2, aspect="auto", origin="lower")
        figure.colorbar(image, ax=axes[1, 1])
    else:
        axes[1, 1].axis("off")
    for axis in axes.ravel()[:3]:
        axis.grid(True)
    figure.tight_layout()
    return figure


__all__ = ["TFCycleCalcResult", "tf_cycle_calc"]
