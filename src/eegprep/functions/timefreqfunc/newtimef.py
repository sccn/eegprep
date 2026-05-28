"""Deterministic EEGLAB-style ``newtimef`` numerical core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


@dataclass(frozen=True)
class TimeFrequencyResult:
    """Computed event-related spectral perturbation and ITC arrays."""

    ersp: np.ndarray
    itc: np.ndarray
    powbase: np.ndarray
    times: np.ndarray
    freqs: np.ndarray
    tfdata: np.ndarray
    figure: Any | None = None


def newtimef(
    data: Any,
    frames: int,
    tlimits: Any,
    srate: float,
    cycles: Any = 0,
    **kwargs: Any,
) -> TimeFrequencyResult:
    """Compute an EEGLAB-like ERSP/ITC time-frequency decomposition.

    This standalone core intentionally implements the deterministic STFT path
    used by EEGPrep's plot wrappers. EEGLAB's full ``newtimef`` also includes
    multiple wavelet implementations, bootstrapping, and time-warping; those are
    outside this stable numerical core and are rejected clearly by the wrappers.
    """
    epochs = _as_epochs(data, int(frames))
    frames = epochs.shape[0]
    srate = float(srate)
    tlimits = _limits(tlimits, frames, srate)
    cycles_array = _numeric_vector(cycles)
    winsize = _winsize(frames, srate, cycles_array, kwargs.get("winsize"), kwargs.get("freqs"))
    padratio = int(_first_numeric(kwargs.get("padratio"), 2))
    nfft = max(winsize, int(2 ** np.ceil(np.log2(max(winsize, 1)))) * max(padratio, 1))
    noverlap = min(winsize - 1, max(0, int(_first_numeric(kwargs.get("overlap"), winsize // 2))))

    freqs, times_seconds, tfdata = _trial_stft(epochs, srate, winsize, noverlap, nfft)
    freqs, tfdata = _select_freqs(freqs, tfdata, kwargs.get("freqs"), kwargs.get("nfreqs"), kwargs.get("freqscale"))
    times = tlimits[0] + times_seconds * 1000.0
    times, tfdata = _select_times(times, tfdata, kwargs.get("timesout"))

    power = np.abs(tfdata) ** 2
    powbase = _baseline_power(power, times, kwargs.get("baseline", 0))
    scale = str(kwargs.get("scale", "log")).lower()
    if scale == "abs":
        ersp = np.nanmean(power, axis=2) / powbase[:, np.newaxis]
    elif scale == "log":
        ersp = 10.0 * np.log10(np.maximum(np.nanmean(power, axis=2), np.finfo(float).tiny) / powbase[:, np.newaxis])
    else:
        raise ValueError("scale must be 'log' or 'abs'")
    phase = np.divide(tfdata, np.maximum(np.abs(tfdata), np.finfo(float).tiny))
    itc = np.nanmean(phase, axis=2)

    figure = None
    if _is_on(kwargs.get("plot", "on")):
        figure = _plot_time_frequency(
            ersp,
            np.abs(itc),
            times,
            freqs,
            title=str(kwargs.get("title", "Time-frequency")),
            plotersp=_is_on(kwargs.get("plotersp", "on")),
            plotitc=_is_on(kwargs.get("plotitc", "on")),
        )
    return TimeFrequencyResult(ersp, itc, powbase, times, freqs, tfdata, figure)


def _as_epochs(data: Any, frames: int) -> np.ndarray:
    values = np.asarray(data, dtype=float)
    if values.ndim == 3:
        if values.shape[0] != 1:
            raise ValueError("newtimef expects a single channel/component")
        return values[0].reshape(values.shape[1], values.shape[2])
    if values.ndim == 2:
        if values.shape[0] == frames:
            return values
        if values.shape[1] == frames:
            return values.T
        if values.shape[0] == 1:
            values = values.ravel()
        else:
            raise ValueError("2-D data must be frames x trials")
    if values.ndim == 1:
        if frames <= 0 or values.size % frames:
            raise ValueError("1-D data length must be a multiple of frames")
        return values.reshape(-1, frames).T
    raise ValueError("newtimef data must be 1-D, 2-D, or 3-D")


def _trial_stft(
    epochs: np.ndarray, srate: float, winsize: int, noverlap: int, nfft: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trial_spectra = []
    freqs = np.asarray([])
    times = np.asarray([])
    for trial in range(epochs.shape[1]):
        freqs, times, spectrum = signal.stft(
            epochs[:, trial],
            fs=srate,
            window="hann",
            nperseg=winsize,
            noverlap=noverlap,
            nfft=nfft,
            detrend=False,
            boundary=None,
            padded=False,
        )
        trial_spectra.append(spectrum)
    tfdata = np.stack(trial_spectra, axis=2)
    return freqs, times, tfdata


def _winsize(frames: int, srate: float, cycles: np.ndarray, explicit: Any, freqs: Any) -> int:
    if explicit is not None and not _empty(explicit):
        return _bounded_window(int(_first_numeric(explicit, frames)), frames)
    if cycles.size and cycles[0] > 0:
        freq_values = _numeric_vector(freqs)
        fmin = float(np.nanmin(freq_values)) if freq_values.size else max(1.0, srate / max(frames, 1))
        return _bounded_window(int(round(cycles[0] * srate / max(fmin, np.finfo(float).eps))), frames)
    default = min(frames, max(8, int(2 ** np.floor(np.log2(max(frames / 8, 1))))))
    return _bounded_window(default, frames)


def _bounded_window(value: int, frames: int) -> int:
    return max(2, min(int(value), int(frames)))


def _select_freqs(
    freqs: np.ndarray, tfdata: np.ndarray, requested: Any, nfreqs: Any, freqscale: Any
) -> tuple[np.ndarray, np.ndarray]:
    freq_values = _numeric_vector(requested)
    if freq_values.size == 1:
        target = freq_values
    elif freq_values.size == 2:
        mask = (freqs >= freq_values[0]) & (freqs <= freq_values[1])
        target = freqs[mask]
    elif freq_values.size > 2:
        target = freq_values
    else:
        target = freqs[freqs > 0]
        if target.size == 0:
            target = freqs
        target = target[target <= min(50.0, freqs[-1])]
    if target.size == 0:
        target = freqs
    count_values = _numeric_vector(nfreqs, dtype=int)
    if count_values.size and count_values[0] > 0 and target.size > int(count_values[0]):
        count = int(count_values[0])
        positive = target[target > 0]
        if str(freqscale).lower() == "log" and positive.size:
            grid = np.geomspace(np.nanmin(positive), np.nanmax(target), count)
        else:
            grid = np.linspace(float(target[0]), float(target[-1]), count)
        target = grid
    indices = np.unique([int(np.argmin(np.abs(freqs - value))) for value in target])
    return freqs[indices], tfdata[indices, :, :]


def _select_times(times: np.ndarray, tfdata: np.ndarray, timesout: Any) -> tuple[np.ndarray, np.ndarray]:
    values = _numeric_vector(timesout, dtype=float)
    if values.size == 0:
        return times, tfdata
    if values.size == 1 and values[0] > 0:
        count = min(int(values[0]), times.size)
        indices = np.unique(np.linspace(0, times.size - 1, count).round().astype(int))
    elif values.size == 1 and values[0] < 0:
        step = max(1, int(abs(values[0])))
        indices = np.arange(0, times.size, step)
    else:
        indices = np.unique([int(np.argmin(np.abs(times - value))) for value in values])
    return times[indices], tfdata[:, indices, :]


def _baseline_power(power: np.ndarray, times: np.ndarray, baseline: Any) -> np.ndarray:
    values = _numeric_vector(baseline)
    if values.size and np.any(np.isnan(values)):
        return np.ones(power.shape[0], dtype=float)
    if values.size == 0:
        mask = times < 0
    elif values.size == 1:
        mask = times <= values[0]
    elif values.size == 2:
        mask = (times >= values[0]) & (times <= values[1])
    else:
        raise ValueError("baseline must be empty, scalar, [min max], or NaN")
    if not np.any(mask):
        baseline_power = np.nanmean(power, axis=(1, 2))
    else:
        baseline_power = np.nanmean(power[:, mask, :], axis=(1, 2))
    return np.maximum(baseline_power, np.finfo(float).tiny)


def _plot_time_frequency(
    ersp: np.ndarray,
    itc: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    *,
    title: str,
    plotersp: bool,
    plotitc: bool,
):
    panels = int(plotersp) + int(plotitc)
    if panels == 0:
        return None
    fig, axes = plt.subplots(panels, 1, figsize=(7.5, 5.0), squeeze=False)
    row = 0
    if plotersp:
        image = axes[row, 0].imshow(
            ersp,
            aspect="auto",
            origin="lower",
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            interpolation="nearest",
        )
        axes[row, 0].set_title(title)
        axes[row, 0].set_ylabel("Frequency (Hz)")
        fig.colorbar(image, ax=axes[row, 0], label="ERSP")
        row += 1
    if plotitc:
        image = axes[row, 0].imshow(
            itc,
            aspect="auto",
            origin="lower",
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
            interpolation="nearest",
            vmin=0,
            vmax=max(1.0, float(np.nanmax(itc))),
        )
        axes[row, 0].set_ylabel("Frequency (Hz)")
        axes[row, 0].set_xlabel("Time (ms)")
        fig.colorbar(image, ax=axes[row, 0], label="ITC")
    fig.tight_layout()
    return fig


def _limits(value: Any, frames: int, srate: float) -> np.ndarray:
    values = _numeric_vector(value)
    if values.size == 2:
        return values.astype(float)
    duration = (frames - 1) / float(srate) * 1000.0
    return np.asarray([0.0, duration], dtype=float)


def _numeric_vector(value: Any, *, dtype: Any = float) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=dtype)
    if isinstance(value, np.ndarray):
        return value.astype(dtype).ravel()
    if isinstance(value, (int, float, np.integer, np.floating)):
        return np.asarray([value], dtype=dtype)
    if isinstance(value, str):
        text = value.strip().strip("[]")
        if not text:
            return np.asarray([], dtype=dtype)
        values = []
        for token in text.replace(",", " ").split():
            if ":" in token:
                values.extend(_colon_sequence(token))
            else:
                values.append(float(token))
        return np.asarray(values, dtype=dtype)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=dtype).ravel()
    return np.asarray([value], dtype=dtype)


def _first_numeric(value: Any, default: float) -> float:
    values = _numeric_vector(value)
    return float(values[0]) if values.size else float(default)


def _colon_sequence(token: str) -> list[float]:
    pieces = token.split(":")
    if len(pieces) not in {2, 3}:
        raise ValueError(f"Invalid colon range: {token}")
    start = float(pieces[0])
    if len(pieces) == 2:
        stop = float(pieces[1])
        step = 1.0 if stop >= start else -1.0
    else:
        step = float(pieces[1])
        stop = float(pieces[2])
    if step == 0 or (stop - start) * step < 0:
        return []
    count = int(np.floor((stop - start) / step + 1e-9)) + 1
    values = [float(start + index * step) for index in range(max(count, 0))]
    if values and np.isclose(values[-1], stop, rtol=0.0, atol=max(abs(step), 1.0) * 1e-12):
        values[-1] = stop
    return values


def _empty(value: Any) -> bool:
    return (
        value is None
        or (isinstance(value, str) and not value.strip())
        or (isinstance(value, (list, tuple, np.ndarray)) and np.asarray(value).size == 0)
    )


def _is_on(value: Any) -> bool:
    return str(value).lower() not in {"0", "false", "off", "no", "none"}


__all__ = ["TimeFrequencyResult", "newtimef"]
