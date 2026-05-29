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
    *,
    winsize: Any = None,
    freqs: Any = None,
    nfreqs: Any = None,
    freqscale: Any = None,
    timesout: Any = None,
    padratio: Any = None,
    overlap: Any = None,
    baseline: Any = 0,
    scale: str = "log",
    plot: Any = "on",
    plotersp: Any = "on",
    plotitc: Any = "on",
    title: str = "Time-frequency",
) -> TimeFrequencyResult:
    """Compute an EEGLAB-like ERSP/ITC time-frequency decomposition."""
    freqs, times, tfdata = compute_time_frequency(
        data,
        frames,
        tlimits,
        srate,
        cycles,
        winsize=winsize,
        freqs=freqs,
        nfreqs=nfreqs,
        freqscale=freqscale,
        timesout=timesout,
        padratio=padratio,
        overlap=overlap,
    )

    power = np.abs(tfdata) ** 2
    powbase = _baseline_power(power, times, baseline)
    scale_mode = str(scale).lower()
    if scale_mode == "abs":
        ersp = np.nanmean(power, axis=2) / powbase[:, np.newaxis]
    elif scale_mode == "log":
        ersp = 10.0 * np.log10(np.maximum(np.nanmean(power, axis=2), np.finfo(float).tiny) / powbase[:, np.newaxis])
    else:
        raise ValueError("scale must be 'log' or 'abs'")
    phase = np.divide(tfdata, np.maximum(np.abs(tfdata), np.finfo(float).tiny))
    itc = np.nanmean(phase, axis=2)

    figure = None
    if _is_on(plot):
        figure = _plot_time_frequency(
            ersp,
            np.abs(itc),
            times,
            freqs,
            title=str(title),
            plotersp=_is_on(plotersp),
            plotitc=_is_on(plotitc),
        )
    return TimeFrequencyResult(ersp, itc, powbase, times, freqs, tfdata, figure)


def compute_time_frequency(
    data: Any,
    frames: int,
    tlimits: Any,
    srate: float,
    cycles: Any = 0,
    *,
    winsize: Any = None,
    freqs: Any = None,
    nfreqs: Any = None,
    freqscale: Any = None,
    timesout: Any = None,
    padratio: Any = None,
    overlap: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(freqs, times_ms, tfdata)`` for one signal using EEGLAB-like defaults."""
    epochs = _as_epochs(data, int(frames))
    frames = epochs.shape[0]
    srate = float(srate)
    tlimits = _limits(tlimits, frames, srate)
    cycles_array = _numeric_vector(cycles)
    if cycles_array.size and cycles_array[0] > 0:
        return _trial_wavelet(
            epochs,
            srate,
            tlimits,
            cycles_array,
            winsize,
            freqs,
            nfreqs,
            freqscale,
            timesout,
            padratio,
        )
    winsize = _winsize(frames, srate, cycles_array, winsize, freqs)
    padratio_value = int(_first_numeric(padratio, 2))
    nfft = max(winsize, int(2 ** np.ceil(np.log2(max(winsize, 1)))) * max(padratio_value, 1))
    noverlap = min(winsize - 1, max(0, int(_first_numeric(overlap, winsize // 2))))
    stft_freqs, times_seconds, tfdata = _trial_stft(epochs, srate, winsize, noverlap, nfft)
    selected_freqs, tfdata = _select_freqs(stft_freqs, tfdata, freqs, nfreqs, freqscale)
    times = tlimits[0] + times_seconds * 1000.0
    times, tfdata = _select_times(times, tfdata, timesout)
    return selected_freqs, times, tfdata


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


def _trial_wavelet(
    epochs: np.ndarray,
    srate: float,
    tlimits: np.ndarray,
    cycles: np.ndarray,
    explicit_winsize: Any,
    requested_freqs: Any,
    nfreqs: Any,
    freqscale: Any,
    timesout: Any,
    padratio: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    winsize = _winsize(epochs.shape[0], srate, np.asarray([0.0]), explicit_winsize, None)
    freqs = _wavelet_freqs(srate, epochs.shape[0], winsize, cycles, requested_freqs, nfreqs, freqscale, padratio)
    wavelets = _morlet_wavelets(freqs, _wavelet_cycles(freqs, cycles), srate)
    max_winsize = max(len(wavelet) for wavelet in wavelets)
    times, indices = _time_output_indices(epochs.shape[0], tlimits, max_winsize, timesout)
    tfdata = np.empty((len(freqs), len(indices), epochs.shape[1]), dtype=complex)
    for trial in range(epochs.shape[1]):
        signal_values = epochs[:, trial]
        for freq_index, wavelet in enumerate(wavelets):
            half_width = len(wavelet) // 2
            for time_index, center in enumerate(indices):
                segment = signal_values[center - half_width : center + half_width + 1]
                segment = segment - np.nanmean(segment)
                tfdata[freq_index, time_index, trial] = np.sum(wavelet * segment)
    return freqs, times, tfdata


def _winsize(frames: int, srate: float, cycles: np.ndarray, explicit: Any, freqs: Any) -> int:
    if explicit is not None and not _empty(explicit):
        return _bounded_window(int(_first_numeric(explicit, frames)), frames)
    if cycles.size and cycles[0] > 0 and not _empty(freqs):
        freq_values = _numeric_vector(freqs)
        fmin = float(np.nanmin(freq_values)) if freq_values.size else max(1.0, srate / max(frames, 1))
        return _bounded_window(int(round(cycles[0] * srate / max(fmin, np.finfo(float).eps))), frames)
    default = min(frames, max(4, int(2 ** max(np.ceil(np.log2(max(frames, 2))) - 3, 2))))
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


def _wavelet_freqs(
    srate: float,
    frames: int,
    winsize: int,
    cycles: np.ndarray,
    requested: Any,
    nfreqs: Any,
    freqscale: Any,
    padratio: Any,
) -> np.ndarray:
    freq_values = _numeric_vector(requested)
    if freq_values.size > 2:
        return freq_values.astype(float)
    min_freq = max(float(cycles[0]) * float(srate) / max(winsize, 1), float(srate) / max(frames, 1))
    max_freq = min(50.0, float(srate) / 2.0)
    if freq_values.size == 1:
        return freq_values.astype(float)
    if freq_values.size == 2:
        min_freq = float(freq_values[0])
        max_freq = float(freq_values[1])
    if min_freq > max_freq:
        min_freq = max_freq
    count_values = _numeric_vector(nfreqs, dtype=int)
    if count_values.size and count_values[0] > 0:
        count = int(count_values[0])
    else:
        count = max(1, int(round(winsize / 2 * max(1, int(_first_numeric(padratio, 2))))))
    if count == 1:
        return np.asarray([min_freq], dtype=float)
    if str(freqscale).lower() == "log" and min_freq > 0 and max_freq > 0:
        return np.geomspace(min_freq, max_freq, count)
    return np.linspace(min_freq, max_freq, count)


def _wavelet_cycles(freqs: np.ndarray, cycles: np.ndarray) -> np.ndarray:
    if cycles.size == 1:
        return np.full(freqs.shape, float(cycles[0]))
    if cycles.size == 2:
        high_cycles = float(cycles[1])
        if high_cycles < 1 and freqs.size and freqs[0] > 0:
            high_cycles = float(cycles[0]) * float(freqs[-1]) / float(freqs[0]) * (1.0 - high_cycles)
        return np.linspace(float(cycles[0]), high_cycles, freqs.size)
    if cycles.size != freqs.size:
        raise ValueError("cycles must be scalar, length two, or match the number of frequencies")
    return cycles.astype(float)


def _morlet_wavelets(freqs: np.ndarray, cycles: np.ndarray, srate: float) -> list[np.ndarray]:
    wavelets = []
    for freq, cycle_count in zip(freqs, cycles):
        normalized_freq = float(freq) / float(srate)
        sigma_freq = normalized_freq / max(float(cycle_count), np.finfo(float).eps)
        sigma_time = 1.0 / (2.0 * np.pi * sigma_freq)
        half_width = int(np.floor(sigma_time * 7.0 / 2.0))
        samples = np.arange(0, half_width * 2 + 1, dtype=float) - half_width
        amplitude = 1.0 / np.sqrt(sigma_time * np.sqrt(np.pi))
        wavelet = (
            amplitude * np.exp(-(samples**2) / (2.0 * sigma_time**2)) * np.exp(2j * np.pi * normalized_freq * samples)
        )
        wavelets.append(wavelet)
    return wavelets


def _time_output_indices(
    frames: int, tlimits: np.ndarray, winsize: int, timesout: Any
) -> tuple[np.ndarray, np.ndarray]:
    full_times = np.linspace(float(tlimits[0]), float(tlimits[1]), int(frames))
    half_width = int(winsize) // 2
    start = half_width
    stop = int(frames) - half_width - 1
    if stop < start:
        raise ValueError("Not enough data points, reduce the window size or lowest frequency")
    requested = _numeric_vector(timesout)
    if requested.size == 0:
        count = 200
        indices = np.linspace(start, stop, min(count, stop - start + 1)).round().astype(int)
    elif requested.size == 1 and requested[0] > 0:
        count = min(int(requested[0]), stop - start + 1)
        indices = np.linspace(start, stop, count).round().astype(int)
    elif requested.size == 1 and requested[0] < 0:
        step = max(1, int(abs(requested[0])))
        indices = np.arange(start, stop + 1, step)
    else:
        indices = np.asarray([int(np.argmin(np.abs(full_times - value))) for value in requested], dtype=int)
        indices = indices[(indices >= start) & (indices <= stop)]
        if indices.size == 0:
            raise ValueError("No time points. Reduce time window or minimum frequency.")
    indices = np.unique(indices)
    return full_times[indices], indices


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


__all__ = ["TimeFrequencyResult", "compute_time_frequency", "newtimef"]
