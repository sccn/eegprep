"""Shared phase-amplitude coupling helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, stats

from eegprep.functions.popfunc._pop_utils import is_on, parse_numeric_sequence
from eegprep.functions.statistics.fdr import fdr
from eegprep.functions.timefreqfunc.timefreq import timefreq


PAC_UNSUPPORTED_MESSAGE = (
    "EEGPrep implements standalone PAC computation for modulation and "
    "correlation methods. PAC bootstrap significance and EEGLAB latphase "
    "histograms require a separate tested backend and are not silently emulated."
)


@dataclass(frozen=True)
class PacResult:
    """Result returned by :func:`pac`."""

    pac: np.ndarray
    times: np.ndarray
    freqs1: np.ndarray
    freqs2: np.ndarray
    alltf_x: np.ndarray
    alltf_y: np.ndarray
    method: str

    def __iter__(self) -> Iterator[np.ndarray]:
        yield self.pac
        yield self.times
        yield self.freqs1
        yield self.freqs2


@dataclass(frozen=True)
class PacContResult:
    """Result returned by :func:`pac_cont`."""

    pac: np.ndarray
    times: np.ndarray
    pvalues: np.ndarray | None
    indices: np.ndarray
    significant: np.ndarray | None = None
    figure: Any | None = None

    def __iter__(self) -> Iterator[np.ndarray | None]:
        yield self.pac
        yield self.times
        yield self.pvalues


def compute_pac(
    X: Any,
    Y: Any,
    srate: float,
    *,
    alpha: Any = None,
    detrend: str = "off",
    freqs: Any = None,
    freqs2: Any = None,
    freqscale: str = "linear",
    itctype: str = "phasecoher",
    nfreqs: Any = None,
    method: str = "mod",
    ntimesout: Any = 200,
    timesout: Any = None,
    padratio: int = 2,
    rmerp: str = "off",
    subitc: str = "off",
    subwin: Any = None,
    tlimits: Any = None,
    title: str = "",
    vert: Any = None,
    wavelet: Any = 0,
    wavelet2: Any = None,
    winsize: Any = None,
    cycles: Any = None,
    cycles2: Any = None,
    newfig: str = "on",
    **kwargs: Any,
) -> PacResult:
    """Compute EEGLAB-style phase-amplitude coupling from epoched data."""
    _ = title, vert, newfig
    unsupported = sorted(kwargs)
    if unsupported:
        raise TypeError(f"Unsupported pac option(s): {', '.join(unsupported)}")
    if alpha is not None:
        raise NotImplementedError(PAC_UNSUPPORTED_MESSAGE)
    method_name = str(method or "mod").lower()
    if method_name == "modulation":
        method_name = "mod"
    if method_name not in {"mod", "corrsin", "corrcos", "latphase"}:
        raise ValueError("method must be 'mod', 'corrsin', 'corrcos', or 'latphase'")
    if method_name == "latphase":
        raise NotImplementedError(PAC_UNSUPPORTED_MESSAGE)

    x = _as_trial_matrix(X, name="X")
    y = _as_trial_matrix(Y, name="Y")
    if x.shape != y.shape:
        raise ValueError("X and Y must have the same time x trials shape")
    if is_on(rmerp):
        x = x - np.nanmean(x, axis=1, keepdims=True)
        y = y - np.nanmean(y, axis=1, keepdims=True)

    srate = float(srate)
    limits = _tlimits(tlimits, x.shape[0], srate)
    wavelet = cycles if cycles is not None else wavelet
    wavelet2 = cycles2 if cycles2 is not None else wavelet2
    if wavelet2 is None or _numeric_vector(wavelet2).size == 0:
        wavelet2 = wavelet
    if freqs is None:
        freqs = [0, srate / 2.0]
    if freqs2 is None or _numeric_vector(freqs2).size == 0:
        freqs2 = freqs

    decomp_x = timefreq(
        x,
        srate,
        wavelet=wavelet,
        winsize=_optional_int(winsize),
        tlimits=limits,
        detrend=detrend,
        freqs=freqs,
        nfreqs=nfreqs,
        freqscale=freqscale,
        padratio=padratio,
        itctype=itctype,
        subitc=subitc,
        timesout=timesout,
        ntimesout=ntimesout,
    )
    decomp_y = timefreq(
        y,
        srate,
        wavelet=wavelet2,
        winsize=_optional_int(winsize),
        tlimits=limits,
        detrend=detrend,
        freqs=freqs2,
        nfreqs=nfreqs,
        freqscale=freqscale,
        padratio=padratio,
        itctype=itctype,
        subitc=subitc,
        timesout=timesout,
        ntimesout=ntimesout,
    )
    alltf_x = _ensure_tf_trials(decomp_x.tfdata)
    alltf_y = _ensure_tf_trials(decomp_y.tfdata)
    times, alltf_x, alltf_y = _align_times(decomp_x.times, decomp_y.times, alltf_x, alltf_y)
    times, alltf_x, alltf_y = _apply_subwindow(times, alltf_x, alltf_y, subwin)
    values = _pac_from_tfdata(alltf_x, alltf_y, method_name)
    return PacResult(values, times, decomp_x.freqs, decomp_y.freqs, alltf_x, alltf_y, method_name)


def compute_pac_cont(
    X: Any,
    Y: Any,
    srate: float,
    *,
    alpha: Any = None,
    baseline: Any = None,
    freqphase: Any = None,
    freqamp: Any = None,
    mcorrect: str = "none",
    method: str = "modulation",
    naccu: int = 250,
    instantstat: str = "off",
    nofig: str = "off",
    statlim: str = "parametric",
    timesout: Any = None,
    filterphase: Any = None,
    filteramp: Any = None,
    ntimesout: Any = 200,
    tlimits: Any = None,
    title: str = "",
    vert: Any = None,
    winsize: Any = None,
    rng: Any = None,
    **kwargs: Any,
) -> PacContResult:
    """Compute sliding-window PAC for continuous vectors."""
    unsupported = sorted(key for key in kwargs if key not in {"filterfunc", "newfig"})
    if unsupported:
        raise TypeError(f"Unsupported pac_cont option(s): {', '.join(unsupported)}")
    x = _as_vector(X, name="X")
    y = _as_vector(Y, name="Y")
    if x.size != y.size:
        raise ValueError("X and Y must contain the same number of samples")
    srate = float(srate)
    method_name = str(method or "modulation").lower()
    if method_name == "mod":
        method_name = "modulation"
    if method_name not in {"modulation", "plv", "corr", "glm"}:
        raise ValueError("method must be 'modulation', 'plv', 'corr', or 'glm'")
    statlim_name = str(statlim or "parametric").lower()
    if statlim_name not in {"parametric", "surrogate"}:
        raise ValueError("statlim must be 'parametric' or 'surrogate'")
    if str(mcorrect or "none").lower() not in {"none", "fdr"}:
        raise ValueError("mcorrect must be 'none' or 'fdr'")

    phase_range = _frequency_range(freqphase, default=[0, srate / 2.0])
    amp_range = _frequency_range(freqamp, default=phase_range)
    x_phase = _filter_or_call(filterphase, x, srate, phase_range)
    x_amp = _filter_or_call(filteramp, y, srate, amp_range)
    phase_data = signal.hilbert(x_phase)
    amp_data = signal.hilbert(x_amp)
    phase = np.angle(phase_data)
    amplitude = np.abs(amp_data)
    z = amplitude * np.exp(1j * phase)
    window_size = _window_size(x.size, winsize)
    limits = _tlimits(tlimits, x.size, srate)
    times, indices = _pac_cont_windows(x.size, limits, window_size, timesout, ntimesout)
    values = np.zeros(indices.size, dtype=float)
    pvalues = (
        np.full(indices.size, np.nan, dtype=float) if alpha is not None or method_name in {"corr", "glm"} else None
    )
    generator = np.random.default_rng(rng)

    for out_index, center in enumerate(indices):
        span = _window_span(int(center), window_size)
        phase_epoch = x_phase[span]
        amp_epoch = x_amp[span]
        amp_hilbert_epoch = amp_data[span]
        z_epoch = z[span]
        values[out_index], pvalue = _pac_cont_value(
            method_name,
            phase_epoch,
            amp_epoch,
            z_epoch,
            amp_hilbert_epoch,
            srate,
            phase_range,
        )
        if pvalues is not None and pvalue is not None:
            pvalues[out_index] = pvalue
        if alpha is not None and is_on(instantstat) and method_name not in {"corr", "glm"}:
            surrogates = _instant_surrogates(
                method_name, phase_epoch, amp_epoch, amplitude[span], phase[span], naccu, generator
            )
            pvalues[out_index] = _surrogate_pvalue(surrogates, values[out_index], statlim_name)

    significant = None
    alpha_value = _alpha_value(alpha)
    if alpha_value is not None and not is_on(instantstat) and method_name not in {"corr", "glm"}:
        pvalues = _baseline_pvalues(values, times, baseline, statlim_name)
    if alpha_value is not None and pvalues is not None:
        significant = np.asarray(pvalues <= alpha_value, dtype=bool)
        if str(mcorrect or "none").lower() == "fdr":
            fdr_result = fdr(pvalues, alpha_value)
            significant = fdr_result.mask

    figure = None if is_on(nofig) else _plot_pac_cont(times, values, pvalues, significant, title, vert)
    return PacContResult(values, times, pvalues, indices + 1, significant=significant, figure=figure)


def raise_pac_not_implemented() -> None:
    """Raise EEGPrep's explicit unsupported-PAC limitation."""
    raise NotImplementedError(PAC_UNSUPPORTED_MESSAGE)


def _as_trial_matrix(value: Any, *, name: str) -> np.ndarray:
    values = np.asarray(value, dtype=float)
    values = np.squeeze(values)
    if values.ndim == 1:
        return values[:, np.newaxis]
    if values.ndim != 2:
        raise ValueError(f"{name} must be 1-D, 2-D, or singleton-channel 3-D data")
    if values.shape[0] == 1:
        return values.reshape(values.size, 1)
    return values


def _as_vector(value: Any, *, name: str) -> np.ndarray:
    values = np.asarray(value, dtype=float)
    if values.ndim == 3:
        raise ValueError("pac_cont cannot process 3-D input")
    values = np.squeeze(values)
    if values.ndim != 1:
        raise ValueError(f"{name} must be a vector")
    return values.astype(float)


def _ensure_tf_trials(value: np.ndarray) -> np.ndarray:
    values = np.asarray(value)
    if values.ndim == 2:
        return values[:, :, np.newaxis]
    if values.ndim != 3:
        raise ValueError("time-frequency decomposition must be frequency x time x trials")
    return values


def _align_times(
    times_x: np.ndarray, times_y: np.ndarray, alltf_x: np.ndarray, alltf_y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times_x = np.asarray(times_x, dtype=float).ravel()
    times_y = np.asarray(times_y, dtype=float).ravel()
    if times_x.shape == times_y.shape and np.allclose(times_x, times_y, rtol=0, atol=1e-9):
        return times_x, alltf_x, alltf_y
    rounded_x = np.round(times_x, 9)
    rounded_y = np.round(times_y, 9)
    common, ind_x, ind_y = np.intersect1d(rounded_x, rounded_y, return_indices=True)
    if common.size == 0:
        low = max(float(np.min(times_x)), float(np.min(times_y)))
        high = min(float(np.max(times_x)), float(np.max(times_y)))
        if low >= high:
            raise ValueError("PAC decompositions have no overlapping time points")
        count = min(times_x.size, times_y.size)
        common_times = np.linspace(low, high, count)
        return (
            common_times,
            _interp_complex_tf(alltf_x, times_x, common_times),
            _interp_complex_tf(alltf_y, times_y, common_times),
        )
    return times_x[ind_x], alltf_x[:, ind_x, :], alltf_y[:, ind_y, :]


def _interp_complex_tf(values: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    output = np.empty((values.shape[0], target_times.size, values.shape[2]), dtype=np.complex128)
    for freq_index in range(values.shape[0]):
        for trial_index in range(values.shape[2]):
            series = values[freq_index, :, trial_index]
            real = np.interp(target_times, source_times, np.real(series))
            imag = np.interp(target_times, source_times, np.imag(series))
            output[freq_index, :, trial_index] = real + 1j * imag
    return output


def _apply_subwindow(
    times: np.ndarray, alltf_x: np.ndarray, alltf_y: np.ndarray, subwin: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bounds = _numeric_vector(subwin)
    if bounds.size == 0:
        return times, alltf_x, alltf_y
    if bounds.size != 2:
        raise ValueError("subwin must contain [min max] in milliseconds")
    mask = (times > bounds[0]) & (times < bounds[1])
    if not np.any(mask):
        raise ValueError("subwin does not include any PAC time points")
    return times[mask], alltf_x[:, mask, :], alltf_y[:, mask, :]


def _pac_from_tfdata(alltf_x: np.ndarray, alltf_y: np.ndarray, method: str) -> np.ndarray:
    amplitude = np.abs(alltf_x)
    phase = np.angle(alltf_y)
    if method == "mod":
        return np.sum(amplitude[:, np.newaxis, :, :] * np.exp(1j * phase[np.newaxis, :, :, :]), axis=-1)
    transform = np.sin if method == "corrsin" else np.cos
    values = np.full((amplitude.shape[0], phase.shape[0], amplitude.shape[1]), np.nan, dtype=float)
    phase_values = transform(phase)
    for amp_freq in range(amplitude.shape[0]):
        for phase_freq in range(phase.shape[0]):
            for time_index in range(amplitude.shape[1]):
                values[amp_freq, phase_freq, time_index] = _corr(
                    phase_values[phase_freq, time_index, :], amplitude[amp_freq, time_index, :]
                )
    return values


def _pac_cont_value(
    method: str,
    phase_epoch: np.ndarray,
    amp_epoch: np.ndarray,
    z_epoch: np.ndarray,
    amp_hilbert_epoch: np.ndarray,
    srate: float,
    phase_range: np.ndarray,
) -> tuple[float, float | None]:
    if method == "modulation":
        return float(np.abs(np.mean(z_epoch))), None
    if method == "plv":
        amp_phase_signal = _bandpass(np.abs(amp_hilbert_epoch), srate, phase_range)
        amp_phase = np.angle(signal.hilbert(amp_phase_signal))
        return float(np.abs(np.mean(np.exp(1j * (phase_epoch - amp_phase))))), None
    if method == "corr":
        result = stats.pearsonr(phase_epoch, np.abs(amp_hilbert_epoch))
        return float(result.statistic), float(result.pvalue)
    result = stats.linregress(phase_epoch, np.abs(amp_epoch))
    return float(result.slope), float(result.pvalue)


def _instant_surrogates(
    method: str,
    phase_epoch: np.ndarray,
    amp_epoch: np.ndarray,
    amplitude_epoch: np.ndarray,
    phase_angle_epoch: np.ndarray,
    naccu: int,
    generator: np.random.Generator,
) -> np.ndarray:
    count = max(int(naccu), 1)
    if phase_epoch.size < 4:
        raise ValueError("PAC surrogate statistics require at least four samples per window")
    values = np.zeros(count, dtype=float)
    for index in range(count):
        shift = int(generator.integers(1, phase_epoch.size))
        shifted_amp = np.roll(amplitude_epoch, shift)
        if method == "modulation":
            values[index] = np.abs(np.mean(shifted_amp * np.exp(1j * phase_angle_epoch)))
        else:
            shifted_phase = np.angle(signal.hilbert(np.roll(amp_epoch, shift)))
            values[index] = np.abs(np.mean(np.exp(1j * (phase_epoch - shifted_phase))))
    return values


def _baseline_pvalues(values: np.ndarray, times: np.ndarray, baseline: Any, statlim: str) -> np.ndarray:
    bounds = _numeric_vector(baseline)
    if bounds.size == 0:
        baseline_values = np.abs(values)
    else:
        if bounds.size != 2:
            raise ValueError("baseline must contain [min max] in milliseconds")
        mask = (times >= bounds[0]) & (times <= bounds[1])
        if not np.any(mask):
            raise ValueError("baseline does not include any PAC time points")
        baseline_values = np.abs(values[mask])
    if statlim == "surrogate":
        return np.asarray([_empirical_pvalue(baseline_values, value) for value in np.abs(values)], dtype=float)
    mean = float(np.nanmean(baseline_values))
    std = float(np.nanstd(baseline_values, ddof=1))
    if not np.isfinite(std) or std == 0:
        return np.where(np.abs(values) <= mean, 1.0, 0.0)
    zscores = (np.abs(values) - mean) / std
    return stats.norm.sf(zscores)


def _surrogate_pvalue(surrogates: np.ndarray, observed: float, statlim: str) -> float:
    if statlim == "surrogate":
        return _empirical_pvalue(surrogates, observed)
    mean = float(np.nanmean(surrogates))
    std = float(np.nanstd(surrogates, ddof=1))
    if not np.isfinite(std) or std == 0:
        return 1.0 if abs(observed) <= mean else 0.0
    return float(stats.norm.sf((abs(observed) - mean) / std))


def _empirical_pvalue(distribution: np.ndarray, observed: float) -> float:
    values = np.asarray(distribution, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan
    return float((np.count_nonzero(values >= abs(observed)) + 1) / (values.size + 1))


def _filter_or_call(callback: Any, values: np.ndarray, srate: float, freq_range: np.ndarray) -> np.ndarray:
    if callback is None or callback == "":
        return _bandpass(values, srate, freq_range)
    if not callable(callback):
        raise TypeError("filterphase and filteramp must be callables when provided")
    filtered = callback(values)
    return np.asarray(filtered, dtype=float).ravel()


def _bandpass(values: np.ndarray, srate: float, freq_range: np.ndarray, order: int = 4) -> np.ndarray:
    data = np.asarray(values, dtype=float).ravel()
    nyquist = srate / 2.0
    low = max(float(freq_range[0]), 0.0)
    high = min(float(freq_range[-1]), nyquist)
    if low <= 0 and high >= nyquist:
        return data.copy()
    if high <= 0 or low >= nyquist or high <= low:
        raise ValueError("frequency range must overlap 0..Nyquist")
    if low <= 0:
        sos = signal.butter(order, high / nyquist, btype="lowpass", output="sos")
    elif high >= nyquist:
        sos = signal.butter(order, low / nyquist, btype="highpass", output="sos")
    else:
        sos = signal.butter(order, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    padlen = min(max(0, data.size - 1), 3 * (2 * sos.shape[0] + 1))
    if padlen < 1:
        return signal.sosfilt(sos, data)
    return signal.sosfiltfilt(sos, data, padlen=padlen)


def _pac_cont_windows(
    frames: int, limits: np.ndarray, winsize: int, timesout: Any, ntimesout: Any
) -> tuple[np.ndarray, np.ndarray]:
    time_vector = np.linspace(float(limits[0]), float(limits[1]), frames)
    left = winsize // 2
    right = winsize - left
    valid = np.arange(left, frames - right + 1, dtype=int)
    if valid.size == 0:
        raise ValueError("winsize is too large for the data length")
    requested_times = _numeric_vector(timesout)
    if requested_times.size:
        indices = []
        for value in requested_times:
            index = int(np.argmin(np.abs(time_vector - value)))
            if left <= index <= frames - right:
                indices.append(index)
        if not indices:
            raise ValueError("timesout does not include any full PAC windows")
        indices = np.asarray(sorted(set(indices)), dtype=int)
        return time_vector[indices], indices
    count_values = _numeric_vector(ntimesout, dtype=int)
    count = int(count_values[0]) if count_values.size else 200
    if count < 0:
        step = abs(count)
        indices = valid[::step]
    else:
        count = min(max(count, 1), valid.size)
        positions = np.linspace(0, valid.size - 1, count)
        indices = valid[np.unique(np.rint(positions).astype(int))]
    return time_vector[indices], indices


def _window_span(center: int, winsize: int) -> slice:
    left = winsize // 2
    right = winsize - left
    return slice(center - left, center + right)


def _plot_pac_cont(
    times: np.ndarray,
    values: np.ndarray,
    pvalues: np.ndarray | None,
    significant: np.ndarray | None,
    title: str,
    vert: Any,
) -> Any:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(times, values, color="#2563eb", linewidth=1.5)
    if pvalues is not None and significant is not None and np.any(significant):
        ax.scatter(times[significant], values[significant], color="#dc2626", s=18, zorder=3)
    markers = _numeric_vector(vert)
    for marker in markers:
        ax.axvline(float(marker), color="#16a34a", linewidth=0.9, alpha=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("PAC")
    ax.set_title(str(title or "Phase-amplitude coupling"))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or right.size < 2:
        return np.nan
    if np.nanstd(left) == 0 or np.nanstd(right) == 0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def _window_size(frames: int, explicit: Any) -> int:
    values = _numeric_vector(explicit, dtype=int)
    if values.size:
        winsize = int(values[0])
    else:
        winsize = max(int(2 ** (np.ceil(np.log2(max(frames, 2))) - 3)), 4)
    if winsize < 2:
        raise ValueError("winsize must be at least 2")
    if winsize > frames:
        raise ValueError("winsize must be less than or equal to the data length")
    return winsize


def _frequency_range(value: Any, *, default: Any) -> np.ndarray:
    values = _numeric_vector(value)
    if values.size == 0:
        values = _numeric_vector(default)
    if values.size == 1:
        return np.asarray([values[0], values[0]], dtype=float)
    if values.size < 2:
        raise ValueError("frequency range must contain at least one frequency")
    return np.asarray([values[0], values[-1]], dtype=float)


def _tlimits(value: Any, frames: int, srate: float) -> np.ndarray:
    values = _numeric_vector(value)
    if values.size == 2:
        return values.astype(float)
    return np.asarray([0.0, (frames - 1) / float(srate) * 1000.0], dtype=float)


def _optional_int(value: Any) -> int | None:
    values = _numeric_vector(value, dtype=int)
    if values.size == 0:
        return None
    return int(values[0])


def _alpha_value(value: Any) -> float | None:
    values = _numeric_vector(value)
    if values.size == 0:
        return None
    alpha = float(values[0])
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be between 0 and 1")
    return alpha


def _numeric_vector(value: Any, dtype: Any = float) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=dtype)
    if isinstance(value, str) and value.strip() == "":
        return np.asarray([], dtype=dtype)
    return np.asarray(parse_numeric_sequence(value, dtype=dtype), dtype=dtype).ravel()


__all__ = [
    "PAC_UNSUPPORTED_MESSAGE",
    "PacContResult",
    "PacResult",
    "compute_pac",
    "compute_pac_cont",
    "raise_pac_not_implemented",
]
