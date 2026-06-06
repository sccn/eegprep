"""Compute-only EEGLAB-style time-frequency decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal

from eegprep.functions.timefreqfunc.dftfilt2 import dftfilt2
from eegprep.functions.timefreqfunc.dftfilt3 import dftfilt3, symmetric_hanning
from eegprep.functions.timefreqfunc.angtimewarp import angtimewarp
from eegprep.functions.timefreqfunc.newtimefitc import newtimefitc
from eegprep.functions.timefreqfunc.timewarp import timewarp


@dataclass(frozen=True)
class TimeFrequencyDecomposition:
    """Computed single-trial spectral estimates."""

    tfdata: np.ndarray
    freqs: np.ndarray
    times: np.ndarray
    itcvals: np.ndarray | None
    winsize: int
    cycles: np.ndarray


def timefreq(
    data: Any,
    srate: float,
    *,
    frames: int | None = None,
    cycles: Any = 0,
    wavelet: Any = 0,
    winsize: int | None = None,
    tlimits: Any = None,
    detrend: str = "off",
    causal: str = "off",
    freqs: Any = None,
    nfreqs: Any = None,
    freqscale: str = "linear",
    ffttaper: str = "hanning",
    padratio: int = 2,
    itctype: str = "phasecoher",
    subitc: str = "off",
    timesout: Any = None,
    ntimesout: Any = None,
    timestretch: Any = None,
    wletmethod: str = "dftfilt3",
    verbose: str = "off",
) -> TimeFrequencyDecomposition:
    """Return EEGLAB-like spectral estimates for each trial.

    Args mirror EEGLAB's ``timefreq`` where practical. The optional ``frames``
    keyword is an EEGPrep convenience for reshaping flat trial-concatenated
    input before applying the EEGLAB algorithm.
    """
    _ = verbose
    data_3d = _as_channel_epoch_data(data, frames=frames)
    _, frame_count, trial_count = data_3d.shape
    srate = float(srate)
    cycle_values = _cycles(cycles, wavelet)
    window_size = _window_size(frame_count, cycle_values, winsize)
    pad_ratio = _padratio(padratio)
    limits = _tlimits(tlimits, frame_count, srate)
    requested_freqs, fft_freqs = _requested_freqs(
        srate,
        window_size,
        cycle_values,
        freqs,
        nfreqs,
        str(freqscale).lower(),
        pad_ratio,
    )
    if cycle_values[0] == 0:
        all_freqs = fft_freqs
        tfdata = _fft_timefreq(
            data_3d,
            srate,
            limits,
            window_size,
            pad_ratio,
            all_freqs,
            timesout,
            ntimesout,
            str(ffttaper).lower(),
            str(detrend).lower(),
            str(causal).lower(),
        )
    else:
        all_freqs = requested_freqs
        wavelets = _wavelets(all_freqs, cycle_values, srate, str(freqscale).lower(), str(wletmethod).lower())
        window_size = max(len(wavelet) for wavelet in wavelets)
        tfdata = _wavelet_timefreq(
            data_3d,
            srate,
            limits,
            wavelets,
            timesout,
            ntimesout,
            str(detrend).lower(),
            str(causal).lower(),
        )
    output_freqs, tfdata = _select_requested_freqs(all_freqs, tfdata, requested_freqs)
    times, output_indices = _output_times(
        frame_count, limits, srate, window_size, timesout, ntimesout, str(causal).lower()
    )
    if timestretch is not None:
        tfdata = _time_stretch_tfdata(tfdata, output_indices + 1, timestretch)

    if tfdata.shape[0] == 1:
        tfdata = tfdata[0]
    itcvals = newtimefitc(tfdata, itctype) if tfdata.ndim >= 3 else None
    if str(subitc).lower() == "on":
        tfdata = _subtract_itc(tfdata, itcvals)
        itcvals = newtimefitc(tfdata, itctype)
    return TimeFrequencyDecomposition(tfdata, output_freqs, times, itcvals, window_size, cycle_values)


def _as_channel_epoch_data(data: Any, *, frames: int | None) -> np.ndarray:
    values = np.asarray(data, dtype=float)
    if values.ndim == 1:
        if frames is None:
            return values.reshape(1, values.size, 1)
        frames = int(frames)
        if frames <= 0 or values.size % frames:
            raise ValueError("1-D data length must be a multiple of frames")
        return values.reshape(-1, frames).T.reshape(1, frames, -1)
    if values.ndim == 2:
        if values.shape[0] == 1:
            return values.reshape(1, values.shape[1], 1)
        if frames is not None:
            frames = int(frames)
            if values.shape[0] == frames:
                return values[np.newaxis, :, :]
            if values.shape[1] == frames:
                return values.T[np.newaxis, :, :]
        return values[np.newaxis, :, :]
    if values.ndim == 3:
        return values
    raise ValueError("timefreq data must be 1-D, 2-D, or 3-D")


def _cycles(cycles: Any, wavelet: Any) -> np.ndarray:
    cycle_values = _numeric_vector(cycles)
    if cycle_values.size == 0:
        cycle_values = np.asarray([0.0])
    wavelet_values = _numeric_vector(wavelet)
    if cycle_values[0] == 0 and wavelet_values.size and wavelet_values[0] != 0:
        cycle_values = wavelet_values
    return cycle_values.astype(float)


def _window_size(frames: int, cycles: np.ndarray, explicit: int | None) -> int:
    if explicit is not None and int(explicit) > 0:
        winsize = int(explicit)
    else:
        winsize = max(int(2 ** (np.ceil(np.log2(max(frames, 2))) - 3)), 4)
    if winsize > frames:
        raise ValueError("Value of winsize must be less than or equal to frame length")
    if winsize < 2:
        raise ValueError("winsize must be at least 2")
    _ = cycles
    return winsize


def _padratio(value: Any) -> int:
    ratio = int(_first_numeric(value, 2))
    if ratio < 1 or 2 ** int(np.ceil(np.log2(ratio))) != ratio:
        raise ValueError("padratio must be an integer power of two")
    return ratio


def _tlimits(value: Any, frames: int, srate: float) -> np.ndarray:
    limits = _numeric_vector(value)
    if limits.size == 2:
        return limits.astype(float)
    return np.asarray([0.0, frames / float(srate) * 1000.0], dtype=float)


def _requested_freqs(
    srate: float,
    winsize: int,
    cycles: np.ndarray,
    requested: Any,
    nfreqs: Any,
    freqscale: str,
    padratio: int,
) -> tuple[np.ndarray, np.ndarray]:
    freq_values = _numeric_vector(requested)
    if freq_values.size == 0:
        freq_values = np.asarray([0.0, srate / 2.0])
    if cycles[0] != 0 and freq_values.size >= 2 and freq_values[0] == 0:
        freq_values[0] = srate * cycles[0] / winsize

    fft_freqs = np.linspace(0.0, srate / 2.0, int(winsize * padratio / 2) + 1)[1:]
    count_values = _numeric_vector(nfreqs, dtype=int)
    if freq_values.size == 1:
        return freq_values.astype(float), fft_freqs
    if freq_values.size > 2:
        return freq_values.astype(float), fft_freqs

    low, high = float(freq_values[0]), float(freq_values[1])
    if count_values.size:
        count = int(count_values[0])
    else:
        count = int(winsize * padratio / 2) + 1
        tmpfreqs = np.linspace(0.0, srate / 2.0, count)[1:]
        if cycles[0] == 0 and freqscale != "log":
            low = float(tmpfreqs[np.argmin(np.abs(tmpfreqs - low))])
            high = float(tmpfreqs[np.argmin(np.abs(tmpfreqs - high))])
        mask = (tmpfreqs >= low) & (tmpfreqs <= high)
        count = int(np.count_nonzero(mask))
        if np.isclose(low, high):
            count = 1
    count = max(1, count)
    if count == 1:
        return np.asarray([low], dtype=float), fft_freqs
    if freqscale == "log":
        if low <= 0 or high <= 0:
            raise ValueError("log frequency scale requires positive frequency limits")
        return np.exp(np.linspace(np.log(low), np.log(high), count)), fft_freqs
    return np.linspace(low, high, count), fft_freqs


def _fft_timefreq(
    data: np.ndarray,
    srate: float,
    tlimits: np.ndarray,
    winsize: int,
    padratio: int,
    freqs: np.ndarray,
    timesout: Any,
    ntimesout: Any,
    taper: str,
    detrend_mode: str,
    causal: str,
) -> np.ndarray:
    _, indices = _output_times(data.shape[1], tlimits, srate, winsize, timesout, ntimesout, causal)
    window = _fft_window(winsize, taper)
    nfft = winsize * padratio
    output = np.empty((data.shape[0], freqs.size, indices.size, data.shape[2]), dtype=complex)
    half = winsize // 2
    for time_index, center in enumerate(indices):
        if causal == "on":
            start = center - winsize + 1
            stop = center + 1
        else:
            start = center - half + 1
            stop = center + half + 1
        segment = data[:, start:stop, :].copy()
        segment -= np.nanmean(segment, axis=1, keepdims=True)
        if detrend_mode == "on":
            segment = signal.detrend(segment, axis=1, type="linear")
        transformed = np.fft.fft(segment * window[np.newaxis, :, np.newaxis], n=nfft, axis=1)
        output[:, :, time_index, :] = transformed[:, 1 : nfft // 2 + 1, :]
    return output


def _wavelet_timefreq(
    data: np.ndarray,
    srate: float,
    tlimits: np.ndarray,
    wavelets: list[np.ndarray],
    timesout: Any,
    ntimesout: Any,
    detrend_mode: str,
    causal: str,
) -> np.ndarray:
    winsize = max(len(wavelet) for wavelet in wavelets)
    _, indices = _output_times(data.shape[1], tlimits, srate, winsize, timesout, ntimesout, causal)
    output = np.empty((data.shape[0], len(wavelets), indices.size, data.shape[2]), dtype=complex)
    for time_index, center in enumerate(indices):
        for freq_index, wavelet in enumerate(wavelets):
            half = (len(wavelet) - 1) // 2
            if causal == "on":
                start = center - len(wavelet) + 1
                stop = center + 1
            else:
                start = center - half
                stop = center + half + 1
            segment = data[:, start:stop, :].copy()
            segment -= np.nanmean(segment, axis=1, keepdims=True)
            if detrend_mode == "on":
                segment = signal.detrend(segment, axis=1, type="linear")
            output[:, freq_index, time_index, :] = np.sum(segment * wavelet[np.newaxis, :, np.newaxis], axis=1)
    return output


def _output_times(
    frames: int,
    tlimits: np.ndarray,
    srate: float,
    winsize: int,
    timesout: Any,
    ntimesout: Any,
    causal: str,
) -> tuple[np.ndarray, np.ndarray]:
    timevect = np.linspace(float(tlimits[0]), float(tlimits[1]), int(frames))
    derived_srate = 1000.0 * (frames - 1) / max(float(tlimits[1]) - float(tlimits[0]), np.finfo(float).eps)
    srate = derived_srate if np.isfinite(derived_srate) and derived_srate > 0 else float(srate)
    explicit_times = _numeric_vector(timesout)
    ntimes = int(_first_numeric(ntimesout, 200))
    wintime = 500.0 * int(winsize) / srate
    if explicit_times.size == 0:
        if ntimes > 0:
            if ntimes > frames - winsize:
                ntimes = frames - winsize
            if ntimes <= 0:
                raise ValueError("Not enough data points, reduce the window size or lowest frequency")
            if causal == "on":
                wanted = np.linspace(float(tlimits[0]) + 2.0 * wintime, float(tlimits[1]), ntimes)
            else:
                wanted = np.linspace(float(tlimits[0]) + wintime, float(tlimits[1]) - wintime, ntimes)
            indices = _nearest_time_indices(timevect, wanted)
        else:
            step = max(1, -ntimes)
            if causal == "on":
                indices = np.arange(int(np.ceil(winsize + step)) - 1, len(timevect), step, dtype=int)
            else:
                start = int(np.ceil(winsize / 2.0 + step / 2.0)) - 1
                stop = len(timevect) - int(np.ceil(winsize / 2.0)) - 1
                indices = np.arange(start, stop + 1, step, dtype=int)
    else:
        if causal == "on":
            mask = (explicit_times >= float(tlimits[0]) + 2.0 * wintime - 0.0001) & (
                explicit_times <= float(tlimits[1])
            )
        else:
            mask = (explicit_times >= float(tlimits[0]) + wintime - 0.0001) & (
                explicit_times <= float(tlimits[1]) - wintime + 0.0001
            )
        wanted = explicit_times[mask]
        if wanted.size == 0:
            raise ValueError("No time points. Reduce time window or minimum frequency.")
        indices = _nearest_time_indices(timevect, wanted)
    indices = np.unique(indices[(indices >= 0) & (indices < frames)])
    if indices.size == 0:
        raise ValueError("No valid time points for time-frequency decomposition")
    return timevect[indices], indices


def _nearest_time_indices(timevect: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    return np.asarray([int(np.argmin(np.abs(timevect - value))) for value in wanted], dtype=int)


def _fft_window(winsize: int, taper: str) -> np.ndarray:
    if taper == "hanning":
        return symmetric_hanning(winsize)
    if taper == "hamming":
        return np.hamming(winsize)
    if taper == "blackmanharris":
        return signal.windows.blackmanharris(winsize)
    if taper == "none":
        return np.ones(winsize, dtype=float)
    raise ValueError("ffttaper must be 'hanning', 'hamming', 'blackmanharris', or 'none'")


def _wavelets(freqs: np.ndarray, cycles: np.ndarray, srate: float, freqscale: str, method: str) -> list[np.ndarray]:
    cycle_values = cycles
    if cycle_values.size == 2 and cycle_values[1] < 1:
        cycle_values = np.asarray([cycle_values[0], cycle_values[0] * freqs[-1] / freqs[0] * (1.0 - cycle_values[1])])
    if method == "dftfilt2":
        return dftfilt2(freqs, cycle_values, srate, cycleinc=freqscale)
    if method == "dftfilt3":
        wavelets, _, _, _ = dftfilt3(freqs, cycle_values, srate, cycleinc=freqscale)
        if isinstance(wavelets, np.ndarray):
            return [row for row in wavelets]
        return wavelets
    raise ValueError("wletmethod must be 'dftfilt2' or 'dftfilt3'")


def _select_requested_freqs(
    all_freqs: np.ndarray, tfdata: np.ndarray, requested_freqs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if requested_freqs.size == all_freqs.size and np.allclose(requested_freqs, all_freqs):
        return all_freqs, tfdata
    indices = np.asarray([int(np.argmin(np.abs(all_freqs - freq))) for freq in requested_freqs], dtype=int)
    _, unique_positions = np.unique(indices, return_index=True)
    indices = indices[np.sort(unique_positions)]
    return all_freqs[indices], tfdata[:, indices, :, :]


def _time_stretch_tfdata(tfdata: np.ndarray, output_frames: np.ndarray, timestretch: Any) -> np.ndarray:
    marks, refs = _timestretch_inputs(timestretch, tfdata.shape[3])
    if marks.size == 0:
        return tfdata
    if refs.size == 0:
        refs = np.median(marks, axis=0)
    refs_pos = _nearest_output_positions(refs, output_frames)
    refs_pos = np.sort(np.concatenate([refs_pos, np.asarray([1, output_frames.size], dtype=int)]))
    output = np.array(tfdata, copy=True)
    for trial_index in range(output.shape[3]):
        marks_pos = _nearest_output_positions(marks[trial_index], output_frames)
        marks_pos = np.sort(np.concatenate([marks_pos, np.asarray([1, output_frames.size], dtype=int)]))
        warp_matrix = timewarp(marks_pos, refs_pos)
        trial_values = output[:, :, :, trial_index]
        magnitude = np.abs(trial_values)
        phase = np.angle(trial_values)
        flat_magnitude = magnitude.reshape(-1, magnitude.shape[-1])
        warped_magnitude = (warp_matrix @ flat_magnitude.T).T.reshape(magnitude.shape)
        flat_phase = phase.reshape(-1, phase.shape[-1])
        warped_phase = np.empty_like(flat_phase)
        for row_index, row in enumerate(flat_phase):
            warped_phase[row_index] = angtimewarp(marks_pos, refs_pos, row)
        warped_phase = warped_phase.reshape(phase.shape)
        output[:, :, :, trial_index] = warped_magnitude * np.exp(1j * warped_phase)
    return output


def _timestretch_inputs(timestretch: Any, trial_count: int) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(timestretch, (list, tuple)) or len(timestretch) < 1:
        raise ValueError("timestretch must be a tuple/list containing event marks and optional reference frames")
    marks = np.asarray(timestretch[0], dtype=float)
    if marks.size == 0:
        return np.empty((trial_count, 0), dtype=float), np.asarray([], dtype=float)
    if marks.ndim == 1:
        marks = marks.reshape(1, -1)
    if marks.shape[0] != trial_count:
        raise ValueError("timestretch event marks must have one row per trial")
    refs = np.asarray(
        timestretch[1] if len(timestretch) > 1 and timestretch[1] is not None else [], dtype=float
    ).ravel()
    if refs.size and refs.size != marks.shape[1]:
        raise ValueError("timestretch reference frames must match the number of event columns")
    return marks, refs


def _nearest_output_positions(frame_values: np.ndarray, output_frames: np.ndarray) -> np.ndarray:
    return np.asarray(
        [int(np.argmin(np.abs(output_frames - frame))) + 1 for frame in np.asarray(frame_values, dtype=float).ravel()],
        dtype=int,
    )


def _subtract_itc(tfdata: np.ndarray, itcvals: np.ndarray | None) -> np.ndarray:
    if itcvals is None:
        return tfdata
    magnitude = np.maximum(np.abs(tfdata), np.finfo(float).tiny)
    expanded = np.expand_dims(itcvals, axis=-1)
    return (tfdata - magnitude * expanded) / magnitude


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
        return np.asarray([float(token) for token in text.replace(",", " ").split()], dtype=dtype)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=dtype).ravel()
    return np.asarray([value], dtype=dtype)


def _first_numeric(value: Any, default: float) -> float:
    values = _numeric_vector(value)
    return float(values[0]) if values.size else float(default)


__all__ = ["TimeFrequencyDecomposition", "timefreq"]
