"""Headless Welch spectra and recording-quality summaries.

``compute_spectra`` is the numerical estimator ``spectopo`` plots. Importing
this module does not import Matplotlib.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import get_window, welch
from scipy.stats import median_abs_deviation

from eegprep.functions.miscfunc.misc import round_mat

# A requested line frequency must land on a returned bin. This is much tighter
# than half a bin, so a coarse grid cannot silently report a neighboring bin.
LINE_BIN_TOLERANCE_HZ = 1e-6


def compute_spectra(
    data: np.ndarray,
    frames: int,
    srate: float,
    *,
    percent: float = 100,
    winsize: int | None = None,
    overlap: int = 0,
    nfft: int | None = None,
    mapnorm: Any = None,
    wintype: str = "hamming",
    blckhn: int = 2,
) -> tuple[np.ndarray, np.ndarray, None]:
    """Return Welch power-spectral-density spectra in dB, shaped ``channels x frequencies``.

    Each value is ``10 * log10`` of a Welch density sample (power per Hz), not
    power integrated over a band and not SciPy's ``scaling="spectrum"`` bin
    power. The one-sided grid is ``k * srate / nfft`` for ``k = 0 .. nfft // 2``.
    The default segment length is ``min(round(srate), pnts)``, with half-away-from-zero
    rounding, and the default ``nfft`` is that length. The spacing is therefore
    1 Hz when ``srate`` is an integer and the record contains at least one
    second. Frequencies from 1 to 70 Hz are the samples whose grid frequencies
    fall in that range; they are still density samples. The last frequency is
    the Nyquist rate when ``nfft`` is even, and the greatest grid frequency at
    or below Nyquist when ``nfft`` is odd. No bin is above the Nyquist rate.

    Epoched input (3-D ``(nchan, pnts, trials)`` or 2-D ``(nchan, pnts*trials)`` with
    ``frames < sample_count``) is handled per epoch and averaged in linear power
    before the dB conversion, matching EEGLAB ``spectopo``'s ``spectcomp``. The
    dB values are not averaged. 2-D concatenated data is unstacked column-major
    to match MATLAB's ``reshape``.

    ``mapnorm`` (a component scalp-map column) scales the linear power by
    ``sqrt(mean(mapnorm**4))`` before the dB conversion, so a single component's
    activation spectrum reflects its projected scalp power (EEGLAB ``spectopo.m``).
    """
    values = np.asarray(data, dtype=float)
    if values.ndim == 3:
        _, pnts, trials = values.shape
        epochs = values
    elif values.ndim == 2:
        nchan, sample_count = values.shape
        frames = int(frames) if frames > 0 else sample_count
        if sample_count % frames:
            raise ValueError("compute_spectra: sample count is not an integer multiple of frames")
        trials = sample_count // frames
        # column-major so trial ``k`` occupies samples ``[k*frames:(k+1)*frames]``,
        # matching MATLAB's ``reshape(data, nchan, pnts, trials)``
        epochs = values.reshape(nchan, frames, trials, order="F") if trials > 1 else values[:, :, np.newaxis]
        pnts = frames
    else:
        raise ValueError("data must be a 2-D or 3-D channel-major array")

    if percent <= 0 or percent > 100:
        raise ValueError("percent must be in the range (0, 100]")
    if percent < 100:
        if trials > 1:
            keep_trials = max(1, int(round(trials * percent / 100.0)))
            epochs = epochs[:, :, :keep_trials]
            trials = keep_trials
        else:
            keep = max(1, int(round_mat(pnts * percent / 100.0)))
            epochs = epochs[:, :keep, :]
            pnts = keep

    window_type = str(wintype).strip().lower()
    if window_type not in {"hamming", "blackmanharris"}:
        raise ValueError("wintype must be 'hamming' or 'blackmanharris'")
    nperseg = int(winsize or min(int(round_mat(srate)), pnts))
    if window_type == "blackmanharris":
        divisor = int(blckhn)
        if divisor <= 0:
            raise ValueError("blckhn must be a positive integer")
        nperseg = int(np.floor(nperseg / divisor + 0.5))
    nperseg = max(1, min(nperseg, pnts))
    noverlap = max(0, min(int(overlap), nperseg - 1))
    # EEGLAB uses symmetric windows and disables detrending in pwelch.
    window = get_window(window_type, nperseg, fftbins=False)
    if window_type == "blackmanharris":
        nfft = 1 << int(np.ceil(np.log2(nperseg)))
    freqs = None
    psd_sum: np.ndarray | None = None
    for index in range(trials):
        freqs, power = welch(
            epochs[:, :, index],
            fs=float(srate),
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            axis=1,
            detrend=False,
            scaling="density",
        )
        if psd_sum is None:
            psd_sum = np.zeros_like(power)
        psd_sum += power
    psd_mean = psd_sum / trials
    if mapnorm is not None:
        psd_mean = psd_mean * np.sqrt(np.mean(np.asarray(mapnorm, dtype=float) ** 4))
    spectra = 10.0 * np.log10(np.maximum(psd_mean, np.finfo(float).tiny))
    return spectra, freqs, None


def line_noise_excess_db(
    eeg: dict[str, Any],
    freqs: Any = (50.0, 60.0),
    neighbours: int = 1,
) -> np.ndarray:
    """Return excess log-density at line frequencies over neighboring bins, in dB.

    The spectrum is the default ``compute_spectra`` result: Welch density in
    dB, with epochs averaged in linear power and then converted once. When the
    sampling rate is an integer and the record is at least one second long, the
    bins are 1 Hz apart.

    A requested frequency above ``srate / 2`` raises ``ValueError``, including
    when the Nyquist bin is the nearest bin. Otherwise the frequency must equal
    a returned bin within ``LINE_BIN_TOLERANCE_HZ`` (1e-6 Hz). The excess at
    that bin is its dB value minus the arithmetic mean of the ``neighbours``
    bins on each side. The center bin is excluded, and the mean is taken in dB
    rather than in linear power. A neighbor window that leaves the grid raises
    ``ValueError``; a line on the Nyquist bin has no upper neighbor.

    The result has shape ``(channels, len(freqs))``. The dataset is not modified.

    Args:
        eeg: EEG dataset with channel-major ``data`` and ``srate`` in Hz.
            ``pnts`` is the epoch length when ``data`` holds concatenated epochs.
        freqs: Line frequencies in Hz. A bare number requests one frequency.
            The default is 50 and 60.
        neighbours: Number of bins on each side of the line bin. A positive
            integer.

    Returns:
        Excess in dB, one column per requested frequency, in the requested order.
    """
    values, frames, srate = _dataset_signal(eeg)
    requested = _line_frequencies(freqs)
    count = _neighbour_count(neighbours)
    spectra, frequency_values, _std = compute_spectra(values, frames, srate)
    grid = np.asarray(frequency_values, dtype=float).ravel()
    nyquist = srate / 2.0
    columns = []
    for frequency in requested:
        index = _line_bin(grid, float(frequency), nyquist)
        if index - count < 0 or index + count >= grid.size:
            raise ValueError(
                f"the {count} neighboring bins on each side of {float(frequency):g} Hz fall outside "
                f"the spectrum grid ({float(grid[0]):g} to {float(grid[-1]):g} Hz)"
            )
        neighborhood = np.concatenate(
            (spectra[:, index - count : index], spectra[:, index + 1 : index + count + 1]),
            axis=1,
        )
        columns.append(spectra[:, index] - neighborhood.mean(axis=1))
    return np.column_stack(columns)


def channel_robust_amplitude(eeg: dict[str, Any]) -> np.ndarray:
    """Return a normal-scaled MAD amplitude for each channel.

    Every sample of the channel is pooled, including samples from every epoch.
    The amplitude is not the mean of per-epoch amplitudes. It is
    ``median(|x - median(x)|) / norm.ppf(0.75)``, computed by
    ``scipy.stats.median_abs_deviation(..., scale="normal")``. That scale
    factor is approximately 1.4826, so the result estimates the standard
    deviation of a normal distribution. Units are the units of ``EEG["data"]``
    (microvolts when the recording is in microvolts). A channel containing a
    NaN has amplitude NaN. The dataset is not modified.

    Args:
        eeg: EEG dataset with channel-major ``data`` of shape
            ``(channels, samples)`` or ``(channels, samples, epochs)``.

    Returns:
        Amplitude of shape ``(channels,)``.
    """
    values = _dataset_data(eeg)
    samples = np.reshape(np.asarray(values, dtype=float), (values.shape[0], -1))
    return median_abs_deviation(samples, axis=1, scale="normal", nan_policy="propagate")


def _dataset_data(eeg: dict[str, Any]) -> np.ndarray:
    if not isinstance(eeg, dict):
        raise TypeError("EEG must be a dataset dictionary")
    try:
        data = eeg["data"]
    except KeyError as exc:
        raise ValueError("EEG dataset needs 'data'") from exc
    values = np.asarray(data)
    if values.ndim not in (2, 3) or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("EEG data must have shape (channels, samples) or (channels, samples, epochs)")
    if values.ndim == 3 and values.shape[2] == 0:
        raise ValueError("EEG data must have shape (channels, samples) or (channels, samples, epochs)")
    return values


def _dataset_signal(eeg: dict[str, Any]) -> tuple[np.ndarray, int, float]:
    values = _dataset_data(eeg)
    try:
        srate = eeg["srate"]
    except KeyError as exc:
        raise ValueError("EEG dataset needs 'srate'") from exc
    rate = float(srate)
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError(f"EEG srate must be a positive finite sampling rate in Hz, got {srate!r}")
    pnts = eeg.get("pnts")
    frames = int(pnts) if pnts else int(values.shape[1])
    return values, frames, rate


def _line_frequencies(freqs: Any) -> np.ndarray:
    requested = np.asarray(freqs, dtype=float)
    if requested.ndim == 0:
        requested = requested.reshape(1)
    if requested.ndim != 1 or requested.size == 0 or not np.all(np.isfinite(requested)):
        raise ValueError("freqs must be one or more finite frequencies in Hz")
    return requested


def _neighbour_count(neighbours: int) -> int:
    if isinstance(neighbours, (bool, np.bool_)) or not isinstance(neighbours, (int, np.integer)):
        raise ValueError("neighbours must be a positive integer")
    count = int(neighbours)
    if count < 1:
        raise ValueError("neighbours must be a positive integer")
    return count


def _line_bin(frequency_values: np.ndarray, frequency: float, nyquist: float) -> int:
    if frequency > nyquist:
        raise ValueError(f"{frequency:g} Hz is above the Nyquist frequency ({nyquist:g} Hz)")
    index = int(np.argmin(np.abs(frequency_values - frequency)))
    if abs(float(frequency_values[index]) - frequency) > LINE_BIN_TOLERANCE_HZ:
        raise ValueError(
            f"{frequency:g} Hz is not a bin on the spectrum grid (nearest bin is {float(frequency_values[index]):g} Hz)"
        )
    return index
