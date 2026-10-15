"""DC-padded FIR filtering helper."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import lfilter

# fftfilt.m chooses a power-of-two length by minimizing block count times these
# FFT flop estimates from the Signal Processing Toolbox.
_FFT_FLOPS = (
    18,
    59,
    138,
    303,
    660,
    1441,
    3150,
    6875,
    14952,
    32373,
    69762,
    149647,
    319644,
    680105,
    1441974,
    3047619,
    6422736,
    13500637,
    28311786,
    59244791,
)


def fir_filterdcpadded(
    b: Any,
    a: Any,
    data: Any,
    causal: bool = False,
    usefftfilt: bool = False,
) -> np.ndarray:
    """Pad data with endpoint constants and filter along the first axis."""
    coeffs = np.asarray(b, dtype=float).ravel()
    denom = np.asarray(a, dtype=float).ravel()
    if coeffs.size == 0 or denom.size != 1 or denom[0] != 1:
        raise ValueError(
            "Not a FIR filter. onepass-zerophase and onepass-minphase filtering is available for FIR filters only."
        )
    if coeffs.size % 2 != 1:
        raise ValueError("Filter order is not even.")
    x = np.asarray(data)
    if x.ndim == 1:
        x = x[:, np.newaxis]
        squeeze = True
    elif x.ndim == 2:
        squeeze = False
    else:
        raise ValueError("data must be 1D or 2D with time along the first axis.")
    group_delay = (coeffs.size - 1) // 2
    if group_delay == 0:
        return np.asarray(data).copy()
    if not causal:
        is_sym = np.allclose(coeffs[:group_delay], coeffs[:group_delay:-1])
        is_antisym = np.allclose(coeffs[:group_delay], -coeffs[:group_delay:-1]) and np.isclose(coeffs[group_delay], 0)
        if not (is_sym or is_antisym):
            raise ValueError(
                "Filter is not anti-/symmetric. For onepass-zerophase filtering the filter must be anti-/symmetric."
            )
    was_single = x.dtype == np.float32
    x = x.astype(float, copy=False)
    start_width = 2 * group_delay if causal else group_delay
    start_pad = np.repeat(x[:1, :], start_width, axis=0)
    end_pad = np.empty((0, x.shape[1]), dtype=float) if causal else np.repeat(x[-1:, :], group_delay, axis=0)
    padded = np.concatenate([start_pad, x, end_pad], axis=0)
    if usefftfilt:
        filtered = _fftfilt(coeffs, padded)
    else:
        filtered = lfilter(coeffs, [1.0], padded, axis=0)
    filtered = filtered[2 * group_delay :, :]
    if was_single:
        filtered = filtered.astype(np.float32)
    if squeeze:
        return filtered[:, 0]
    return filtered


def _fftfilt_lengths(n_coeff: int, n_samples: int) -> tuple[int, int]:
    if n_samples <= 0:
        return 1, 1
    if n_coeff >= n_samples or n_coeff > 2**20:
        nfft = 1 << int(np.ceil(np.log2(max(n_coeff + n_samples - 1, 1))))
        return nfft, n_samples
    sizes = 2 ** np.arange(1, 21)
    valid = sizes > (n_coeff - 1)
    sizes = sizes[valid]
    block_lengths = sizes - (n_coeff - 1)
    flops = np.asarray(_FFT_FLOPS, dtype=float)[valid]
    index = int(np.argmin(np.ceil(n_samples / block_lengths) * flops))
    return int(sizes[index]), int(block_lengths[index])


def _fftfilt(coeffs: np.ndarray, data: np.ndarray) -> np.ndarray:
    """Overlap-add FIR filter matching MATLAB ``fftfilt`` along the first axis."""
    n_samples = data.shape[0]
    nfft, block = _fftfilt_lengths(coeffs.size, n_samples)
    spectrum = np.fft.fft(coeffs, nfft)
    output = np.zeros((n_samples, data.shape[1]), dtype=np.complex128)
    start = 0
    while start < n_samples:
        stop = min(start + block, n_samples)
        chunk = data[start:stop]
        # fftfilt.m repeats a one-sample block out to the FFT length.
        if chunk.shape[0] == 1:
            transformed = np.fft.fft(np.repeat(chunk, nfft, axis=0), axis=0)
        else:
            transformed = np.fft.fft(chunk, nfft, axis=0)
        filtered = np.fft.ifft(transformed * spectrum[:, None], axis=0)
        end = min(n_samples, start + nfft)
        output[start:end] += filtered[: end - start]
        start += block
    if np.isrealobj(coeffs) and np.isrealobj(data):
        return np.real(output)
    return output
