"""FIR design helpers shared by firfilt and clean_rawdata."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from eegprep.functions.miscfunc.parity import round_mat

__all__ = ["design_fir", "design_kaiser"]


def design_kaiser(lo: float, hi: float, atten: float, want_odd: bool, use_scipy: bool = False) -> np.ndarray:
    """Design a Kaiser window for a low-pass FIR filter.

    Parameters
    ----------
    lo : float
        Normalized lower edge of the transition band.
    hi : float
        Normalized upper edge of the transition band.
    atten : float
        Stop-band attenuation in dB (-20log10(ratio)).
    want_odd : bool
        Whether the desired window length shall be odd.
    use_scipy : bool, optional
        Whether to use scipy's kaiserord() function, which gives
        an approx. 2x longer window than the original function clean_rawdata.

    Returns
    -------
    np.ndarray
        The Kaiser window.
    """
    from scipy.signal import kaiserord
    from scipy.signal.windows import kaiser

    if not use_scipy:
        if atten < 21:
            beta = 0
        elif atten < 50:
            beta = 0.5842 * (atten - 21) ** 0.4 + 0.07886 * (atten - 21)
        else:
            beta = 0.1102 * (atten - 8.7)
        n_points = int(round_mat((atten - 7.95) / (2 * np.pi * 2.285 * (hi - lo))) + 1)
    else:
        n_points, beta = kaiserord(atten, hi - lo)

    if want_odd and n_points % 2 == 0:
        n_points += 1
    return kaiser(n_points, beta, sym=True)


def design_fir(
    n: int,
    f: Union[np.ndarray, Sequence[float]],
    a: Union[np.ndarray, Sequence[float]],
    *,
    nfft: Optional[int] = None,
    w: Optional[np.ndarray] = None,
    compat: bool = True,
) -> np.ndarray:
    """Design an FIR filter using the frequency-sampling method.

    The frequency response is interpolated cubically between the specified
    frequency points.

    Parameters
    ----------
    n : int
        Order of the filter.
    f : array_like
        Vector of frequencies at which amplitudes shall be defined
        (starts with 0 and goes up to 1; try to avoid too sharp transitions).
    a : array_like
        Vector of amplitudes, one value per specified frequency.
    nfft : int, optional
        Optionally number of FFT bins to use.
    w : array_like, optional
        Optionally the window function to use.
    compat : bool, optional
        Whether to use the original MATLAB-compatible filter design
        (where the window is off by 1 sample).

    Returns
    -------
    np.ndarray
        The filter coefficients.
    """
    from scipy.interpolate import PchipInterpolator

    f, a = np.asarray(f), np.asarray(a)
    if nfft is None:
        nfft = max([512, 2 ** np.ceil(np.log(n) / np.log(2))])
    if w is None:
        if compat:
            w = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n + 1) / n)
        else:
            from scipy.signal.windows import hamming

            w = hamming(n)

    response = PchipInterpolator(round_mat(f * nfft), a)(np.arange(nfft + 1))
    response = response * np.exp(-(0.5 * n) * 1j * np.pi * np.arange(nfft + 1) / nfft)
    b = np.real(np.fft.ifft(np.concatenate((response, np.conj(response[::-1][1:-1])))))
    return b[: len(w)] * w
