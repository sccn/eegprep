from typing import *

import numpy as np
from scipy.signal import fftconvolve

__all__ = ['design_kaiser', 'design_fir', 'filtfilt_fast']


def design_kaiser(
        lo: float,
        hi: float,
        atten: float,
        want_odd: bool,
        use_scipy: bool = False
) -> np.ndarray:
    """
    Design a Kaiser window for a low-pass FIR filter.

    Args:
        lo: normalized lower edge of the transition band
        hi: normalized upper edge of the transition band
        atten: stop-band attenuation in dB (-20log10(ratio))
        want_odd: whether the desired window length shall be odd
        use_scipy: whether to use scipy's kaiserord() function, which gives
          an approx. 2x longer window than the original function clean_rawdata

    Returns:
        the Kaiser window

    """
    from scipy.signal import kaiserord
    from scipy.signal.windows import kaiser

    if not use_scipy:
        # determine beta of the kaiser window
        if atten < 21:
            beta = 0
        elif atten < 50:
            beta = 0.5842*(atten-21)**0.4 + 0.07886*(atten-21)
        else:
            beta = 0.1102*(atten-8.7)

        #  determine the number of points
        N = int(round((atten - 7.95) / (2 * np.pi * 2.285 * (hi - lo))) + 1)
    else:
        N, beta = kaiserord(atten, hi - lo)

    if want_odd and N % 2 == 0:
        N = N + 1
    #  design the actual window
    return kaiser(N, beta, sym=True)


def design_fir(
        n: int,
        f: Union[np.ndarray, Sequence[float]],
        a: Union[np.ndarray, Sequence[float]],
        *,
        nfft: Optional[int] = None,
        w: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Design an FIR filter using the frequency-sampling method.
    The frequency response is interpolated cubically between the specified
    frequency points.

    Args:
      n: order of the filter
      f: vector of frequencies at which amplitudes shall be defined
         (starts with 0 and goes up to 1; try to avoid too sharp transitions)
      a: vector of amplitudes, one value per specified frequency
      nfft: optionally number of FFT bins to use
      w: optionally the window function to use

    """
    from scipy.interpolate import PchipInterpolator
    f, a = np.asarray(f), np.asarray(a)
    if nfft is None:
        nfft = max([512, 2**np.ceil(np.log(n) / np.log(2))])
    if w is None:
        from scipy.signal.windows import hamming
        w = hamming(n)

    # calculate interpolated frequency response
    # noinspection PyTypeChecker
    f = PchipInterpolator(np.round(f * nfft), a)(np.arange(nfft + 1))

    # set phase & transform into time domain
    f = f * np.exp(-(0.5 * n) * 1j * np.pi * np.arange(nfft + 1) / nfft)
    b = np.real(np.fft.ifft(np.concatenate((f, np.conj(f[::-1][1:-1])))))

    # apply window to kernel
    return b[:len(w)] * w


def filtfilt_fast(
        b: np.ndarray,
        a: Union[float, np.ndarray],
        x: np.ndarray,
) -> np.ndarray:
    """
    Apply a zero-phase forward-backward filter to a signal using FFTs; this is a
    drop-in replacement for scipy.signal.filtfilt() that is considerably faster
    for long signals.

    Args:
      b: numerator coefficients of the filter
      a: must be 1
      x: signal to filter (1-D array)
    """
    assert a == 1, "a must be 1; use filtfilt() for IIR filters"
    n = len(b)
    # pad the signal at both ends
    x_padded = np.pad(x, (n, n), mode='reflect', reflect_type='odd')
    # filter, reverse
    y_forward = fftconvolve(x_padded, b, mode='full')[::-1]
    # filter, reverse
    y_filtered = fftconvolve(y_forward, b, mode='full')[::-1]
    # trim off padding
    excess = len(y_filtered) - len(x)
    y_depadded = y_filtered[excess//2:-excess//2]
    return y_depadded
