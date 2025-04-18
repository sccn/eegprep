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
        compat: bool = True,
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
      compat: whether to use the original MATLAB-compatible filter design
        (where the window is off by 1 sample)
    """
    from scipy.interpolate import PchipInterpolator
    f, a = np.asarray(f), np.asarray(a)
    if nfft is None:
        nfft = max([512, 2**np.ceil(np.log(n) / np.log(2))])
    if w is None:
        if compat:
            w = 0.54 - 0.46*np.cos(2*np.pi*np.arange(n+1)/n)
        else:
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


def moving_average(X, *, N=3, axis=-1, Z=None, inplace=False, transform=None, init=None):
    """lfilter()-style moving average function with support for state.

    Args:
        X: signal to filter
        N: number of points that shall be averaged (window length)
        axis: axis along which to filter; note: IF you use transform, and if
          it inserts additional axes, the same index needs to work before and
          after the transform (e.g., you can use negative indices to count from
          the end if needed to accomplish that)
        Z: initial state (or None)
        inplace: whether to overwrite the input
        transform: optionally a transformation to apply to each input sample,
          usually to generate higher-dimensional data; one use case is to calculate
          covariance matrices per sample on the fly instead of having the moving average
          to apply to and buffer potentially very large covariance data
          (by passing lambda x: x[:, None] @ x[None, :])
        init: how to behave on the first N samples of input; if set to 0,
          this will behave as if the data were pre-pended by zeros; if set to None,
          this will average the (fewer, noisier) samples in the buffer.

    Returns:
        X': filtered signal
        Z': final state (can be passed into the next call to moving_average())
        
    License:
        Copyright (c) 2015-2025 Syntrogi Inc. dba Intheon.

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.

    """
    class MovAvgState:
        """State representation for moving_average() filter function."""
        def __init__(self, p, buf, acc, n):
            self.p, self.buf, self.acc, self.n = p, buf, acc, n

    if transform and inplace:
        raise ValueError("You cannot use inplace and transform at the same time.")
    if transform is None:
        def transform(x): return x
    # we're doing some extra homework here to be able to buffer and transform the data
    # without swizzling axes (which creates temporaries that can exceed the memory),
    # so we have to be able to do all operations on input and output along the desired axis,
    # which may also count from the end

    def slice_at(x, k):
        """Generate an index slice that will slice x at the desired axis."""
        slices = [slice(None)]*x.ndim
        slices[axis] = k
        return tuple(slices)

    if not inplace:
        # Complicated expression to generate a new shape after transform with the
        # right shape at axis
        Yshp = list(np.stack([transform(X[slice_at(X, 0)])], axis=axis).shape)
        Yshp[axis] = X.shape[axis]
        Y = np.zeros_like(X, shape=Yshp)
    else:
        Y = None
    if not Z:
        if init is None:
            init_n = 0
        elif init == 0:
            init_n = N
        else:
            raise ValueError("init must be 0 or None")
        Z = MovAvgState(p=0, buf=np.zeros_like(X[slice_at(X, [0]*N)]),
                        acc=np.zeros_like(transform(X[slice_at(X, 0)])), n=init_n)

    for k in range(X.shape[axis]):
        # this is basically the buffered moving average trick (updating/downdating
        # the covariance matrix with each added/removed sample), but additionally
        # we're allowing the samples to be transformed to e.g. higher dimensions
        # to reduce buffer space, which can be very large for long moving averages
        e = X[slice_at(X, k)]
        Z.n += 1
        Z.acc += transform(e) - transform(Z.buf[slice_at(Z.buf, Z.p)])
        Z.buf[slice_at(Z.buf, Z.p)] = e
        res = Z.acc / min(N, Z.n)
        if inplace:
            X[slice_at(X, k)] = res
        else:
            Y[slice_at(Y, k)] = res
        Z.p = (Z.p + 1) % N
    return (X if inplace else Y), Z
