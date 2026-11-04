"""Signal processing utilities."""

from typing import Union

import numpy as np
from scipy.signal import fftconvolve, lfilter

# filter_fast.m uses time-domain filter() unless the numerator and the signal
# are both long enough that the FFT path is faster.
_FFT_MIN_NUMERATOR = 256
_FFT_MIN_SIGNAL = 1024
_FFT_MIN_PRODUCT = 4_000_000

__all__ = ['filtfilt_fast', 'moving_average']


def filtfilt_fast(
    b: np.ndarray,
    a: Union[float, np.ndarray],
    x: np.ndarray,
) -> np.ndarray:
    """Apply a zero-phase forward-backward filter.

    Short filters use time-domain filtering, as ``filter_fast.m`` does when
    ``length(B) < 256``. Longer filters use FFT convolution. A few NaNs then
    stay local instead of blanking the channel.

    Parameters
    ----------
    b : np.ndarray
        Numerator coefficients of the filter.
    a : float or np.ndarray
        Must be 1.
    x : np.ndarray
        Signal to filter (1-D array).

    Returns
    -------
    np.ndarray
        The filtered signal.
    """
    assert a == 1, "a must be 1; use filtfilt() for IIR filters"
    b = np.asarray(b, dtype=float).ravel()
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = b.size
    if x.size == 0:
        return x.copy()
    # filtfilt_fast.m pads before filter_fast, and filter_fast sees that length.
    padded = _odd_reflection_pad(x, n)
    if _filter_fast_uses_time_domain(n, padded.shape[0]):
        filtered = lfilter(b, [1.0], padded, axis=0)
        filtered = lfilter(b, [1.0], filtered[::-1], axis=0)[::-1]
        return filtered[n : n + x.shape[0]]
    # pad the signal at both ends
    x_padded = np.pad(x, (n, n), mode='reflect', reflect_type='odd')
    y_forward = fftconvolve(x_padded, b, mode='full')[::-1]
    y_filtered = fftconvolve(y_forward, b, mode='full')[::-1]
    # trim off padding
    excess = len(y_filtered) - len(x)
    y_depadded = y_filtered[excess // 2 : -excess // 2]
    return y_depadded


def _filter_fast_uses_time_domain(numerator_length: int, signal_length: int) -> bool:
    return (
        numerator_length < _FFT_MIN_NUMERATOR
        or signal_length < _FFT_MIN_SIGNAL
        or signal_length <= numerator_length
        or numerator_length * signal_length < _FFT_MIN_PRODUCT
    )


def _odd_reflection_pad(x: np.ndarray, width: int) -> np.ndarray:
    """Odd reflection of ``width`` samples, matching filtfilt_fast.m."""
    length = x.shape[0]
    left_index = np.mod(np.arange(width + 1, 1, -1) - 1, length)
    right_index = np.mod(np.arange(length - 1, length - width - 1, -1) - 1, length)
    left = 2.0 * x[0] - x[left_index]
    right = 2.0 * x[-1] - x[right_index]
    return np.concatenate([left, x, right])


def moving_average(X, *, N=3, axis=-1, Z=None, inplace=False, transform=None, init=None):
    """Lfilter()-style moving average function with support for state.

    Parameters
    ----------
    X : array_like
        Signal to filter.
    N : int, optional
        Number of points that shall be averaged (window length).
    axis : int, optional
        Axis along which to filter; note: IF you use transform, and if
        it inserts additional axes, the same index needs to work before and
        after the transform (e.g., you can use negative indices to count from
        the end if needed to accomplish that).
    Z : object, optional
        Initial state (or None).
    inplace : bool, optional
        Whether to overwrite the input.
    transform : callable, optional
        Optionally a transformation to apply to each input sample,
        usually to generate higher-dimensional data; one use case is to calculate
        covariance matrices per sample on the fly instead of having the moving average
        to apply to and buffer potentially very large covariance data
        (by passing lambda x: x[:, None] @ x[None, :]).
    init : int or None, optional
        How to behave on the first N samples of input; if set to 0,
        this will behave as if the data were pre-pended by zeros; if set to None,
        this will average the (fewer, noisier) samples in the buffer.

    Returns
    -------
    X' : array_like
        Filtered signal.
    Z' : object
        Final state (can be passed into the next call to moving_average()).

    License
    -------
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

        def transform(x):
            return x
    # we're doing some extra homework here to be able to buffer and transform the data
    # without swizzling axes (which creates temporaries that can exceed the memory),
    # so we have to be able to do all operations on input and output along the desired axis,
    # which may also count from the end

    def slice_at(x, k):
        """Generate an index slice that will slice x at the desired axis."""
        slices = [slice(None)] * x.ndim
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
        Z = MovAvgState(
            p=0, buf=np.zeros_like(X[slice_at(X, [0] * N)]), acc=np.zeros_like(transform(X[slice_at(X, 0)])), n=init_n
        )

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
