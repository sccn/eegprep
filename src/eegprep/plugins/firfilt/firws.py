"""Windowed-sinc FIR filter design."""

from typing import Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ["firws"]


def firws(
    m: int, f: Union[float, Sequence[float]], t: Optional[str] = None, w: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float]:
    """Designs windowed sinc type I linear phase FIR filter.

    Parameters
    ----------
    m : int
        Filter order (mandatory even).
    f : float or sequence of float
        Vector or scalar of cutoff frequency/ies (-6 dB; pi rad / sample).
    t : str, optional
        'high' for highpass, 'stop' for bandstop filter (default low-/bandpass).
    w : array_like, optional
        Vector of length m + 1 defining window (default hamming).

    Returns
    -------
    b : np.ndarray
        Filter coefficients.
    a : float
        Always 1 (FIR filter).

    Examples
    --------
    fs = 500; cutoff = 0.5; df = 1;
    m = firwsord('hamming', fs, df)[0]
    b, a = firws(m, cutoff / (fs / 2), 'high', scipy.signal.windows.hamming(m + 1))

    Notes
    -----
    Based on a MATLAB implementation by Andreas Widmann, University of Leipzig, 2005.
    """
    from scipy.signal.windows import hamming

    a = 1.0

    if m <= 0 or not isinstance(m, int) or m % 2 != 0:
        raise ValueError('Filter order must be a real, even, positive integer.')

    # Convert f to array and normalize
    f_arr = np.asarray(f, dtype=float)
    if f_arr.ndim == 0:
        f_arr = f_arr.reshape(1)
    f = f_arr / 2.0

    if np.any(f <= 0) or np.any(f >= 0.5):
        raise ValueError('Frequencies must fall in range between 0 and 1.')

    if t is None:
        t = ''

    if w is None:
        if t is not None and not isinstance(t, str):
            # Handle case where third argument is window, not filter type
            w = t
            t = ''
        else:
            w = hamming(m + 1)

    # Make window row vector
    w = np.asarray(w).flatten()

    b = _fkernel(m, f[0], w)

    if len(f) == 1 and t.lower() == 'high':
        b = _fspecinv(b)

    if len(f) == 2:
        b = b + _fspecinv(_fkernel(m, f[1], w))
        if not t or t.lower() != 'stop':
            b = _fspecinv(b)

    return b, a


def _fkernel(m: int, f: float, w: np.ndarray) -> np.ndarray:
    """Compute filter kernel.

    Parameters
    ----------
    m : int
        Filter order.
    f : float
        Normalized cutoff frequency.
    w : np.ndarray
        Window function.

    Returns
    -------
    b : np.ndarray
        Filter kernel.
    """
    # Create range -m/2 : m/2
    n = np.arange(-m // 2, m // 2 + 1, dtype=float)

    # Compute sinc function
    b = np.zeros_like(n)

    # Handle n == 0 case (no division by zero)
    zero_idx = n == 0
    b[zero_idx] = 2 * np.pi * f

    # Handle n != 0 case
    nonzero_idx = n != 0
    b[nonzero_idx] = np.sin(2 * np.pi * f * n[nonzero_idx]) / n[nonzero_idx]

    # Apply window
    b = b * w

    # Normalization to unity gain at DC
    b = b / np.sum(b)

    return b


def _fspecinv(b: np.ndarray) -> np.ndarray:
    """Perform spectral inversion.

    Parameters
    ----------
    b : np.ndarray
        Filter coefficients.

    Returns
    -------
    b_inv : np.ndarray
        Spectrally inverted filter coefficients.
    """
    b_inv = -b.copy()
    center_idx = (len(b) - 1) // 2
    b_inv[center_idx] = b_inv[center_idx] + 1
    return b_inv
