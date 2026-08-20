"""Recursive (modified Yule-Walker) IIR filter design.

Implements the modified Yule-Walker method of Friedlander & Porat, "The Modified
Yule-Walker Method of ARMA Spectral Estimation", IEEE Trans. Aerospace and
Electronic Systems 20(2), 1984: sample the desired magnitude response on a dense
grid, convert it to an autocorrelation sequence, solve Toeplitz normal equations
for the denominator, then recover the numerator from a minimum-phase spectral
factorization of the additive decomposition.

ASR calibration needs this to build its spectral-shaping filter at the dataset's
own sampling rate. Coefficients are validated against MATLAB's ``yulewalk`` in
``tests/test_utils_asr.py``.

The design problem is intrinsically ill-conditioned when the interesting
structure sits at the very low end of the band, which is what happens at high
sampling rates: the Toeplitz system's condition number grows from ~1e3 at 258 Hz
to ~1e11 at 2048 Hz. Coefficients lose relative accuracy there even though the
resulting magnitude response stays accurate to micro-dB levels, so compare
responses rather than coefficients when testing high rates.
"""

import numpy as np
from scipy.linalg import lstsq, toeplitz
from scipy.signal import freqz, lfilter

# QR with column pivoting, matching the least-squares method MATLAB's mrdivide
# uses for the overdetermined systems in this design.
_LSTSQ_DRIVER = "gelsy"

_DEFAULT_NPT = 512


def _ls_rdivide(rhs, mat):
    """Least-squares solve of ``x @ mat.T == rhs`` (MATLAB's ``rhs / mat'``)."""
    solution, *_ = lstsq(mat, rhs, lapack_driver=_LSTSQ_DRIVER)
    return solution


def _denominator(R, order):
    """Denominator coefficients from the windowed autocorrelation ``R``."""
    nr = R.size
    normal_equations = toeplitz(R[order : nr - 1], R[order:0:-1])
    return np.concatenate(([1.0], _ls_rdivide(-R[order + 1 : nr], normal_equations)))


def _numerator(h, a, order):
    """Numerator of ``order`` whose response through ``1 / a`` best matches ``h``."""
    h = np.asarray(h, dtype=float).ravel()
    impulse = np.zeros(h.size)
    impulse[0] = 1.0
    denominator_response = lfilter(np.array([1.0]), a, impulse)
    first_row = np.zeros(order + 1)
    first_row[0] = 1.0
    return _ls_rdivide(h, toeplitz(denominator_response, first_row))


def _stabilize(a):
    """Reflect roots outside the unit circle inward, preserving magnitude response."""
    a = np.asarray(a, dtype=float)
    if a.size <= 1:
        return a
    roots = np.roots(a).astype(complex)
    nonzero = roots != 0
    outside = 0.5 * (np.sign(np.abs(roots[nonzero]) - 1.0) + 1.0)
    roots[nonzero] = (1.0 - outside) * roots[nonzero] + outside / np.conj(roots[nonzero])
    leading = a[np.flatnonzero(a != 0)[0]]
    return np.real(leading * np.poly(roots))


def _desired_response(ff, aa, npt, lap):
    """Piecewise-linear magnitude response sampled on ``npt`` points, DC to Nyquist.

    Breakpoints land on integer grid indices, so two nearby sampling rates that
    quantize to the same indices yield exactly the same filter.
    """
    response = np.zeros(npt)
    response[0] = aa[0]
    spacing = np.diff(ff)
    start = 1
    for index in range(ff.size - 1):
        if spacing[index] == 0:
            start = start - lap / 2
            stop = start + lap
        else:
            stop = int(np.fix(ff[index + 1] * npt))
        if start < 0 or stop > npt:
            raise ValueError("Frequency breakpoints fall outside the response grid.")
        grid = np.arange(start, stop + 1)
        ramp = np.zeros_like(grid, dtype=float) if stop == start else (grid - start) / (stop - start)
        response[int(start) - 1 : int(stop)] = ramp * aa[index + 1] + (1.0 - ramp) * aa[index]
        start = stop + 1
    return response


def yulewalk(order, ff, aa, npt=_DEFAULT_NPT, lap=None):
    """Design an IIR filter whose magnitude response approximates ``(ff, aa)``.

    Args:
        order: Filter order.
        ff: Breakpoint frequencies normalized to Nyquist, starting at 0, ending at 1,
            and non-decreasing. A repeated frequency introduces a step.
        aa: Desired magnitude at each breakpoint in ``ff``.
        npt: Number of grid points used to sample the desired response.
        lap: Grid points spread across a repeated frequency. Defaults to ``npt // 25``.

    Returns:
        Tuple of numerator and denominator coefficient arrays, each of length
        ``order + 1``.

    Raises:
        ValueError: If ``ff`` and ``aa`` differ in length, or ``ff`` does not run
            from 0 to 1 monotonically.
    """
    ff = np.asarray(ff, dtype=float).ravel()
    aa = np.asarray(aa, dtype=float).ravel()
    if ff.size != aa.size:
        raise ValueError("ff and aa must have the same number of elements.")
    if abs(ff[0]) > np.finfo(float).eps or abs(ff[-1] - 1.0) > np.finfo(float).eps:
        raise ValueError("ff must start at 0 and end at 1.")
    if np.any(np.diff(ff) < 0):
        raise ValueError("ff must be non-decreasing.")
    if lap is None:
        lap = npt // 25

    # Mirror the one-sided desired response into a full periodic spectrum.
    half = _desired_response(ff, aa, npt + 1, lap)
    spectrum = np.concatenate([half, half[npt - 1 : 0 : -1]])
    n = spectrum.size
    causal_len = (n + 1) // 2
    nr = 4 * order

    # Autocorrelation of the magnitude-squared response, Hamming-tapered.
    R = np.real(np.fft.ifft(spectrum * spectrum))[:nr]
    R = R * (0.54 + 0.46 * np.cos(np.pi * np.arange(nr) / (nr - 1)))

    A = _stabilize(_denominator(R, order))

    # Additive decomposition of the spectrum, then a cepstral (minimum-phase)
    # factorization whose causal part gives the numerator.
    additive = _numerator(np.concatenate(([R[0] / 2.0], R[1:nr])), A, order)
    power = 2.0 * np.real(freqz(additive, A, worN=n, whole=True)[1])
    causal_window = np.concatenate(([0.5], np.ones(causal_len - 1), np.zeros(n - causal_len)))
    # `power` dips negative where the fitted denominator overshoots, so the log must
    # take its complex branch; np.log would yield NaN on a real array there.
    log_power = np.log(power.astype(complex))
    impulse = np.fft.ifft(np.exp(np.fft.fft(causal_window * np.fft.ifft(log_power))))
    # _numerator's design matrix is real, so discarding the imaginary part here is
    # equivalent to discarding it after the solve, and keeps the solve real.
    return _numerator(impulse[:nr].real, A, order), A
