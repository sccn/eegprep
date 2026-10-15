"""Riemannian ASR calibration, port of asr_calibrate_r.m."""

import logging
import math

import numpy as np
import scipy.linalg
import scipy.signal

from ...functions.miscfunc.misc import finite_matmul, round_mat
from .asr_calibrate import ASRCalibrationError, spectral_shaping_filter_available
from .private.riemann import rasr_nonlinear_eigenspace
from .private.stats import fit_eeg_distribution, geometric_median
from .private.yulewalk import yulewalk

logger = logging.getLogger(__name__)


def asr_calibrate_r(
    X,
    srate,
    cutoff=None,
    blocksize=None,
    B=None,
    A=None,
    window_len=None,
    window_overlap=None,
    max_dropout_fraction=None,
    min_clean_fraction=None,
    maxmem=None,
):
    """Calibrate Riemannian Artifact Subspace Reconstruction.

    State = asr_calibrate_r(Data, SamplingRate, Cutoff, ...)

    This follows ``asr_calibrate_r.m``. The mixing matrix is the matrix square
    root of the geometric median of block covariances. Component thresholds use
    ``rasr_nonlinear_eigenspace`` rather than a Euclidean eigendecomposition.
    Defaults that differ from ``asr_calibrate`` (cutoff 3, 0.1 s windows,
    overlap 0.5, minimum clean fraction 0.3) are the ones ``clean_asr`` relies
    on when Riemannian processing is requested.

    Args:
        X: Calibration data, channels by samples, zero-mean.
        srate: Sampling rate in Hz.
        cutoff: Standard-deviation cutoff. Default 3.
        blocksize: Covariance block size in samples. Default 10, raised when
            ``maxmem`` requires it.
        B, A: Spectral-shaping IIR coefficients. Default: ``yulewalk`` design
            from ``asr_calibrate_r.m``.
        window_len: Artifact-window length in seconds. Default 0.1.
        window_overlap: Window overlap fraction. Default 0.5.
        max_dropout_fraction: Maximum dropout fraction for threshold fitting.
            Default 0.1.
        min_clean_fraction: Minimum clean fraction for threshold fitting.
            Default 0.3.
        maxmem: Memory budget in MB used to grow ``blocksize``. Default 64.

    Returns:
        dict: Filter state for ``asr_process_r`` (``M``, ``T``, ``B``, ``A``,
        ``iir``, ``cov``, ``carry``, ``last_R``, ``last_trivial``).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError('Input data X must be a 2D array (channels x samples).')
    channels, samples = X.shape
    srate = float(srate)

    if cutoff is None:
        cutoff = 3.0
    if blocksize is None:
        blocksize = 10
    if maxmem is None:
        maxmem = 64
    if window_len is None:
        window_len = 0.1
    if window_overlap is None:
        window_overlap = 0.5
    if max_dropout_fraction is None:
        max_dropout_fraction = 0.1
    if min_clean_fraction is None:
        min_clean_fraction = 0.3

    blocksize = max(blocksize, math.ceil((channels * channels * samples * 8 * 3 * 2) / (maxmem * (2**21))))

    if B is None or A is None:
        if not spectral_shaping_filter_available(srate):
            raise ASRCalibrationError(
                f"ASR cannot design its spectral shaping filter at {srate:g} Hz. "
                "The default frequency breakpoints are not monotonic at this sampling rate."
            )
        freqvals = np.concatenate((np.array([0, 2, 3, 13, 16, 40, min(80.0, srate / 2.0 - 1.0)]) * 2.0 / srate, [1.0]))
        amps = np.array([3, 0.75, 0.33, 0.33, 1, 1, 3, 3], dtype=np.float64)
        if srate < 80:
            freqvals = np.delete(freqvals, freqvals.size - 3)
            amps = amps[:-1]
        B, A = yulewalk(8, freqvals, amps)

    X = X.copy()
    X[~np.isfinite(X)] = 0.0
    zi = np.zeros((channels, max(len(A), len(B)) - 1))
    shaped, iir_state = scipy.signal.lfilter(B, A, X, axis=1, zi=zi)
    if np.any(~np.isfinite(shaped)):
        raise RuntimeError(
            'The IIR filter diverged on your data. Please try using either '
            'a more conservative filter or removing some bad sections/channels from the calibration data.'
        )

    logger.info('Calculating blockwise covariances for Riemannian ASR...')
    outers = _block_outer_sums(shaped, int(blocksize))
    covariance_vectors = np.swapaxes(outers, -1, -2).reshape(outers.shape[0], -1) / blocksize
    median = geometric_median(covariance_vectors)
    robust = np.real(np.reshape(median, (channels, channels), order='F'))
    mixing = np.real(scipy.linalg.sqrtm(robust))

    window_samples = int(round_mat(window_len * srate))
    if samples < window_samples:
        raise ValueError(f'Not enough calibration data. Need at least {window_samples} samples, got {samples}.')

    logger.info('Determining per-component thresholds...')
    vectors, values = rasr_nonlinear_eigenspace(mixing, channels)
    eigenvalues = np.real(np.diag(values))
    order = np.argsort(eigenvalues, kind='mergesort')
    basis = np.real(vectors[:, order])
    transformed = np.abs(finite_matmul(shaped.T, basis))

    step = window_samples * (1.0 - window_overlap)
    if step <= 0:
        logger.warning('Window overlap >= 1, using step=1')
        step = 1
    window_starts = round_mat(np.arange(0, samples - window_samples, step)).astype(int)
    if len(window_starts) <= 1:
        raise ValueError(
            f"Fewer than 2 calibration windows: got {len(window_starts)} "
            f"from {samples} samples with window length {window_samples}."
        )
    window_indices = window_starts[:, None] + np.arange(window_samples)

    mu = np.zeros(channels)
    sig = np.zeros(channels)
    for component in reversed(range(channels)):
        rms_windows = np.sqrt(np.mean((transformed[:, component] ** 2)[window_indices], axis=1))
        try:
            mu_c, sig_c, _, _ = fit_eeg_distribution(
                rms_windows,
                min_clean_fraction=min_clean_fraction,
                max_dropout_fraction=max_dropout_fraction,
            )
            mu[component] = mu_c
            sig[component] = sig_c
        except Exception as exc:
            logger.warning(f'Distribution fitting failed for component {component}: {exc}')
            mu[component] = np.nan
            sig[component] = np.nan

    if np.any(np.isnan(mu)) or np.any(np.isnan(sig)):
        logger.warning('NaN values in threshold calculation. Results may be unreliable.')
        mu = np.nan_to_num(mu, nan=np.nanmedian(mu) if np.any(~np.isnan(mu)) else 1.0)
        sig = np.nan_to_num(sig, nan=np.nanmedian(sig) if np.any(~np.isnan(sig)) else 0.5)
    sig = np.maximum(sig, 0)
    thresholds = finite_matmul(np.diag(mu + cutoff * sig), basis.T)

    return {
        'M': mixing,
        'T': thresholds,
        'B': B,
        'A': A,
        'cov': None,
        'carry': None,
        'iir': iir_state,
        'last_R': None,
        'last_trivial': True,
    }


def _block_outer_sums(data, blocksize):
    """Sum per-sample outer products on the blocks used by ``asr_calibrate_r.m``.

    ``data`` is channels by samples. MATLAB clamps each block index to the last
    sample, so a short tail block repeats that sample instead of shrinking.
    """
    channels, samples = data.shape
    block_count = int(np.arange(0, samples, blocksize).size)
    totals = np.zeros((block_count, channels, channels))
    for offset in range(blocksize):
        matlab_index = np.arange(offset + 1, samples + offset + 1, blocksize)
        matlab_index = matlab_index[matlab_index <= samples + offset]
        sample_index = np.minimum(matlab_index, samples) - 1
        if sample_index.size != block_count:
            raise RuntimeError(f'ASR block index length {sample_index.size} != {block_count}')
        block = data[:, sample_index]
        totals += np.einsum('ib,jb->bij', block, block)
    return totals
