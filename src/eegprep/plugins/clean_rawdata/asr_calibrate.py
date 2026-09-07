"""Artifact Subspace Reconstruction (ASR) utilities."""

import logging
import math
import numpy as np
import scipy.signal
import scipy.linalg

from ...functions.miscfunc.misc import canonicalize_signs, finite_matmul, round_mat
from .private.covariance import cov_mean, cov_shrinkage
from .private.stats import fit_eeg_distribution, geometric_median
from .private.yulewalk import yulewalk

logger = logging.getLogger(__name__)


def asr_calibrate(
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
    useriemannian=None,
    compatibility=None,
):
    """Calibration function for the Artifact Subspace Reconstruction (ASR) method.

    State = asr_calibrate(Data, SamplingRate, Cutoff, BlockSize, FilterB, FilterA, WindowLength, WindowOverlap, MaxDropoutFraction, MinCleanFraction, MaxMemory)

    The input to this data is a multi-channel time series of calibration data. In typical uses the
    calibration data is clean resting EEG data of ca. 1 minute duration (can also be longer). One can
    also use on-task data if the fraction of artifact content is below the breakdown point of the
    robust statistics used for estimation (50% theoretical, ~30% practical). If the data has a
    proportion of more than 30-50% artifacts then bad time windows should be removed beforehand. This
    data is used to estimate the thresholds that are used by the ASR processing function to identify
    and remove artifact components.

    The calibration data must have been recorded for the same cap design from which data for cleanup
    will be recorded, and ideally should be from the same session and same subject, but it is possible
    to reuse the calibration data from a previous session and montage to the extent that the cap is
    placed in the same location (where loss in accuracy is more or less proportional to the mismatch
    in cap placement).

    The calibration data should have been high-pass filtered (for example at 0.5Hz or 1Hz using a
    Butterworth IIR filter).

    Args:
      X (np.ndarray): Calibration data [#channels x #samples]; *zero-mean* (e.g., high-pass filtered) and
                      reasonably clean EEG of not much less than 30 seconds length (this method is typically
                      used with 1 minute or more).
      srate (float): Sampling rate of the data, in Hz.

      cutoff (float, optional): Standard deviation cutoff for rejection. Data portions whose variance is larger
                                than this threshold relative to the calibration data are considered missing
                                data and will be removed. The most aggressive value that can be used without
                                losing too much EEG is 5. Default: 5.0.
      blocksize (int, optional): Block size for calculating the robust data covariance and thresholds, in samples;
                                 allows to reduce the memory and time requirements of the robust estimators by this
                                 factor. Default: 10. (Note: Memory-based dynamic calculation from MATLAB not implemented).
      B (np.ndarray, optional): Numerator coefficients of an IIR filter used for shaping the spectrum for artifact statistics.
                                Default: designed for `srate` with `yulewalk`, as asr_calibrate.m does.
      A (np.ndarray, optional): Denominator coefficients of an IIR filter used for shaping the spectrum for artifact statistics.
                                Default: designed for `srate` with `yulewalk`, as asr_calibrate.m does.
      window_len (float, optional): Window length in seconds for checking artifact content. Default: 0.5.
      window_overlap (float, optional): Window overlap fraction (0-1). Default: 0.66.
      max_dropout_fraction (float, optional): Maximum fraction (0-1) of windows subject to dropouts. Default: 0.1.
      min_clean_fraction (float, optional): Minimum fraction (0-1) of windows that must be clean. Default: 0.25.
      maxmem (int, optional): Maximum memory in MB (for very large data/many channels). Default: 64.
      useriemannian (str, optional): Option to use a Riemannian ASR variant. Can be set to 'calib' to use a Riemannian estimate
            at calibration time; this make somewhat different statistical tradeoffs than the default, resulting in a potentially
            different baseline rejection threshold; as a result it is suggested to visually check results and adjust
            the cutoff as needed. Default: None (disabled).
      compatibility (str, optional): MATLAB compatibility level.
        * 'standard' (default) aims for 5 significant digits compatibility and may apply
          slightly better numerical methods (e.g. using SOS filters for IIR filtering)
          that are not available in stock MATLAB and therefore not used in the ASR
          reference implementation.
        * 'max' aims for maximum compatibility with MATLAB's results, aiming to match
          results as closely as possible, perhaps trading off numerical robustness in
          turn. Note the effects will mostly likely be miniscule and the MATLAB ASR
          implementation is known to be highly robust.

    Returns
    -------
      dict: State dictionary containing calibration results ('M', 'T') and filter parameters ('B', 'A', 'sos', 'iir_state')
            needed for `asr_process`.
    """
    # Ensure X is a numpy array and C x S
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("Input data X must be a 2D array (channels x samples).")
    C, S = X.shape
    srate = float(srate)

    # Parameter defaults
    if cutoff is None:
        cutoff = 5.0
    if blocksize is None:
        blocksize = 10
    if maxmem is None:
        maxmem = 64  # in MB
    if window_len is None:
        window_len = 0.5
    if window_overlap is None:
        window_overlap = 0.66
    if max_dropout_fraction is None:
        max_dropout_fraction = 0.1
    if min_clean_fraction is None:
        min_clean_fraction = 0.25
    if compatibility is None:
        compatibility = 'standard'

    # there's no record of when or how this formula crept into the MATLAB code, but
    # to match it, we'll have to use it here as well
    blocksize = max(blocksize, math.ceil((C * C * S * 8 * 3 * 2) / (maxmem * (2**21))))

    # Spectral-shaping filter for the artifact statistics. asr_calibrate.m designs
    # this with yulewalk() at the dataset's own sampling rate; the frequency/amplitude
    # breakpoints below are its emphasis curve (attenuate the 3-13 Hz alpha/theta
    # range, emphasise DC drift and high-frequency muscle).
    if B is None or A is None:
        freqvals = np.concatenate((np.array([0, 2, 3, 13, 16, 40, min(80.0, srate / 2.0 - 1.0)]) * 2.0 / srate, [1.0]))
        amps = np.array([3, 0.75, 0.33, 0.33, 1, 1, 3, 3], dtype=np.float64)
        if srate < 80:
            # min(80, srate/2-1) is already below 40 Hz here, so drop that breakpoint
            # rather than attenuating it.
            freqvals = np.delete(freqvals, freqvals.size - 3)
            amps = amps[:-1]
        B, A = yulewalk(8, freqvals, amps)

    # Ensure data is finite
    X[~np.isfinite(X)] = 0.0

    # Apply the signal shaping filter based on compatibility mode
    if compatibility == 'max':
        # Maximum MATLAB compatibility: use B/A form with lfilter
        # Initialize filter state to zeros (matching MATLAB's filter(..., [], 2))
        # For multi-channel data (C x S) filtering along axis=1, zi shape is (C, max(len(A), len(B)) - 1)
        zi = np.zeros((C, max(len(A), len(B)) - 1))
        Xf, iir_state = scipy.signal.lfilter(B, A, X, axis=1, zi=zi)
        sos = None  # Not used in this mode
    else:
        # Standard mode: use second-order sections (SOS) for numerical stability
        sos = scipy.signal.tf2sos(B, A)
        # Need initial state per channel: shape (n_sections, n_channels, 2)
        # (since the data are assumed to be zero-mean, use a zero state, as in MATLAB)
        zi = np.zeros((sos.shape[0], C, 2))
        Xf, iir_state = scipy.signal.sosfilt(sos, X, axis=1, zi=zi)

    if np.any(~np.isfinite(Xf)):
        raise RuntimeError(
            'The IIR filter diverged on your data. Please try using either '
            'a more conservative filter or removing some bad sections/channels from the calibration data.'
        )

    # Calculate the sample covariance matrices U (averaged in blocks of blocksize successive samples)
    # U will be shape (C, C, num_blocks)
    logger.info("Calculating blockwise covariances...")

    # Determine the number of blocks
    num_blocks = int(np.ceil(S / blocksize))
    U = np.zeros((C, C, num_blocks))
    block_starts = np.arange(0, S, blocksize)

    # Accumulate outer products in blocks for memory efficiency
    for k in range(blocksize):
        # Calculate indices for this step, avoiding going past the end
        range_indices = np.minimum(block_starts + k, S - 1)
        if range_indices.size == 0:
            continue  # Skip if no indices

        # Extract data for these indices
        X_k = Xf[:, range_indices]

        # Calculate and accumulate outer products
        outer_products = np.reshape(X_k, (C, 1, -1)) * np.reshape(X_k, (1, C, -1))

        # Add to U, ensuring shape alignment
        if outer_products.shape[2] < U.shape[2]:
            U[:, :, : outer_products.shape[2]] += outer_products
        else:
            U += outer_products

    # Average the accumulated covariances
    U /= blocksize

    # compute a robust average of the covariance matrices
    med = None
    if useriemannian in ('calib', 'all', True):
        logger.info("Calculating Riemannian geometric median covariance...")
        U = U.transpose(2, 0, 1)
        # small amount of shrinkage to prevent singularities
        U = cov_shrinkage(U, 1e-4, target='scaled-eye')
        med = cov_mean(U, robust=True)
    if med is None or np.any(np.isnan(med)):
        if med is not None:
            logger.warning(
                "Riemannian geometric median calculation resulted in NaNs. Using standard geometric median as fallback."
            )
        logger.info("Calculating robust geometric median covariance...")
        med = geometric_median(U.reshape(C * C, -1).T)
    if np.any(np.isnan(med)):
        logger.warning("Geometric median calculation resulted in NaNs. Using standard median as fallback.")
        med = np.median(U, axis=-1)

    # make sure median is reshaped back to matrix form
    M_robust = np.reshape(med, (C, C))

    # Get the mixing matrix M (matrix square root of the robust covariance)
    M = scipy.linalg.sqrtm(np.real(M_robust))
    M = np.real(M)  # Ensure M is real

    # ----- Calculate Thresholds -----
    # Window length for calculating thresholds
    N = int(round_mat(window_len * srate))
    if S < N:
        raise ValueError(f'Not enough calibration data. Need at least {N} samples, got {S}.')

    logger.info('Determining per-component thresholds...')

    # Eigendecomposition of M plus some massaging
    # to ensure reproducibility across platforms
    M = 0.5 * (M + M.T)  # Ensure symmetry
    D, V = np.linalg.eigh(M)  # eigh returns sorted eigenvalues
    V = canonicalize_signs(V)

    # Transform data into component space (using eigenvectors)
    X_transformed = np.abs(finite_matmul(Xf.T, V))  # Shape: (S, C)

    # Calculate window indices for RMS calculation
    step = N * (1.0 - window_overlap)
    if step <= 0:
        logger.warning("Window overlap >= 1, using step=1")
        step = 1
    window_starts = round_mat(np.arange(0, S - N, step)).astype(int)

    if len(window_starts) <= 1:
        raise ValueError(f'Not enough windows possible. Need length > {N}, got {S}.')

    # Create window indices matrix
    window_indices = window_starts[:, None] + np.arange(N)

    # Initialize arrays for mu and sigma
    mu = np.zeros(C)
    sig = np.zeros(C)

    # Calculate thresholds for each component
    for c in reversed(range(C)):
        comp_data = X_transformed[:, c] ** 2

        # Calculate RMS amplitude for each window
        rms_windows = np.sqrt(np.mean(comp_data[window_indices], axis=1))

        # Fit a distribution to the clean part
        try:
            mu_c, sig_c, _, _ = fit_eeg_distribution(
                rms_windows, min_clean_fraction=min_clean_fraction, max_dropout_fraction=max_dropout_fraction
            )
            mu[c] = mu_c
            sig[c] = sig_c
        except Exception as e:
            logger.warning(f"Distribution fitting failed for component {c}: {e}")
            mu[c] = np.nan
            sig[c] = np.nan

    # Check for NaN values and provide warning
    if np.any(np.isnan(mu)) or np.any(np.isnan(sig)):
        logger.warning("NaN values in threshold calculation. Results may be unreliable.")
        # Replace NaNs with reasonable values
        mu = np.nan_to_num(mu, nan=np.nanmedian(mu) if np.any(~np.isnan(mu)) else 1.0)
        sig = np.nan_to_num(sig, nan=np.nanmedian(sig) if np.any(~np.isnan(sig)) else 0.5)

    # Ensure sigma is non-negative
    sig = np.maximum(sig, 0)

    # Calculate threshold matrix T
    T = finite_matmul(np.diag(mu + cutoff * sig), V.T)

    logger.info('Thresholds calculation complete.')

    # Return the state dictionary
    state = {
        'M': M,  # Mixing matrix
        'T': T,  # Threshold matrix
        'B': B,  # Original filter coefficients (for reference)
        'A': A,
        'sos': sos,  # SOS filter representation for processing (None if compatibility='max')
        'iir_state': iir_state,  # Initial filter state
        'cov': None,  # Initial covariance buffer (will be set in process)
        'carry': None,  # Initial carry buffer (will be set in process)
        'last_R': None,  # Initial reconstruction matrix (will be set in process)
        'last_trivial': True,  # Initial trivial flag
        'useriemannian': useriemannian,  # Riemannian ASR variant option
        'compatibility': compatibility,  # Compatibility mode for IIR filtering
    }

    return state
