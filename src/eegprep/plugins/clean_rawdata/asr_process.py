"""Artifact Subspace Reconstruction (ASR) utilities."""

import logging

import numpy as np
import scipy.signal

from ...functions.miscfunc.misc import round_mat
from .private.sigproc import moving_average

logger = logging.getLogger(__name__)


def asr_process(data, srate, state, window_len=0.5, lookahead=None, step_size=32, max_dims=0.66, max_mem=None, use_gpu=False):
    """Process data using the Artifact Subspace Reconstruction (ASR) method.

    CleanedData, State = asr_process(Data, SamplingRate, State, WindowLength, LookAhead, StepSize, MaxDimensions, MaxMemory, UseGPU)

    This function is used to clean multi-channel signal using the ASR method. The required inputs are
    the data matrix, the sampling rate of the data, and the filter state (as initialized by
    asr_calibrate or from the previous call to asr_process).

    Args:
        data (np.ndarray): Chunk of data to process [#channels x #samples]. Assumed to be
                           a continuation of previous data if 'state' is provided.
                           Data should be zero-mean (e.g., high-pass filtered).
        srate (float): Sampling rate of the data in Hz.
        state (dict): State dictionary from asr_calibrate or previous asr_process call.
                      Contains M, T, sos, iir_state, cov, carry, last_R, last_trivial.
        window_len (float, optional): Length of the statistics window in seconds. Should not be much
                                     longer than artifact time scale. Min samples: 1.5x channels. Default: 0.5.
        lookahead (float, optional): Look-ahead amount in seconds (causes delay). Recommended: window_len/2.
                                     Range [0, window_len/2]. Default: window_len/2.
        step_size (int, optional): Update statistics every this many samples. Larger is faster.
                                  Max: window_len * srate. Default: 32.
        max_dims (float or int, optional): Maximum dimensions/fraction of dimensions to remove.
                                         Default: 0.66 (fraction).
        max_mem (int, optional): Maximum memory in MB for processing large chunks. Process in one block if None.
                                 Default: None.
        use_gpu (bool, optional): Whether to use GPU (not implemented). Default: False.

    Returns
    -------
        tuple: (outdata, outstate)
            outdata (np.ndarray): Cleaned data chunk (delayed by lookahead).
            outstate (dict): Updated state dictionary for subsequent calls.
    """
    # Check and sanitize data
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array (channels x samples).")
    C, S = data.shape

    if S == 0:
        return data, state  # Return empty data as is

    # Parameter handling
    if lookahead is None:
        lookahead = window_len / 2
    if max_mem is None:
        # use at most half of available memory
        import psutil
        max_mem = psutil.virtual_memory().free / 1024**2 / 2

    # Ensure window length is adequate
    window_len = max(window_len, 1.5 * C / srate)

    # Convert max_dims to actual number if given as fraction
    if max_dims < 1:
        max_dims_num = int(round_mat(C * max_dims))
    else:
        max_dims_num = int(max_dims)

    # Number of samples in sliding window and lookahead
    N = int(round_mat(window_len * srate))
    P = int(round_mat(lookahead * srate))

    # Fix NaN and Inf values
    data[~np.isfinite(data)] = 0

    # Extract state variables
    M = state['M']                  # Mixing matrix
    T = state['T']                  # Threshold matrix
    sos = state.get('sos')          # SOS filter representation (None if compatibility='max')
    b = state.get('B')              # Filter numerator coefficients
    a = state.get('A')              # Filter denominator coefficients
    compatibility = state.get('compatibility', 'standard')  # Compatibility mode
    iir_state = state.get('iir_state')  # Filter state
    carry = state.get('carry')      # Carry buffer (previous lookahead data)
    cov = state.get('cov')          # Covariance state (MovAvgState or None)
    # If cov is from an older run and is not a MovAvgState, reset it
    if cov is not None and not hasattr(cov, 'buf'):
        cov = None
    last_R = state.get('last_R')    # Last reconstruction matrix
    last_trivial = state.get('last_trivial', True)  # Was last step trivial (no artifacts)

    # Initialize prior filter state by extrapolating available data into the past
    if carry is None:
        ind = np.mod(np.arange(P + 1, 1, -1) - 1, S)
        carry = 2 * data[:, [0]] - data[:, ind]

    # Prepend the carry buffer to the data
    X = np.concatenate((carry, data), axis=1)

    # Calculate number of splits for memory management

    if max_mem*1024*1024 - C*C*P*8*3 < 0:
        logger.warning("Memory too low, increasing it (rejection block size now "
             "depends on available memory so it might not be fully reproducible)...")
        import psutil
        max_mem = psutil.virtual_memory().free / 1024**2 / 2
        if max_mem*1024*1024 - C*C*P*8*3 < 0:
            raise RuntimeError('Not enough memory')

    # Calculate memory bytes needed (following reference implementation formula)
    bytes_needed = (C * C * S * 8 * 8 +
                    C * C * 8 * S / step_size +
                    C * S * 8 * 2 +
                    S * 8 * 5)

    # Available memory in bytes (subtract fixed overhead)
    mem_available = max_mem * 1024**2 - C * C * P * 8 * 3
    mem_available = max(mem_available, 1)  # Ensure positive

    # Number of splits needed
    splits = int(np.ceil(bytes_needed / mem_available))
    # Cap at reasonable value
    splits = min(splits, 10000)

    if splits > 1:
        logger.info(f'Cleaning data in {splits} blocks')

    # Process data in chunks
    for k in range(splits):
        # Calculate range for this chunk in the original data space
        chunk_start = int(np.floor(k * S / splits))
        chunk_end = int(min(S, np.floor((k + 1) * S / splits)))
        range_ = np.arange(chunk_start, chunk_end)

        if len(range_) == 0:
            continue

        # Get spectrally shaped data for statistics computation (range shifted by lookahead)
        Xraw = X[:, range_ + P]

        # Filter the data window based on compatibility mode
        if compatibility == 'max':
            # Maximum MATLAB compatibility: use B/A form with lfilter
            Xfilt, iir_state = scipy.signal.lfilter(b, a, Xraw, axis=1, zi=iir_state)
        else:
            # Standard mode: use SOS form
            Xfilt, iir_state = scipy.signal.sosfilt(sos, Xraw, axis=1, zi=iir_state)

        # Calculate per‑sample covariance vectors and compute the running mean
        # covariance using the stateful `moving_average` implementation that
        # replicates MATLAB's `moving_average` helper. This yields a smoothed
        # covariance estimate for *every* sample and updates the internal
        # circular buffer/state stored in `cov`.
        Xcov_sample = np.reshape(
            np.reshape(Xfilt, (C, 1, -1)) * np.reshape(Xfilt, (1, C, -1)),
            (C * C, -1)
        )

        # Running mean over a window of N samples (along the last / time axis)
        Xcov_filtered, cov = moving_average(Xcov_sample, N=N, axis=1, Z=cov, init=0)

        # Determine points at which to update the reconstruction matrix
        update_at = np.arange(step_size, Xfilt.shape[1] + step_size - 1, step_size, dtype=int)
        update_at = np.minimum(update_at, Xfilt.shape[1])

        # If there is no previous R, initialize at first sample
        if last_R is None:
            update_at = np.insert(update_at, 0, 1)
            last_R = np.eye(C)

        update_at -= 1 # prepare for 0-based indexing

        # Extract the covariance matrices at our update points (already
        # averaged by the moving window) and reshape to C × C × #updates.
        Xcov_matrices = np.reshape(Xcov_filtered[:, update_at], (C, C, len(update_at)))

        # Process each update point
        last_n = -1  # MATLAB uses 1‑based indexing; align so first sample is included
        for j, n in enumerate(update_at):
            # Eigendecomposition to find potential artifact components
            try:
                D, V = np.linalg.eigh(Xcov_matrices[:, :, j])
                # Sort in ascending order (eigh already does this)
                # D and V are already sorted in ascending order by eigh
            except np.linalg.LinAlgError:
                # Fallback if eigendecomposition fails
                logger.warning(f"Eigendecomposition failed at update point {j}. Using identity matrix.")
                D, V = np.ones(C), np.eye(C)

            # Determine which components to keep (variance below threshold or not admissible for rejection)
            try:
                thresholds = np.sum((T @ V)**2, axis=0)
                keep = (D < thresholds) | (np.arange(C) < (C - max_dims_num))
                trivial = np.all(keep)
            except Exception as e:
                logger.error(f"Error in component selection: {e}")
                keep = np.ones(C, dtype=bool)
                trivial = True

            # Update the reconstruction matrix R
            if not trivial:
                try:
                    # Following reference implementation:
                    # Get V[:, keep] equivalent by multiplying V by a diagonal selection matrix
                    keep_mask = keep[np.newaxis, :]  # Make column vector
                    A = V.T @ M  # V.T × M
                    masked_A_T = keep_mask * A.T  # Zero out rows where keep is False
                    Q = masked_A_T.T  # Back to original orientation

                    # Calculate reconstruction matrix
                    Z = np.linalg.pinv(Q)
                    R = np.real(M @ Z @ V.T)
                except np.linalg.LinAlgError:
                    logger.warning(f"Failed to calculate inverse at update point {j}. Using identity matrix.")
                    R = np.eye(C)
                    trivial = True
            else:
                R = np.eye(C)

            # Apply reconstruction to data
            if not trivial or not last_trivial:
                # Get subrange of data to process
                subrange = range(last_n + 1, n + 1)
                if len(subrange) > 0:
                    # Calculate blend coefficients (raised cosine)
                    blend = (1 - np.cos(np.pi * np.arange(1, len(subrange) + 1) / len(subrange))) / 2

                    # Extract data segment to process (from extended data X)
                    idx_in_X = range_[subrange]
                    segment = X[:, idx_in_X]

                    # Apply blended reconstruction
                    X[:, idx_in_X] = (blend * (R @ segment) +
                                      (1 - blend) * (last_R @ segment))

            # Update state for next iteration
            last_n = n
            last_R = R
            last_trivial = trivial

        if splits > 1 and k % 10 == 0:
            logger.debug(f'Processing block {k+1}/{splits}')

    if splits > 1:
        logger.info('Finished cleaning.')

    # Update the carry buffer for next call (last P samples)
    new_carry = X[:, -P:] if X.shape[1] >= P else X

    # Return cleaned data (without the lookahead portion)
    outdata = X[:, P:P+S]

    # Update state dictionary
    outstate = {
        'M': M,
        'T': T,
        'sos': sos,
        'iir_state': iir_state,
        'cov': cov,
        'carry': new_carry,
        'last_R': last_R,
        'last_trivial': last_trivial,
        # Include original filter coefficients and compatibility mode
        'B': b,
        'A': a,
        'compatibility': compatibility,
        'useriemannian': state.get('useriemannian')
    }

    return outdata, outstate
