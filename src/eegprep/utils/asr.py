import logging
import math
from warnings import warn
import numpy as np
import scipy.signal
import scipy.linalg

from .stats import block_geometric_median, fit_eeg_distribution
from .sigproc import moving_average

logger = logging.getLogger(__name__)


def asr_calibrate(X, srate, cutoff=None, blocksize=None, B=None, A=None,
                  window_len=None, window_overlap=None, max_dropout_fraction=None,
                  min_clean_fraction=None, maxmem=None):
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
                                Default: Calculated using pre-computed values based on srate (approximating yulewalk).
      A (np.ndarray, optional): Denominator coefficients of an IIR filter used for shaping the spectrum for artifact statistics.
                                Default: Calculated using pre-computed values based on srate (approximating yulewalk).
      window_len (float, optional): Window length in seconds for checking artifact content. Default: 0.5.
      window_overlap (float, optional): Window overlap fraction (0-1). Default: 0.66.
      max_dropout_fraction (float, optional): Maximum fraction (0-1) of windows subject to dropouts. Default: 0.1.
      min_clean_fraction (float, optional): Minimum fraction (0-1) of windows that must be clean. Default: 0.25.
      maxmem (int, optional): Maximum memory in MB (for very large data/many channels). Default: 64.

    Returns:
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
    if cutoff is None: cutoff = 5.0
    if blocksize is None: blocksize = 10
    if maxmem is None: maxmem = 64  # in MB
    if window_len is None: window_len = 0.5
    if window_overlap is None: window_overlap = 0.66
    if max_dropout_fraction is None: max_dropout_fraction = 0.1
    if min_clean_fraction is None: min_clean_fraction = 0.25

    # there's no record of when or how this formula crept into the MATLAB code, but 
    # to match it, we'll have to use it here as well
    blocksize = max(blocksize, math.ceil((C*C*S*8*3*2)/(maxmem*(2**21))))

    # Default IIR filter coefficients (approximating MATLAB's yulewalk defaults)
    # Based on artifact_removal_legacy.py and asr_calibrate.m logic
    if B is None or A is None:
        sr_round = int(round(srate))
        if sr_round == 100:
            B = np.array([0.9314233528641650, -1.0023683814963549, -0.4125359862018213, 0.7631567476327510, 0.4160430392910331, -0.6549131038692215, -0.0372583518046807, 0.1916268458752655, 0.0462411971592346], dtype=np.float64)
            A = np.array([1.0000000000000000, -0.4544220180303844, -1.0007038682936749, 0.5374925521337940, 0.4905013360991340, -0.4861062879351137, -0.1995986490699414, 0.1830048420730026, 0.0457678549234644], dtype=np.float64)
        elif sr_round == 128:
            B = np.array([1.1027301639165037, -2.0025621813611867, 0.8942119516481342, 0.1549979524226999, 0.0192366904488084, 0.1782897770278735, -0.5280306696498717, 0.2913540603407520, -0.0262209802526358], dtype=np.float64)
            A = np.array([1.0000000000000000, -1.1042042046423233, -0.3319558528606542, 0.5802946221107337, -0.0010360013915635, 0.0382167091925086, -0.2609928034425362, 0.0298719057761086, 0.0935044692959187], dtype=np.float64)
        elif sr_round == 200:
            B = np.array([1.4489483325802353, -2.6692514764802775, 2.0813970620731115, -0.9736678877049534, 0.1054605060352928, -0.1889101692314626, 0.6111331636592364, -0.3616483013075088, 0.1834313060776763], dtype=np.float64)
            A = np.array([1.0000000000000000, -0.9913236099393967, 0.3159563145469344, -0.0708347481677557, -0.0558793822071149, -0.2539619026478943, 0.2473056615251193, -0.0420478437473110, 0.0077455718334464], dtype=np.float64)
        elif sr_round == 256:
            B = np.array([1.7587013141770287, -4.3267624394458641, 5.7999880031015953, -6.2396625463547508, 5.3768079046882207, -3.7938218893374835, 2.1649108095226470, -0.8591392569863763, 0.2569361125627988], dtype=np.float64)
            A = np.array([1.0000000000000000, -1.7008039639301735, 1.9232830391058724, -2.0826929726929797, 1.5982638742557307, -1.0735854183930011, 0.5679719225652651, -0.1886181499768189, 0.0572954115997261], dtype=np.float64)
        elif sr_round == 300:
             B = np.array([1.9153920676433143, -5.7748421104926795, 9.1864764859103936, -10.7350356619363630, 9.6423672437729007, -6.6181939699544277, 3.4219421494177711, -1.2622976569994351, 0.2968423019363821], dtype=np.float64)
             A = np.array([1.0000000000000000, -2.3143703322055491, 3.2222567327379434, -3.6030527704320621, 2.9645154844073698, -1.8842615840684735, 0.9222455868758080, -0.3103251703648485, 0.0634586449896364], dtype=np.float64)
        elif sr_round == 500:
            B = np.array([2.3133520086975823, -11.9471223009159130, 29.1067166493384340, -43.7550171007238190, 44.3385767452216370, -30.9965523846388000, 14.6209883020737190, -4.2743412400311449, 0.5982553583777899], dtype=np.float64)
            A = np.array([1.0000000000000000, -4.6893329084452580, 10.5989986701080210, -14.9691518101365230, 14.3320358399731820, -9.4924317069169977, 4.2425899618982656, -1.1715600975178280, 0.1538048427717476], dtype=np.float64)
        elif sr_round == 512:
             B = np.array([2.3275475636130865, -12.2166478485960430, 30.1632789058248850, -45.8009842020820410, 46.7261263011068880, -32.7796858196767220, 15.4623349612560630, -4.5019779685307473, 0.6242733481676324], dtype=np.float64)
             A = np.array([1.0000000000000000, -4.7827378944258703, 10.9780696236622980, -15.6795187888195360, 15.1281978667576310, -10.0632079834518220, 4.5014690636505614, -1.2394100873286753, 0.1614727510688058], dtype=np.float64)
        else:
            # Fallback if no precomputed filter matches or yulewalk is unavailable
            # Consider adding a call to a yulewalk implementation if available,
            # or raising a more specific error/warning.
            warn(f"No pre-computed spectral filter for srate {srate}. "
                 f"Using a simple default (may be suboptimal).")
            B = np.array([1.0, -1.0]) # Simple high-pass/difference filter as a basic fallback
            A = np.array([1.0])
            # Original MATLAB error:
            # error('asr_calibrate:NoYulewalk','The yulewalk() function was not found and there is no pre-computed spectral filter for your sampling rate...');

    # Ensure data is finite
    X[~np.isfinite(X)] = 0.0

    # Convert filter B, A to second-order sections (SOS) format for numerical stability
    sos = scipy.signal.tf2sos(B, A)

    # Apply the signal shaping filter and initialize the IIR filter state
    # Initialize filter state using sosfilt_zi and scale by initial data
    # zi_init = scipy.signal.sosfilt_zi(sos) # Shape (n_sections, 2)

    # Need initial state per channel: shape (n_sections, n_channels, 2)
    # (since the data are assumed to be zero-mean, use a zero state, as in MATLAB)
    zi = np.zeros((sos.shape[0], C, 2))

    # Filter the data
    Xf, iir_state = scipy.signal.sosfilt(sos, X, axis=1, zi=zi)

    if np.any(~np.isfinite(Xf)):
        raise RuntimeError('The IIR filter diverged on your data. Please try using either '
                           'a more conservative filter or removing some bad sections/channels from the calibration data.')

    # Calculate the sample covariance matrices U (averaged in blocks of blocksize successive samples)
    # U will be shape (C, C, num_blocks)
    print("Calculating blockwise covariances...")
    
    # Determine the number of blocks
    num_blocks = int(np.ceil(S / blocksize))
    U = np.zeros((C, C, num_blocks))
    block_starts = np.arange(0, S, blocksize)

    # Accumulate outer products in blocks for memory efficiency
    for k in range(blocksize):
        # Calculate indices for this step, avoiding going past the end
        range_indices = np.minimum(block_starts + k, S - 1)
        if range_indices.size == 0: continue # Skip if no indices

        # Extract data for these indices
        X_k = Xf[:, range_indices]
        
        # Calculate and accumulate outer products
        outer_products = np.reshape(X_k, (C, 1, -1)) * np.reshape(X_k, (1, C, -1))
        
        # Add to U, ensuring shape alignment
        if outer_products.shape[2] < U.shape[2]:
            U[:, :, :outer_products.shape[2]] += outer_products
        else:
            U += outer_products

    # Average the accumulated covariances
    U /= blocksize

    # Reshape for geometric median calculation
    U_reshaped = U.reshape(C * C, -1).T  # Shape: (num_blocks, C*C)

    # Calculate the geometric median of covariance matrices
    print("Calculating robust geometric median covariance...")
    med = block_geometric_median(U_reshaped)
    
    # Handle NaN cases (can happen with single observation or degenerate data)
    if np.any(np.isnan(med)):
        if U_reshaped.shape[0] == 1:
            med = np.median(U_reshaped, axis=0)
        else:
            warn("Geometric median calculation resulted in NaNs. Using standard median as fallback.")
            med = np.median(U_reshaped, axis=0)

    # Reshape median back to matrix form
    M_robust = np.reshape(med, (C, C))

    # Get the mixing matrix M (matrix square root of the robust covariance)
    M = scipy.linalg.sqrtm(np.real(M_robust))
    M = np.real(M)  # Ensure M is real

    # ----- Calculate Thresholds -----
    # Window length for calculating thresholds
    N = int(round(window_len * srate))
    if S < N:
        raise ValueError(f'Not enough calibration data. Need at least {N} samples, got {S}.')

    print('Determining per-component thresholds...')
    
    # Eigendecomposition of M
    D, V = np.linalg.eigh(M)  # eigh returns sorted eigenvalues
    
    # Transform data into component space (using eigenvectors)
    X_transformed = np.abs(Xf.T @ V)  # Shape: (S, C)
    
    # Calculate window indices for RMS calculation
    step = N * (1.0 - window_overlap)
    if step <= 0:
        warn("Window overlap >= 1, using step=1")
        step = 1
    window_starts = np.round(np.arange(0, S - N + 1, step)).astype(int)
    
    if len(window_starts) <= 1:
        raise ValueError(f'Not enough windows possible. Need length > {N}, got {S}.')
    
    # Create window indices matrix
    window_indices = window_starts[:, None] + np.arange(N)
    
    # Initialize arrays for mu and sigma
    mu = np.zeros(C)
    sig = np.zeros(C)
    
    # Calculate thresholds for each component
    for c in reversed(range(C)):
        comp_data = X_transformed[:, c]**2
        
        # Calculate RMS amplitude for each window
        rms_windows = np.sqrt(np.mean(comp_data[window_indices], axis=1))
        
        # Fit a distribution to the clean part
        try:
            mu_c, sig_c, _, _ = fit_eeg_distribution(
                rms_windows,
                min_clean_fraction=min_clean_fraction,
                max_dropout_fraction=max_dropout_fraction
            )
            mu[c] = mu_c
            sig[c] = sig_c
        except Exception as e:
            warn(f"Distribution fitting failed for component {c}: {e}")
            mu[c] = np.nan
            sig[c] = np.nan
    
    # Check for NaN values and provide warning
    if np.any(np.isnan(mu)) or np.any(np.isnan(sig)):
        warn("NaN values in threshold calculation. Results may be unreliable.")
        # Replace NaNs with reasonable values
        mu = np.nan_to_num(mu, nan=np.nanmedian(mu) if np.any(~np.isnan(mu)) else 1.0)
        sig = np.nan_to_num(sig, nan=np.nanmedian(sig) if np.any(~np.isnan(sig)) else 0.5)
    
    # Ensure sigma is non-negative
    sig = np.maximum(sig, 0)
    
    # Calculate threshold matrix T
    T = np.diag(mu + cutoff * sig) @ V.T
    
    print('done.')
    
    # Return the state dictionary
    state = {
        'M': M,                 # Mixing matrix
        'T': T,                 # Threshold matrix 
        'B': B,                 # Original filter coefficients (for reference)
        'A': A,
        'sos': sos,             # SOS filter representation for processing
        'iir_state': iir_state, # Initial filter state
        'cov': None,            # Initial covariance buffer (will be set in process)
        'carry': None,          # Initial carry buffer (will be set in process)
        'last_R': None,         # Initial reconstruction matrix (will be set in process)
        'last_trivial': True,   # Initial trivial flag
    }
    
    return state


def asr_process(data, srate, state, window_len=0.5, lookahead=None, step_size=32, max_dims=0.66, max_mem=None, use_gpu=False):
    """Processing function for the Artifact Subspace Reconstruction (ASR) method.

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

    Returns:
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
        max_dims_num = round(C * max_dims)
    else:
        max_dims_num = int(max_dims)
    
    # Number of samples in sliding window and lookahead
    N = round(window_len * srate)
    P = round(lookahead * srate)
    
    # Fix NaN and Inf values
    data[~np.isfinite(data)] = 0
    
    # Extract state variables
    M = state['M']                  # Mixing matrix
    T = state['T']                  # Threshold matrix
    sos = state['sos']              # SOS filter representation
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
        warn("Memory too low, increasing it (rejection block size now "
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
        print(f'Now cleaning data in {splits} blocks', end='',flush=True)
    
    # Process data in chunks
    for k in range(splits):
        # Calculate range for this chunk in the original data space
        chunk_start = int(np.floor(k * S / splits))
        chunk_end = int(min(S, np.floor((k + 1) * S / splits)))
        range_ = np.arange(chunk_start, chunk_end)
        
        if len(range_) == 0:
            continue
        
        # Get spectrally shaped data for statistics computation (range shifted by lookahead)
        Xfilt = X[:, range_ + P]
        
        # Filter the data window
        Xfilt, iir_state = scipy.signal.sosfilt(sos, Xfilt, axis=1, zi=iir_state)
        
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
        # Include original filter coefficients if present
        'B': state.get('B'),
        'A': state.get('A')
    }
    
    return outdata, outstate

