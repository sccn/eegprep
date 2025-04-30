import logging
from typing import Dict, Any, Optional, Union, Tuple
from copy import deepcopy

import numpy as np

# Assuming these utilities exist and are correctly ported/placed
from .utils.asr import asr_calibrate, asr_process
from .clean_windows import clean_windows


logger = logging.getLogger(__name__)


def clean_asr(
    EEG: Dict[str, Any],
    cutoff: float = 5.0,
    window_len: Optional[float] = None,
    step_size: Optional[int] = None,
    max_dims: float = 0.66,
    ref_maxbadchannels: Union[float, str, np.ndarray] = 0.075,
    ref_tolerances: Union[Tuple[float, float], str] = (-3.5, 5.5),
    ref_wndlen: Union[float, str] = 1.0,
    use_gpu: bool = False,
    useriemannian: bool = False,
    maxmem: Optional[int] = 64
) -> Dict[str, Any]:
    """Run the Artifact Subspace Reconstruction (ASR) method on EEG data.

    This is an automated artifact rejection function that ensures that the data
    contains no events that have abnormally strong power; the subspaces on which
    those events occur are reconstructed (interpolated) based on the rest of the
    EEG signal during these time periods.

    Args:
        EEG (Dict[str, Any]): EEG data structure. Expected fields:
            'data' (np.ndarray): Channels x Samples matrix.
            'srate' (float): Sampling rate in Hz.
            'nbchan' (int): Number of channels.
            It's assumed the data is zero-mean (e.g., high-pass filtered).
        cutoff (float, optional): Standard deviation cutoff for rejection. Data portions whose variance
                                   is larger than this threshold relative to the calibration data are
                                   considered artifactual and removed. Aggressive: 3, Default: 5, Conservative: 20.
        window_len (float, optional): Length of the statistics window in seconds. Should not be much longer
                                      than artifact timescale. Samples in window should be >= 1.5x channels.
                                      Default: max(0.5, 1.5 * nbchan / srate).
        step_size (int, optional): Step size for processing in samples. Reconstruction matrix updated every
                                   `step_size` samples. If None, defaults to window_len / 2 samples.
        max_dims (float, optional): Maximum dimensionality/fraction of dimensions to reconstruct. Default: 0.66.
        ref_maxbadchannels (Union[float, str, np.ndarray], optional): Parameter for automatic calibration data selection.
            float: Max fraction (0-1) of bad channels tolerated in a window for it to be used as calibration data. Lower is more aggressive (e.g., 0.05). Default: 0.075.
            'off': Use all data for calibration. Assumes artifact contamination < ~30-50%.
            np.ndarray: Directly provides the calibration data (channels x samples).
        ref_tolerances (Union[Tuple[float, float], str], optional): Power tolerances (lower, upper) in SDs from robust EEG power
                                    for a channel to be considered 'bad' during calibration data selection. Default: (-3.5, 5.5). Use 'off' to disable.
        ref_wndlen (Union[float, str], optional): Window length in seconds for calibration data selection granularity. Default: 1.0. Use 'off' to disable.
        use_gpu (bool, optional): Whether to try using GPU (requires compatible hardware and libraries, currently ignored). Default: False.
        useriemannian (bool, optional): Whether to use Riemannian ASR variant (NOT IMPLEMENTED). Default: False.
        maxmem (Optional[int], optional): Maximum memory in MB (passed to asr_calibrate/process, but chunking based on it is not implemented in Python port). Default: 64.

    Returns:
        Dict[str, Any]: The EEG dictionary with the 'data' field containing the cleaned data.

    Raises:
        NotImplementedError: If useriemannian is True.
        ImportError: If automatic calibration data selection is needed (`ref_maxbadchannels` is float) but `clean_windows` cannot be imported.
        ValueError: If input arguments are invalid or calibration fails critically.
    """
    if useriemannian:
        raise NotImplementedError("The Riemannian ASR variant is not implemented in this Python port.")

    if 'data' not in EEG or 'srate' not in EEG or 'nbchan' not in EEG:
        raise ValueError("EEG dictionary must contain 'data', 'srate', and 'nbchan'.")

    data = np.asarray(EEG['data'], dtype=np.float64)
    srate = float(EEG['srate'])
    nbchan = int(EEG['nbchan'])
    C, S = data.shape

    if C != nbchan:
         logger.warning(f"Mismatch between EEG['nbchan'] ({nbchan}) and EEG['data'].shape[0] ({C}). Using shape[0].")
         nbchan = C # Use the actual dimension from data

    # --- Handle Defaults ---
    if window_len is None:
        window_len = max(0.5, 1.5 * nbchan / srate)

    # --- Ensure Data Type ---
    # Already done with np.asarray above

    # --- Determine Reference/Calibration Data ---
    ref_section_data = None
    if isinstance(ref_maxbadchannels, (int, float)) and isinstance(ref_tolerances, (tuple, list)) and isinstance(ref_wndlen, (int, float)):
        logger.info('Finding a clean section of the data for calibration...')
        try:
            # clean_windows is assumed to return the selected data array (C x S_clean)
            # It needs the EEG dict structure, similar to other clean_* funcs
            temp_EEG_for_cleanwin = deepcopy(EEG)
            temp_EEG_for_cleanwin['data'] = data # ensure it has the float64 data
            cleaned_EEG, _ = clean_windows(temp_EEG_for_cleanwin, ref_maxbadchannels, ref_tolerances, ref_wndlen)
            ref_section_data = np.asarray(cleaned_EEG['data'], dtype=np.float64)
            if ref_section_data.size == 0 or ref_section_data.shape[1] == 0:
                logger.warning("clean_windows returned no data. Falling back to using all data for calibration.")
                ref_section_data = data
        except Exception as e:
            logger.error(f"An error occurred during clean_windows: {e}")
            logger.warning("Could not automatically identify clean calibration data. Falling back to using the entire data for calibration.")
            ref_section_data = data
    elif isinstance(ref_maxbadchannels, str) and ref_maxbadchannels.lower() == 'off':
        logger.info("Using the entire data for calibration ('ref_maxbadchannels' set to 'off').")
        ref_section_data = data
    elif isinstance(ref_tolerances, str) and ref_tolerances.lower() == 'off':
        logger.info("Using the entire data for calibration ('ref_tolerances' set to 'off').")
        ref_section_data = data
    elif isinstance(ref_wndlen, str) and ref_wndlen.lower() == 'off':
        logger.info("Using the entire data for calibration ('ref_wndlen' set to 'off').")
        ref_section_data = data
    elif isinstance(ref_maxbadchannels, np.ndarray):
        logger.info("Using user-supplied data array for calibration.")
        ref_section_data = np.asarray(ref_maxbadchannels, dtype=np.float64)
        if ref_section_data.ndim != 2 or ref_section_data.shape[0] != C:
             raise ValueError(f"User-supplied calibration data must be a 2D array with shape ({C}, n_samples).")
    else:
        raise ValueError(f"Unsupported value or type for 'ref_maxbadchannels': {ref_maxbadchannels}. Must be float, 'off', or numpy array.")

    # --- Calibrate ASR ---
    logger.info('Estimating ASR calibration statistics...')
    # The Python asr_calibrate uses its own defaults for blocksize, filters, etc.
    # We only pass the core parameters specified in the clean_asr call signature.
    try:
        state = asr_calibrate(ref_section_data, srate, cutoff=cutoff, maxmem=maxmem)
    except ValueError as e:
         # Catch specific errors like not enough calibration data
         raise ValueError(f"ASR calibration failed: {e}")
    except Exception as e:
         # Catch unexpected errors during calibration
         logger.exception("An unexpected error occurred during ASR calibration.")
         raise RuntimeError(f"ASR calibration failed unexpectedly: {e}")

    del ref_section_data # Free memory

    # --- Prepare for Processing ---
    if step_size is None:
        step_size = int(round(srate * window_len / 2)) # Samples

    # --- Extrapolate Signal End ---
    # Required because asr_process needs lookahead data beyond the signal end
    # Based on: sig = [signal.data bsxfun(@minus,2*signal.data(:,end),signal.data(:,(end-1):-1:end-round(windowlen/2*signal.srate)))];
    N_extrap = int(round(window_len / 2 * srate))
    if N_extrap > 0:
         # Calculate indices for reflection, handling edge case where N_extrap >= S-1
        extrap_len = min(N_extrap, S - 1 if S > 1 else 0)
        if extrap_len > 0:
            # Indices from second-to-last sample back 'extrap_len' steps
            extrap_indices = np.arange(S - 2, S - extrap_len - 2, -1)
            # Reflect around the last sample: 2*last_sample - samples_before_last
            extrap_part = 2 * data[:, [-1]] - data[:, extrap_indices]
            sig = np.concatenate((data, extrap_part), axis=1)
        else: # Not enough data to extrapolate
             sig = data
    else: # No extrapolation needed
        sig = data


    # --- Process Signal using ASR ---
    logger.info('Applying ASR processing...')
    lookahead_sec = window_len / 2.0 # asr_process expects lookahead in seconds
    outdata, _ = asr_process(
        sig,
        srate,
        state,
        window_len=window_len,
        lookahead=lookahead_sec,
        step_size=step_size,
        max_dims=max_dims,
        max_mem=maxmem,
        use_gpu=use_gpu # Passed but ignored in current Python port
    )

    # --- Finalize ---
    # shift signal content back (to compensate for processing delay)
    outdata = outdata[:, :S]
    EEG['data'] = outdata
    logger.info('ASR cleaning finished.')

    return EEG
