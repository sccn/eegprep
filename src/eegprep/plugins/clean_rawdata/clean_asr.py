"""EEG ASR (Artifact Subspace Reconstruction) cleaning utilities.

This module provides functions for running the Artifact Subspace Reconstruction method
on EEG data to remove artifacts.
"""

import inspect
import logging
import warnings
from typing import Dict, Any, Literal, Union, Tuple, Optional, overload
from copy import deepcopy

import math
import numpy as np

from .asr_calibrate import ASRCalibrationError, asr_calibrate
from .asr_calibrate_r import asr_calibrate_r
from .asr_process import asr_process
from .asr_process_r import asr_process_r
from .asr_state import STATE_FORMAT, check_state
from .clean_windows import clean_windows
from ...functions.miscfunc.misc import round_mat

logger = logging.getLogger(__name__)


@overload
def clean_asr(
    EEG: Dict[str, Any],
    cutoff: float = 5.0,
    window_len: Optional[float] = None,
    step_size: Optional[int] = None,
    max_dims: float = 0.66,
    ref_maxbadchannels: Union[float, str, None, np.ndarray] = 0.075,
    ref_tolerances: Union[Tuple[float, float], str, None] = (-3.5, 5.5),
    ref_wndlen: Union[float, str, None] = 1.0,
    use_gpu: bool = False,
    useriemannian: Optional[str] = None,
    maxmem: Optional[int] = 64,
    *,
    return_state: Literal[False] = False,
    state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]: ...


@overload
def clean_asr(
    EEG: Dict[str, Any],
    cutoff: float = 5.0,
    window_len: Optional[float] = None,
    step_size: Optional[int] = None,
    max_dims: float = 0.66,
    ref_maxbadchannels: Union[float, str, None, np.ndarray] = 0.075,
    ref_tolerances: Union[Tuple[float, float], str, None] = (-3.5, 5.5),
    ref_wndlen: Union[float, str, None] = 1.0,
    use_gpu: bool = False,
    useriemannian: Optional[str] = None,
    maxmem: Optional[int] = 64,
    *,
    return_state: Literal[True],
    state: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]: ...


def clean_asr(
    EEG: Dict[str, Any],
    cutoff: float = 5.0,
    window_len: Optional[float] = None,
    step_size: Optional[int] = None,
    max_dims: float = 0.66,
    ref_maxbadchannels: Union[float, str, None, np.ndarray] = 0.075,
    ref_tolerances: Union[Tuple[float, float], str, None] = (-3.5, 5.5),
    ref_wndlen: Union[float, str, None] = 1.0,
    use_gpu: bool = False,
    useriemannian: Optional[str] = None,
    maxmem: Optional[int] = 64,
    *,
    return_state: bool = False,
    state: Optional[Dict[str, Any]] = None,
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Run the Artifact Subspace Reconstruction (ASR) method on EEG data.

    This is an automated artifact rejection function that ensures that the data
    contains no events that have abnormally strong power; the subspaces on which
    those events occur are reconstructed (interpolated) based on the rest of the
    EEG signal during these time periods. A calibration recording that is too
    short for the generalized-Gaussian fit still drops high-power windows; the
    window scale is then the median and MAD of each channel's window RMS.
    EEGLAB's fit throws in that situation and calibrates on every sample.
    Two or three calibration RMS windows make a fit width round to zero and
    raise ``ASRCalibrationError``. Fewer than two calibration RMS windows, and
    a calibration shorter than one window, raise ``ASRCalibrationError`` too.
    The calibration is the reference ``clean_windows`` kept. A longer recording
    is not substituted when that reference has fewer samples than one 0.5 s RMS
    window. Those errors name ``asr_calibrate`` or ``asr_calibrate_r`` and
    whether to record more data or set ``BurstCriterion`` off. EEGLAB raises an
    error that asks for different ``BurstCriterionRefTolerances`` when the
    reference is shorter than one window or yields fewer than two. A sampling
    rate whose default spectral breakpoints are not monotonic raises
    ``ASRCalibrationError`` as well.

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
        useriemannian (str, optional): Riemannian ASR variant. ``None`` or ``'off'`` uses Euclidean ASR.
            ``'calib'`` uses a Riemannian covariance estimate at calibration time and the Euclidean processor.
            ``True`` or ``'on'`` runs Riemannian calibration and Riemannian processing. Default: None.
        maxmem (Optional[int], optional): Memory budget in MB. Calibration grows its covariance block size from it
            and processing splits the data into blocks sized by it, in every variant, and the block count changes
            the cleaned data. Default: 64. A budget that is ``None`` or smaller than
            ``channels * channels * lookahead_samples * 24`` bytes is replaced by half of the free memory, so the
            block count and the output then differ between runs and machines. Keep it at or above that value, which
            64 MB does not for 128 channels at 1 kHz, when the cleaned data must be reproduced.
        return_state (bool, optional): Also return the calibration state. The state is checked as
            ``asr_state.check_state`` does. Keyword only. Default: False.
        state (dict, optional): A state returned earlier by ``return_state=True``, or read back with
            ``eegprep.plugins.clean_rawdata.asr_state.state_from_json``. It is checked as ``check_state`` does.
            Reference selection and calibration are skipped, and the data is processed with that state and the
            processing arguments it records (``cutoff``, ``window_len``, ``step_size``, ``max_dims``,
            ``useriemannian``, ``maxmem``). The same EEG and state give identical data on one platform and
            numerical library stack; another library can differ in the last bits. A state whose sampling rate or
            channel labels differ from the EEG's is refused, and so is a ``cutoff``, ``window_len``,
            ``step_size``, ``max_dims``, ``ref_*``, ``useriemannian`` or ``maxmem`` beside it that is neither the
            argument's default nor the value the state records, because the state owns them. Keyword only.

    Returns
    -------
    Dict[str, Any] : The EEG dictionary with the 'data' field containing the cleaned data.
    With ``return_state=True``, ``(EEG, state)``: ``state`` is the calibration state as ``asr_calibrate`` (or
    ``asr_calibrate_r`` for full Riemannian ASR) returned it, before processing, plus ``srate``, ``channels`` (the
    channel labels in data order, ``None`` when the EEG has no ``chanlocs``), the processing arguments, and
    ``format``. ``asr_state.state_to_json`` writes it as plain JSON.

    Raises
    ------
    ImportError : If automatic calibration data selection is needed (`ref_maxbadchannels` is float) but `clean_windows` cannot be imported.
    ASRCalibrationError : The spectral shaping filter cannot be designed, the calibration cannot form two RMS windows, or a fit width rounds to a non-positive value.
    ValueError : If input arguments are invalid, or ``state`` does not fit the EEG or is not an ASR state.

    Warns
    -----
    UserWarning : With ``return_state=True`` or ``state=``, when the recorded ``maxmem`` is ``None`` or below the
        processing overhead, because the output then depends on the free memory of the machine.
    """
    if 'data' not in EEG or 'srate' not in EEG or 'nbchan' not in EEG:
        raise ValueError("EEG dictionary must contain 'data', 'srate', and 'nbchan'.")

    data = np.array(EEG['data'], dtype=np.float64, copy=True)
    srate = float(EEG['srate'])
    nbchan = int(EEG['nbchan'])
    C, S = data.shape

    if C != nbchan:
        logger.warning(f"Mismatch between EEG['nbchan'] ({nbchan}) and EEG['data'].shape[0] ({C}). Using shape[0].")
        nbchan = C  # Use the actual dimension from data
    channels = _channel_labels(EEG)

    supplied = state is not None
    if state is None:
        state = _calibrate_state(
            EEG,
            data,
            channels,
            cutoff=cutoff,
            window_len=window_len,
            step_size=step_size,
            max_dims=max_dims,
            ref_maxbadchannels=ref_maxbadchannels,
            ref_tolerances=ref_tolerances,
            ref_wndlen=ref_wndlen,
            useriemannian=useriemannian,
            maxmem=maxmem,
        )
        if return_state:
            check_state(state)
    else:
        check_state(state)
        _refuse_arguments_owned_by_state(
            state,
            cutoff=cutoff,
            window_len=window_len,
            step_size=step_size,
            max_dims=max_dims,
            ref_maxbadchannels=ref_maxbadchannels,
            ref_tolerances=ref_tolerances,
            ref_wndlen=ref_wndlen,
            useriemannian=useriemannian,
            maxmem=maxmem,
        )
        _check_state_fits(state, srate, channels, C)

    # --- Extrapolate Signal End ---
    # Required because asr_process needs lookahead data beyond the signal end
    # Based on: sig = [signal.data bsxfun(@minus,2*signal.data(:,end),signal.data(:,(end-1):-1:end-round(windowlen/2*signal.srate)))];
    window_len = state['window_len']
    N_extrap = int(round_mat(window_len / 2 * srate))
    if supplied or return_state:
        _warn_if_maxmem_cannot_reproduce(state['maxmem'], C, N_extrap)
    if N_extrap > 0:
        extrap_len = min(N_extrap, S - 1 if S > 1 else 0)
        if extrap_len > 0:
            # Indices from second-to-last sample back 'extrap_len' steps
            extrap_indices = np.arange(S - 2, S - extrap_len - 2, -1)
            # Reflect around the last sample: 2*last_sample - samples_before_last
            extrap_part = 2 * data[:, [-1]] - data[:, extrap_indices]
            sig = np.concatenate((data, extrap_part), axis=1)
        else:  # Not enough data to extrapolate
            sig = data
    else:  # No extrapolation needed
        sig = data

    # --- Process Signal using ASR ---
    logger.info('Applying ASR processing...')
    lookahead_sec = window_len / 2.0  # asr_process expects lookahead in seconds
    process = asr_process_r if state['useriemannian'] == 'all' else asr_process
    outdata, _ = process(
        sig,
        srate,
        state,
        window_len=window_len,
        lookahead=lookahead_sec,
        step_size=state['step_size'],
        max_dims=state['max_dims'],
        max_mem=state['maxmem'],
        use_gpu=use_gpu,  # Passed but ignored in current Python port
    )

    # --- Finalize ---
    # shift signal content back (to compensate for processing delay)
    outdata = outdata[:, :S]
    EEG = deepcopy(EEG)
    EEG['data'] = outdata
    logger.info('ASR cleaning finished.')

    return (EEG, state) if return_state else EEG


def _calibrate_state(
    EEG: Dict[str, Any],
    data: np.ndarray,
    channels: Optional[list],
    cutoff: float,
    window_len: Optional[float],
    step_size: Optional[int],
    max_dims: float,
    ref_maxbadchannels: Union[float, str, None, np.ndarray],
    ref_tolerances: Union[Tuple[float, float], str, None],
    ref_wndlen: Union[float, str, None],
    useriemannian: Optional[str],
    maxmem: Optional[int],
) -> Dict[str, Any]:
    """Select the calibration data, calibrate, and return the state ``clean_asr`` records."""
    useriemannian = _normalise_useriemannian(useriemannian)
    srate = float(EEG['srate'])
    C = data.shape[0]

    # --- Handle Defaults ---
    if window_len is None:
        window_len = max(0.5, 1.5 * C / srate)

    # --- Ensure Data Type ---

    # --- Determine Reference/Calibration Data ---
    ref_section_data = None
    if (
        isinstance(ref_maxbadchannels, (int, float))
        and isinstance(ref_tolerances, (tuple, list))
        and isinstance(ref_wndlen, (int, float))
    ):
        logger.info('Finding a clean section of the data for calibration...')
        try:
            # clean_windows is assumed to return the selected data array (C x S_clean)
            # It needs the EEG dict structure, similar to other clean_* funcs
            temp_EEG_for_cleanwin = deepcopy(EEG)
            temp_EEG_for_cleanwin['data'] = data  # ensure it has the float64 data
            cleaned_EEG, _ = clean_windows(temp_EEG_for_cleanwin, ref_maxbadchannels, ref_tolerances, ref_wndlen)
            ref_section_data = np.asarray(cleaned_EEG['data'], dtype=np.float64)
        except ValueError as e:
            # clean_windows raises ValueError for expected calibration-data problems
            # (empty data, window too small, not enough data for one window, or too
            # few windows for the distribution fit). Only those warrant the all-data
            # fallback; unexpected exceptions propagate so genuine bugs are not
            # masked as silently weaker ASR calibration.
            logger.warning(
                f"Could not automatically identify clean calibration data ({e}). "
                "Falling back to using the entire data for calibration."
            )
            ref_section_data = data
    elif (isinstance(ref_maxbadchannels, str) and ref_maxbadchannels.lower() == 'off') or ref_maxbadchannels is None:
        logger.info(f"Using the entire data for calibration ('ref_maxbadchannels' set to {ref_maxbadchannels!r}).")
        ref_section_data = data
    elif (isinstance(ref_tolerances, str) and ref_tolerances.lower() == 'off') or ref_tolerances is None:
        logger.info(f"Using the entire data for calibration ('ref_tolerances' set to {ref_tolerances!r}).")
        ref_section_data = data
    elif (isinstance(ref_wndlen, str) and ref_wndlen.lower() == 'off') or ref_wndlen is None:
        logger.info(f"Using the entire data for calibration ('ref_wndlen' set to {ref_wndlen!r}).")
        ref_section_data = data
    elif isinstance(ref_maxbadchannels, np.ndarray):
        logger.info("Using user-supplied data array for calibration.")
        ref_section_data = np.asarray(ref_maxbadchannels, dtype=np.float64)
        if ref_section_data.ndim != 2 or ref_section_data.shape[0] != C:
            raise ValueError(f"User-supplied calibration data must be a 2D array with shape ({C}, n_samples).")
    else:
        raise ValueError(
            f"Unsupported value or type for 'ref_maxbadchannels': {ref_maxbadchannels}. Must be float, None/'off', or numpy array."
        )

    # --- Calibrate ASR ---
    logger.info('Estimating ASR calibration statistics...')
    # The Python asr_calibrate uses its own defaults for blocksize, filters, etc.
    # We only pass the core parameters specified in the clean_asr call signature.
    try:
        if useriemannian == 'all':
            # clean_asr.m passes empty window arguments, so asr_calibrate_r keeps
            # its own defaults (0.1 s windows, overlap 0.5, clean fraction 0.3).
            calibration = asr_calibrate_r(ref_section_data, srate, cutoff=cutoff, maxmem=maxmem)
        else:
            calibration = asr_calibrate(
                ref_section_data, srate, cutoff=cutoff, maxmem=maxmem, useriemannian=useriemannian
            )
    except ASRCalibrationError:
        raise
    except ValueError as e:
        raise ValueError(f"ASR calibration failed: {e}") from e

    del ref_section_data  # Free memory

    # --- Prepare for Processing ---
    if step_size is None:
        step_size = int(math.floor(srate * window_len / 2))  # Samples
    elif not isinstance(step_size, int) and float(step_size).is_integer():
        step_size = int(step_size)  # a whole number of samples is recorded as an integer

    return {
        **calibration,
        'srate': srate,
        'channels': channels,
        'cutoff': cutoff,
        'window_len': window_len,
        'step_size': step_size,
        'max_dims': max_dims,
        'useriemannian': useriemannian,
        'maxmem': maxmem,
        'format': STATE_FORMAT,
    }


def _channel_labels(EEG: Dict[str, Any]) -> Optional[list]:
    """Channel labels in data order, or None when the EEG has no ``chanlocs``."""
    chanlocs = EEG.get('chanlocs')
    if chanlocs is None or len(chanlocs) == 0:
        return None
    return [str(chan.get('labels', '')) for chan in chanlocs]


def _check_state_fits(state: Dict[str, Any], srate: float, channels: Optional[list], nchan: int) -> None:
    """Refuse a state calibrated for another sampling rate or channel set."""
    if state['srate'] != srate:
        raise ValueError(f"state srate {state['srate']:g} Hz differs from the EEG's {srate:g} Hz.")
    if state['channels'] != channels:
        raise ValueError(
            "state channels differ from the EEG's channels in label or order, or only one of them has labels; "
            "a state from another montage would mix the wrong channels."
        )
    if np.shape(state['M']) != (nchan, nchan):
        raise ValueError(f"state was calibrated for {np.shape(state['M'])[0]} channels, the EEG has {nchan}.")


def _same_argument(name: str, passed: Any, allowed: Any) -> bool:
    if name == 'useriemannian':
        return _normalise_useriemannian(passed) == _normalise_useriemannian(allowed)
    if name == 'ref_tolerances' and isinstance(passed, (list, tuple)) and isinstance(allowed, (list, tuple)):
        return tuple(passed) == tuple(allowed)
    if isinstance(passed, np.ndarray) or isinstance(allowed, np.ndarray):
        return passed is allowed
    return bool(passed == allowed)


def _refuse_arguments_owned_by_state(state: Dict[str, Any], **passed: Any) -> None:
    """A state records its processing arguments, so a different one passed beside it is an error.

    An argument left at its default, or given the value the state records, is the same call.
    """
    parameters = inspect.signature(clean_asr).parameters
    for name, value in passed.items():
        allowed = [parameters[name].default]
        if name in state:
            allowed.append(state[name])
        if not any(_same_argument(name, value, candidate) for candidate in allowed):
            raise ValueError(
                f"{name}={value!r} cannot be combined with state=: the state records its own processing arguments."
            )


def _warn_if_maxmem_cannot_reproduce(maxmem: Optional[float], nchan: int, lookahead_samples: int) -> None:
    """Warn when asr_process would replace ``maxmem`` by half of the free memory.

    asr_process does that when the budget is None or smaller than the ``nchan * nchan * lookahead * 24`` bytes
    it reserves before it sizes its blocks; the block count and the output then depend on the machine.
    """
    overhead = nchan * nchan * lookahead_samples * 8 * 3
    if maxmem is not None and maxmem * 1024**2 >= overhead:
        return
    smallest = max(1, math.ceil(overhead / 1024**2))
    warnings.warn(
        f"maxmem={maxmem!r} is None or below the {smallest} MB that {nchan} channels with {lookahead_samples} "
        "lookahead samples need (channels * channels * lookahead samples * 24 bytes), so asr_process replaces it "
        "with half of the free memory and the block count, and with it the cleaned data, differs between runs "
        f"and machines. Use maxmem={smallest} or more for a state that reproduces its data.",
        UserWarning,
        stacklevel=3,
    )


def _normalise_useriemannian(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return 'all' if value else None
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {'', '0', 'false', 'no', 'none', 'off'}:
            return None
        if lower in {'calib', 'calibration'}:
            return 'calib'
        if lower in {'1', 'true', 'yes', 'on', 'all', 'process', 'riemannian'}:
            return 'all'
    raise ValueError("clean_asr useriemannian must be None, 'off', 'calib', or a full-processing flag such as True.")
