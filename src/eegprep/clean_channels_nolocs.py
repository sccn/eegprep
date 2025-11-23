"""EEG channel cleaning utilities without locations."""

from typing import *
import logging
import traceback

import numpy as np
from scipy.signal import filtfilt

from .utils import design_fir, design_kaiser, filtfilt_fast

logger = logging.getLogger(__name__)



def clean_channels_nolocs(
        EEG: Dict[str, Any],
        min_corr: float = 0.45,
        ignored_quantile: float = 0.1,
        window_len: float = 2.0,
        max_broken_time: float = 0.5,
        linenoise_aware: bool = True
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Remove channels with abnormal data from a continuous data set.

    This is an automated artifact rejection function which ensures that the data
    contains no channels that record only noise for extended periods of time. If
    channels with control signals are contained in the data these are usually also
    removed. The criterion is based on correlation: if a channel is decorrelated
    from all others (pairwise correlation < a given threshold), excluding a given
    fraction of most correlated channels -- and if this holds on for a sufficiently
    long fraction of the data set -- then the channel is removed.

    Args:
      EEG: Continuous data set, assumed to be appropriately high-passed (e.g. >0.5Hz or
        with a 0.5Hz - 2.0Hz transition band).
      min_corr: Minimum correlation between a channel and any other channel (in
        a short period of time) below which the channel is considered abnormal
        for that time period. Reasonable range: 0.4 (very lax) to 0.6 (quite aggressive).
      ignored_quantile: Fraction of channels that need to have at least the given
        MinCorrelation value w.r.t. the channel under consideration. This allows
        to deal with channels or small groups of channels that measure the same
        noise source. Reasonable range: 0.05 (rather lax) to 0.2 (very tolerant re
        disconnected/shorted channels).
      window_len: Length of the windows (in seconds) for which correlation is computed.
      max_broken_time: Maximum time (either in seconds or as fraction of the
        recording) during which a retained channel may be broken. Reasonable
        range: 0.1 (very aggressive) to 0.6 (very lax).
      linenoise_aware: Whether the operation should be performed in a line-noise
        aware manner. If enabled, the correlation measure will not be affected
        by the presence or absence of line noise (using a temporary notch filter).

    Returns:
      EEG: data set with bad channels removed
      removed_channels: boolean array indicating which channels were removed

    """
    Fs = EEG['srate']
    
    # Flag channels
    if 0 < max_broken_time < 1:
        max_broken_time = EEG['data'].shape[1] * max_broken_time
    else:
        max_broken_time = Fs * max_broken_time

    EEG['data'] = np.asarray(EEG['data'], dtype=np.float64)
    C, S, *_ = EEG['data'].shape
    window_len = window_len * Fs
    wnd = np.arange(int(window_len))
    offsets = np.arange(0, int(S - window_len), window_len, dtype=int)
    W = len(offsets)
    retained = np.arange(C - int(np.ceil(C * ignored_quantile)))

    # Optionally ignore both 50 and 60 Hz spectral components
    if linenoise_aware:
        Bwnd = design_kaiser(2 * 45 / Fs, 2 * 50 / Fs, 60, True)
        
        if Fs <= 110:
            raise ValueError('Sampling rate must be above 110 Hz')
        elif Fs <= 130:
            B = design_fir(
                len(Bwnd) - 1,
                2 * np.array([0, 45, 50, 55, Fs/2]) / Fs,
                [1, 1, 0, 1, 1],
                w=Bwnd
            )
        else:
            B = design_fir(
                len(Bwnd) - 1,
                2 * np.array([0, 45, 50, 55, 60, 65, Fs/2]) / Fs,
                [1, 1, 0, 1, 0, 1, 1],
                w=Bwnd
            )
        
        X = np.zeros((S, C))
        for c in range(C):
            X[:, c] = filtfilt_fast(B, 1.0, EEG['data'][c, :])
    else:
        X = EEG['data'].T

    # For each window, flag channels with too low correlation to any other channel
    flagged = np.zeros((C, W), dtype=bool)
    for o in range(W):
        window_data = X[offsets[o] + wnd, :]
        corrmat = np.abs(np.corrcoef(window_data, rowvar=False))
        sortcc = np.sort(corrmat, axis=0)
        flagged[:, o] = np.all(sortcc[retained, :] < min_corr, axis=0)

    # Mark channels for removal which have more flagged samples than the maximum
    removed_channels = np.sum(flagged, axis=1) * window_len > max_broken_time

    # Apply removal
    if np.all(removed_channels):
        logger.warning('All channels are flagged bad according to the used criterion: not removing anything.')
    elif np.any(removed_channels):
        logger.info('Now removing bad channels...')
        try:
            # Try to use pop_select if available
            from eegprep import pop_select
            EEG = pop_select(EEG, nochannel=list(np.where(removed_channels)[0]))
        except Exception as e:
            if isinstance(e, ImportError):
                logger.error('Apparently you do not have access to a pop_select() function.')
            else:
                logger.error('Could not select channels using EEGLAB\'s pop_select(); details: %s', str(e))
                logger.debug('Exception traceback:', exc_info=True)
            
            logger.info('Falling back to a basic substitute and dropping signal meta-data.')
            # Manual channel removal
            if len(EEG['chanlocs']) == EEG['data'].shape[0]:
                EEG['chanlocs'] = np.asarray([ch for i, ch in enumerate(EEG['chanlocs']) if not removed_channels[i]])
            # pop_select() by default truncates the data to float32, so we need to do the same
            EEG['data'] = np.asarray(EEG['data'], dtype=np.float32)
            EEG['data'] = EEG['data'][~removed_channels, :]
            EEG['nbchan'] = EEG['data'].shape[0]
            
            # Clear other fields
            for field in ['icawinv', 'icasphere', 'icaweights', 'icaact', 'stats', 'specdata', 'specicaact']:
                if field in EEG:
                    EEG[field] = np.array([])
        
        # Update clean_channel_mask
        if 'etc' in EEG and 'clean_channel_mask' in EEG['etc'] and sum(EEG['etc']['clean_channel_mask']) == len(removed_channels):
            mask = EEG['etc']['clean_channel_mask']
            EEG['etc']['clean_channel_mask'] = np.logical_and(mask, ~removed_channels[mask])
        else:
            if 'etc' not in EEG:
                EEG['etc'] = {}
            EEG['etc']['clean_channel_mask'] = ~removed_channels

    return EEG, removed_channels

