"""EEG channel cleaning utilities without locations."""

import logging
from typing import Any, Dict, Tuple

import numpy as np

from eegprep.plugins.firfilt.design import design_fir, design_kaiser

from .private.channel_removal import (
    copy_clean_channel_mask,
    remove_channels_without_pop_select,
    update_clean_channel_mask,
)
from .private.sigproc import filtfilt_fast

logger = logging.getLogger(__name__)


def clean_channels_nolocs(
    EEG: Dict[str, Any],
    min_corr: float = 0.45,
    ignored_quantile: float = 0.1,
    window_len: float = 2.0,
    max_broken_time: float = 0.5,
    linenoise_aware: bool = True,
    *,
    return_diagnostics: bool = False,
) -> Tuple[Dict[str, Any], np.ndarray] | Tuple[Dict[str, Any], np.ndarray, Dict[str, Any]]:
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
      return_diagnostics: When true, also return a dict of the correlations
        already computed. ``sorted_absolute_correlations`` has shape
        ``(channels, channels, windows)``: each column is that channel's
        absolute pairwise correlations sorted ascending, including its
        self-correlation. ``criteria['retained_correlation_rows']`` are the
        rows compared with ``min_corr`` after the most-correlated fraction is
        dropped. ``flagged`` is true where every retained row is below
        ``min_corr``. Channel indices and sample offsets are 0-based positions
        in the input. The returned boolean array is this detection result,
        including an all-true array when every channel is flagged and kept.
        ``diagnostics['removed_channels']`` is true only for channels deleted
        from the dataset. ``removal_suppressed`` explains that case.
        ``clean_channel_mask`` is true for original channels that remain and
        is left unchanged when nothing is removed.

    The input ``data`` array is replaced with float64. When nothing is
    removed, the same dataset object is returned inside the tuple and ``etc``
    is unchanged. When channels are removed, the returned dataset is new.
    The caller's channel count and ``etc`` stay as they were, while the
    caller's samples are the float64 array written before that copy. Copy
    the dataset with ``copy.deepcopy`` first when the original recording
    must be preserved.

    Returns
    -------
    EEG : data set with bad channels removed
    removed_channels : boolean array of channels that met the criterion
    diagnostics : dict, only when ``return_diagnostics`` is true
    """
    Fs = EEG['srate']
    window_len_seconds = float(window_len)
    max_broken_time_argument = float(max_broken_time)

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
            B = design_fir(len(Bwnd) - 1, 2 * np.array([0, 45, 50, 55, Fs / 2]) / Fs, [1, 1, 0, 1, 1], w=Bwnd)
        else:
            B = design_fir(
                len(Bwnd) - 1, 2 * np.array([0, 45, 50, 55, 60, 65, Fs / 2]) / Fs, [1, 1, 0, 1, 0, 1, 1], w=Bwnd
            )

        X = np.zeros((S, C))
        for c in range(C):
            X[:, c] = filtfilt_fast(B, 1.0, EEG['data'][c, :])
    else:
        X = EEG['data'].T

    flagged = np.zeros((C, W), dtype=bool)
    sorted_absolute_correlations = np.empty((C, C, W), dtype=float) if return_diagnostics else None
    for o in range(W):
        window_data = X[offsets[o] + wnd, :]
        corrmat = np.abs(np.corrcoef(window_data, rowvar=False))
        sortcc = np.sort(corrmat, axis=0)
        if sorted_absolute_correlations is not None:
            sorted_absolute_correlations[:, :, o] = sortcc
        flagged[:, o] = np.all(sortcc[retained, :] < min_corr, axis=0)

    # Mark channels for removal which have more flagged samples than the maximum
    removed_channels = np.sum(flagged, axis=1) * window_len > max_broken_time
    channels_removed = removed_channels
    removal_suppressed = None
    prior_clean_channel_mask = copy_clean_channel_mask(EEG) if return_diagnostics else None

    if np.all(removed_channels):
        removal_suppressed = 'All channels are flagged bad according to the used criterion: not removing anything.'
        logger.warning(removal_suppressed)
        channels_removed = np.zeros_like(removed_channels)
    elif np.any(removed_channels):
        logger.info('Now removing bad channels...')
        try:
            from eegprep import pop_select

            EEG = pop_select(EEG, nochannel=list(np.where(removed_channels)[0]))
        except Exception as e:
            if isinstance(e, ImportError):
                logger.error('Apparently you do not have access to a pop_select() function.')
            else:
                logger.error('Could not select channels using EEGLAB\'s pop_select(); details: %s', str(e))
                logger.debug('Exception traceback:', exc_info=True)

            logger.info('Falling back to a basic substitute and dropping signal meta-data.')
            EEG = remove_channels_without_pop_select(EEG, removed_channels)
        update_clean_channel_mask(EEG, removed_channels)

    if not return_diagnostics:
        return EEG, removed_channels
    return (
        EEG,
        removed_channels,
        {
            'index_base': 0,
            'criteria': {
                'min_corr': float(min_corr),
                'ignored_quantile': float(ignored_quantile),
                'window_len': window_len_seconds,
                'window_len_samples': float(window_len),
                'max_broken_time': max_broken_time_argument,
                'max_broken_time_samples': float(max_broken_time),
                'linenoise_aware': bool(linenoise_aware),
                'retained_correlation_rows': np.array(retained, dtype=int, copy=True),
            },
            'window_offsets': offsets,
            'window_length': int(wnd.size),
            'sorted_absolute_correlations': sorted_absolute_correlations,
            'flagged': flagged,
            'flagged_window_counts': np.sum(flagged, axis=1),
            'detected_channels': np.array(removed_channels, dtype=bool, copy=True),
            'removed_channels': np.array(channels_removed, dtype=bool, copy=True),
            'removal_suppressed': removal_suppressed,
            'prior_clean_channel_mask': prior_clean_channel_mask,
            'clean_channel_mask': copy_clean_channel_mask(EEG),
        },
    )
