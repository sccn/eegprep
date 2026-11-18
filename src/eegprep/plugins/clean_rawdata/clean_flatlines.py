"""EEG flatline channel removal utilities."""

from typing import Any, Dict
import logging

import numpy as np

from .private.channel_removal import (
    copy_clean_channel_mask,
    remove_channels_without_pop_select,
    update_clean_channel_mask,
)

logger = logging.getLogger(__name__)


def clean_flatlines(
    EEG: Dict[str, Any],
    max_flatline_duration: float = 5.0,
    max_allowed_jitter: float = 20.0,
    *,
    return_diagnostics: bool = False,
) -> Dict[str, Any] | tuple[Dict[str, Any], Dict[str, Any]]:
    """Remove (near-) flat-lined channels.

    This is an automated artifact rejection function which ensures that
    the data contains no flat-lined channels.

    Args:
      EEG: the continuous-time EEG data structure
      max_flatline_duration: maximum tolerated flatline duration. In seconds.
        If a channel has a longer flatline than this, it will be considered
        abnormal.
      max_allowed_jitter: maximum tolerated jitter during flatlines. As a
        multiple of epsilon.
      return_diagnostics: When true, return ``(EEG, diagnostics)`` instead of
        ``EEG``. ``detected_channels`` is true where a flat stretch exceeds
        the duration. ``removed_channels`` is true only for channels deleted
        from the dataset. When every channel is flagged, nothing is removed,
        ``removal_suppressed`` says why, and ``clean_channel_mask`` is left
        as it was. That mask is true for original channels that remain.
        Channel indices are 0-based positions in the input.

    When no channel is removed, the same dataset object is returned and its
    samples and ``etc`` are unchanged. When some channels are removed, the
    returned dataset is new and the dataset that was passed in keeps its
    channel count, samples, and ``etc``. Copy the dataset with
    ``copy.deepcopy`` before the call when the pre-call recording must be
    preserved.

    Returns
    -------
    EEG : the EEG data structure with flatlined channels removed.
    diagnostics : dict, only when ``return_diagnostics`` is true

    Example:
      EEG = clean_flatlines(EEG)
    """
    X = EEG['data']
    max_duration = max_flatline_duration * EEG['srate']
    max_jitter = max_allowed_jitter * np.finfo(np.float64).eps

    # flag channels
    removed_channels = np.zeros(X.shape[0], dtype=bool)
    for i in range(X.shape[0]):
        flat = np.pad(np.abs(np.diff(X[i, :])) < max_jitter, 1)
        flat_intervals = np.reshape(np.where(np.diff(flat) > 0)[0], (-1, 2))
        if flat_intervals.shape[0] > 0:
            if np.max(flat_intervals[:, 1] - flat_intervals[:, 0]) > max_duration:
                removed_channels[i] = True

    channels_removed = removed_channels
    removal_suppressed = None
    prior_clean_channel_mask = copy_clean_channel_mask(EEG) if return_diagnostics else None
    if np.all(removed_channels):
        removal_suppressed = 'All channels have a flat-line portion; not removing anything.'
        logger.warning(removal_suppressed)
        channels_removed = np.zeros_like(removed_channels)
    elif np.any(removed_channels):
        # noinspection PyBroadException
        try:
            # noinspection PyUnresolvedReferences
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
        return EEG
    return EEG, {
        'index_base': 0,
        'criteria': {
            'max_flatline_duration': float(max_flatline_duration),
            'max_flatline_duration_samples': float(max_duration),
            'max_allowed_jitter': float(max_allowed_jitter),
            'max_jitter': float(max_jitter),
        },
        'detected_channels': np.array(removed_channels, dtype=bool, copy=True),
        'removed_channels': np.array(channels_removed, dtype=bool, copy=True),
        'removal_suppressed': removal_suppressed,
        'prior_clean_channel_mask': prior_clean_channel_mask,
        'clean_channel_mask': copy_clean_channel_mask(EEG),
    }
