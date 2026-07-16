"""EEG flatline channel removal utilities."""

from typing import Any, Dict
import logging

import numpy as np

from .private.channel_removal import remove_channels_without_pop_select, update_clean_channel_mask

logger = logging.getLogger(__name__)


def clean_flatlines(EEG: Dict[str, Any], max_flatline_duration: float = 5.0, max_allowed_jitter: float = 20.0):
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

    Returns
    -------
    EEG : the EEG data structure with flatlined channels removed.

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

    # remove them
    if np.all(removed_channels):
        logger.warning('All channels have a flat-line portion; not removing anything.')
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

    return EEG
