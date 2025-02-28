import logging
import traceback
from typing import *

import numpy as np


def clean_flatlines(EEG: Dict[str, Any], max_flatline_duration: float = 5.0, max_allowed_jitter: float = 20.0):
    """Remove (near-) flat-lined channels.

    This is an automated artifact rejection function which ensures that
    the data contains no flat-lined channels.

    Args:
      EEG: the EEG data structure
      max_flatline_duration: maximum tolerated flatline duration. In seconds.
        If a channel has a longer flatline than this, it will be considered
        abnormal.
      max_allowed_jitter: maximum tolerated jitter during flatlines. As a
        multiple of epsilon.

    Returns:
      EEG: the EEG data structure with flatlined channels removed.

    Example:
      EEG = clean_flatlines(EEG)

    """
    X = EEG['data']
    max_duration = max_flatline_duration * EEG['srate']
    max_jitter = max_allowed_jitter * np.finfo(np.float64).eps

    removed_channels = np.zeros(X.shape[0], dtype=bool)
    for i in range(X.shape[0]):
        x = X[i, :]
        dx = np.abs(np.diff(x))
        bad = dx < max_jitter
        bad = np.concatenate(([False], bad, [False]), axis=0)
        transitions = np.diff(bad)
        breakpoints = np.where(transitions > 0)[0]
        flat_intervals = np.reshape(breakpoints, (-1, 2))
        if flat_intervals.shape[0] > 0:
            durations = flat_intervals[:, 1] - flat_intervals[:, 0]
            if np.max(durations) > max_duration:
                removed_channels[i] = True

    if np.all(removed_channels):
        print('Warning: all channels have a flat-line portion; not removing anything.')
    elif np.any(removed_channels):
        # noinspection PyBroadException
        try:
            from eegprep.pop_select import pop_select
            EEG = pop_select(EEG, nochannel=np.where(removed_channels)[0])
        except Exception as e:
            if isinstance(e, ImportError):
                print('Apparently you do not have access to a pop_select() function.')
            else:
                print('Could not select channels using EEGLAB''s pop_select(); details: ')
                traceback.print_exc()
            print('Falling back to a basic substitute and dropping signal meta-data.')
            EEG['data'] = EEG['data'][np.logical_not(removed_channels), :]
            if len(EEG['chanlocs']) == len(removed_channels):
                EEG['chanlocs'] = EEG['chanlocs'][np.logical_not(removed_channels)]
            EEG['nbchan'] = EEG['data'].shape[0]
            for fn in EEG.keys() & {'icawinv', 'icasphere', 'icaweights', 'icaact', 'stats', 'specdata', 'specicaact'}:
                EEG[fn] = np.array([])
            if CCM := EEG['etc'].get('clean_channel_mask') is not None:
                CCM[CCM] = ~removed_channels
            else:
                EEG['etc']['clean_channel_mask'] = ~removed_channels

    return EEG
