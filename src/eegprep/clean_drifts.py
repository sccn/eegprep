from typing import *
import logging

import numpy as np
from scipy.signal import filtfilt

from .utils import design_kaiser, design_fir, filtfilt_fast

logger = logging.getLogger(__name__)


def clean_drifts(
        EEG: Dict[str, Any],
        transition: Sequence[float] = (0.5, 1),
        attenuation: float = 80.0,
        method: str = 'fft',
) -> Dict[str, Any]:
    """Removes drifts from the data using a forward-backward high-pass filter.

    This removes drifts from the data using a forward-backward (non-causal) filter.
    NOTE: If you are doing directed information flow analysis, do no use this filter but some other one.

    Args:
      EEG: the continuous-time EEG data structure
      transition: the transition band in Hz, i.e. lower and upper edge of the
        transition as in (lo,hi)
      attenuation: stop-band attenuation, in dB
      method: the method to use for filtering ('fft' or 'fir')

    Returns:
      EEG: the filtered EEG data structure

    """
    EEG['data'] = np.asarray(EEG['data'], dtype=np.float64)

    # design highpass FIR filter
    transition = 2*np.asarray(transition) / EEG['srate']
    wnd = design_kaiser(transition[0], transition[1], attenuation, True)
    B = design_fir(
        len(wnd)-1,
        np.concatenate(([0], transition, [1])),
        [0, 0, 1, 1],
        w=wnd)

    op = filtfilt if method == 'fir' else filtfilt_fast

    # apply it, channel by channel to save memory
    for i in range(EEG['data'].shape[0]):
        EEG['data'][i, :] = op(B, 1, EEG['data'][i, :])
    EEG['etc']['clean_drifts_kernel'] = B

    return EEG
