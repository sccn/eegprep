"""Windowed-sinc FIR filter order estimation."""

from typing import Optional, Tuple

import numpy as np

__all__ = ["firwsord"]


def firwsord(wintype: str, fs: float, df: float, dev: Optional[float] = None) -> Tuple[int, float]:
    """Estimate windowed sinc FIR filter order depending on window type and requested transition band width.

    Parameters
    ----------
    wintype : str
        Window type. One of 'rectangular', 'hann', 'hamming', 'blackman', or 'kaiser'.
    fs : float
        Sampling frequency.
    df : float
        Requested transition band width.
    dev : float, optional
        Maximum passband deviation/ripple (Kaiser window only).

    Returns
    -------
    m : int
        Estimated filter order.
    dev : float
        Maximum passband deviation/ripple.

    Notes
    -----
    Based on a MATLAB implementation by Andreas Widmann, University of Leipzig, 2005.
    """
    win_type_array = ['rectangular', 'hann', 'hamming', 'blackman', 'kaiser']
    win_df_array = [0.9, 3.1, 3.3, 5.5]
    win_dev_array = [0.089, 0.0063, 0.0022, 0.0002]

    # Check arguments
    if fs is None or df is None or wintype is None:
        raise ValueError('Not enough input arguments.')

    # Window type
    try:
        wintype_idx = win_type_array.index(wintype)
    except ValueError:
        raise ValueError('Unknown window type.')

    df_norm = df / fs  # Normalize transition band width

    if wintype_idx == 4:  # Kaiser window (index 4 in 0-based, was 5 in 1-based MATLAB)
        if dev is None:
            raise ValueError('Not enough input arguments.')
        devdb = -20 * np.log10(dev)
        m = 1 + (devdb - 8) / (2.285 * 2 * np.pi * df_norm)
    else:
        m = win_df_array[wintype_idx] / df_norm
        dev = win_dev_array[wintype_idx]

    m = int(np.ceil(m / 2) * 2)  # Make filter order even (FIR type I)

    return m, dev
