"""Module for converting event latencies from points to time units."""

import numpy as np
from .utils.misc import round_mat


def eeg_point2lat(lat_array, epoch_array=None, srate=None, timewin=None, timeunit=1.0):
    """Convert event latencies in data points to latencies in time units (default
    seconds).

    Following EEGLAB's eeg_point2lat.

    Parameters
    ----------
    lat_array : array-like
        Event latencies in points, assuming concatenated epochs (1-based EEGLAB style).
    epoch_array : array-like or scalar or None
        Epoch index for each latency (1-based). If None, uses ones of same shape as lat_array.
    srate : float
        Sampling rate in Hz.
    timewin : sequence of length 2
        [xmin xmax] in 'timeunit' units (e.g., seconds if timeunit=1, ms if timeunit=1e-3).
    timeunit : float
        Time unit in seconds. Default 1.0, i.e. output in seconds. For milliseconds use 1e-3.

    Returns
    -------
    newlat : ndarray
        Converted latencies in 'timeunit' units (per-epoch time).
    """
    if srate is None:
        raise ValueError("srate is required")

    # defaults
    if epoch_array is None or (isinstance(epoch_array, (list, tuple, np.ndarray)) and len(np.atleast_1d(epoch_array)) == 0):
        epoch_array = 1

    if timewin is None:
        timewin = [0, 0]

    # flatten possible nested lists like MATLAB cell arrays
    lat_array = np.atleast_1d(np.array(lat_array, dtype=float))
    if np.isscalar(epoch_array):
        epoch_array = np.ones(lat_array.shape, dtype=float) * float(epoch_array)
    else:
        epoch_array = np.atleast_1d(np.array(epoch_array, dtype=float))

    if lat_array.size != epoch_array.size:
        if epoch_array.size != 1:
            raise ValueError("eeg_point2lat: latency and epoch arrays must have the same length")
        epoch_array = np.ones(lat_array.shape, dtype=float) * float(epoch_array)

    timewin = np.atleast_1d(np.array(timewin, dtype=float)) * float(timeunit)
    if timewin.size != 2:
        raise ValueError("eeg_point2lat: timelimits array must have length 2")

    # points per epoch (EEGLAB uses inclusive endpoints: pnts = (xmax-xmin)*srate + 1)
    pnts = (timewin[1] - timewin[0]) * float(srate) + 1.0

    # core formula (EEGLAB):
    # newlat = ((lat - (epoch-1)*pnts - 1)/srate + timewin(1)) / timeunit
    newlat = ((lat_array - (epoch_array - 1.0) * pnts - 1.0) / float(srate) + timewin[0]) / float(timeunit)

    # round to 1e-9 like MATLAB code
    newlat = round_mat(newlat * 1e9) / 1e9
    return newlat