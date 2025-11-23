"""EEG latency to point conversion utilities."""

import numpy as np

def eeg_lat2point(lat_array, epoch_array, srate, timewin, timeunit=1.0, **kwargs):
    """
    Convert latencies in time units (relative to per-epoch time 0) to latencies in data points assuming concatenated epochs (EEGLAB style).

    Parameters
    ----------
    lat_array   : array-like
        Latencies in 'timeunit' units (e.g., seconds if timeunit=1, ms if 1e-3).
    epoch_array : array-like or scalar
        Epoch index for each latency (1-based). Scalar is broadcast.
    srate       : float
        Sampling rate in Hz.
    timewin     : sequence length-2
        [xmin xmax] epoch limits in 'timeunit' units.
    timeunit    : float, default 1.0
        Time unit in seconds.
    kwargs:
      outrange  : int {1,0}, default 1
        If 1, replace points out of range with the max valid point.
        If 0, raise an error.

    Returns
    -------
    newlat : np.ndarray
        1-based point indices assuming concatenated epochs.
    flag   : int
        1 if any point was out of range and replaced; else 0.
    """
    outrange = int(kwargs.get('outrange', 1))

    lat_array = np.atleast_1d(np.array(lat_array, dtype=float))
    
    # Handle different epoch_array cases
    if np.isscalar(epoch_array):
        epoch_array = np.ones_like(lat_array, dtype=float) * float(epoch_array)
    elif len(epoch_array) == 0:  # Empty list/array for continuous data
        epoch_array = np.ones_like(lat_array, dtype=float)  # All epochs = 1
    else:
        epoch_array = np.atleast_1d(np.array(epoch_array, dtype=float))
        if lat_array.size != epoch_array.size:
            if epoch_array.size == 1:
                epoch_array = np.ones_like(lat_array, dtype=float) * epoch_array.item()
            else:
                raise ValueError("eeg_lat2point: latency and epochs must have the same length")

    timewin = np.atleast_1d(np.array(timewin, dtype=float))
    if timewin.size != 2:
        raise ValueError("eeg_lat2point: timelimits must have length 2")

    # Scale timewin by timeunit to convert to seconds (following MATLAB logic)
    timewin_sec = timewin * float(timeunit)

    # points per epoch (inclusive endpoints)
    pnts = (timewin_sec[1] - timewin_sec[0]) * float(srate) + 1.0

    # core formula (EEGLAB):
    # newlat = (lat*timeunit - timewin_scaled(1))*srate + 1 + (epoch-1)*pnts
    newlat = (lat_array * float(timeunit) - timewin_sec[0]) * float(srate) + 1.0 \
             + (epoch_array - 1.0) * pnts

    flag = 0
    if newlat.size and epoch_array.size:
        max_valid = np.max(epoch_array * pnts)
        if np.max(newlat) > max_valid + 1e-12:  # tolerance for FP noise
            if outrange == 1:
                idx = newlat > max_valid
                newlat[idx] = max_valid
                flag = 1
                # mirror MATLAB's informational message
                print('eeg_lat2point(): Points out of range detected. Points replaced with maximum value')
            else:
                raise ValueError('Error in eeg_lat2point(): Points out of range detected')

    return newlat, flag