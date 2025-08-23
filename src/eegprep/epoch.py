import numpy as np

def epoch(data, events, lim, **kwargs):
    """
    EPOCH - Extract epochs time locked to specified events from continuous EEG data.

    Python translation of the provided MATLAB function. Assumes:
    - data is a NumPy array shaped (chan, frames) or (chan, frames, epochs)
    - events is a 1-D sequence of event latencies (in seconds if 'srate' != 1, otherwise in samples)
    - lim is [init, end] in seconds, centered on the events (e.g., [-1, 2])

    Optional keyword arguments (mirroring MATLAB 'key', val):
        srate: sampling rate in Hz for events expressed in seconds. Default 1.
        valuelim: [min, max] numeric bounds. Default [-Inf, Inf].
        verbose: 'on' or 'off'. Default 'on'.
        allevents: 1-D sequence of latencies for all events (same unit as events).
        alleventrange: [start, end] window relative to time-locking events (same unit as lim). Default lim.

    Returns:
        epochdat, newtime, indexes, alleventout, alllatencyout, reallim
    """

    # --- helpers to mimic MATLAB semantics ---
    def matlab_round(x):
        x = np.asarray(x, dtype=float)
        return np.sign(x) * np.floor(np.abs(x) + 0.5)

    def _as_1d(a):
        if a is None:
            return None
        arr = np.asarray(a).ravel()
        return arr

    # --- parse g like MATLAB struct(varargin{:}) with defaults ---
    g = {}
    g['srate'] = kwargs.get('srate', 1)
    g['valuelim'] = kwargs.get('valuelim', [-np.inf, np.inf])
    g['verbose'] = kwargs.get('verbose', 'on')
    g['allevents'] = _as_1d(kwargs.get('allevents', []))
    g['alleventrange'] = np.asarray(kwargs.get('alleventrange', lim), dtype=float)

    # --- computing point limits (MATLAB uses 1-based logic; keep math identical) ---
    reallim = np.zeros(2, dtype=int)
    reallim[0] = int(matlab_round(lim[0] * g['srate']))
    reallim[1] = int(matlab_round(lim[1] * g['srate'] - 1))  # minus 1 sample

    # --- epoching ---
    print('Epoching...')

    newdatalength = int(reallim[1] - reallim[0] + 1)

    # eeglab_options; option_memmapdata, mmo are EEGLAB-specific and not available here.
    # if option_memmapdata == 1:
    #     epochdat = mmo([], [data.shape[0], newdatalength, len(events)])  # MISSING: mmo
    # else:
    epochdat = np.zeros((data.shape[0], newdatalength, len(events)), dtype=np.asarray(data).dtype)

    # MATLAB: g.allevents = g.allevents(:)'
    g['allevents'] = _as_1d(g['allevents']) if g['allevents'] is not None else np.array([])

    if data.ndim == 2:
        dataframes = data.shape[1]
        datawidth = dataframes
    elif data.ndim == 3:
        dataframes = data.shape[1]
        datawidth = data.shape[1] * data.shape[2]
    else:
        raise ValueError('data must be 2D or 3D')

    events = _as_1d(events)
    indexes = np.zeros(len(events), dtype=int)
    alleventout = []
    alllatencyout = []

    for index in range(len(events)):
        # MATLAB uses 1-based sample indices, so we need to match that logic
        pos0 = int(np.floor(events[index] * g['srate'])) + 1  # Convert to 1-based like MATLAB
        posinit = pos0 + reallim[0]                           # 1-based + offset
        posend = pos0 + reallim[1]                            # 1-based + offset

        # Boundary check: must fall within the same pre-existing epoch when data is 3D
        # This matches MATLAB's logic exactly
        within_one_epoch = (np.floor((posinit - 1) / dataframes) == np.floor((posend - 1) / dataframes))
        within_bounds = (posinit >= 1) and (posend <= datawidth)

        if within_one_epoch and within_bounds:
            # Extract contiguous slice. Convert MATLAB 1-based inclusive [posinit:posend]
            # to Python 0-based slice [start:end_exclusive].
            start = posinit - 1  # Convert 1-based to 0-based
            end_excl = posend    # MATLAB inclusive end becomes Python exclusive end
            
            if data.ndim == 2:
                tmpdata = data[:, start:end_excl]
            else:
                # For 3D data, MATLAB treats it as linearized across frames*epochs
                # We need to reshape to 2D, slice, then reshape back
                # MATLAB uses column-major (Fortran) order
                data_2d = data.reshape(data.shape[0], -1, order='F')  # (chan, frames*epochs)
                tmpdata = data_2d[:, start:end_excl]

            epochdat[:, :, index] = tmpdata

            if not np.isinf(g['valuelim'][0]) or not np.isinf(g['valuelim'][1]):
                vec = tmpdata.reshape(-1)
                tmpmin = np.min(vec)
                tmpmax = np.max(vec)
                if (tmpmin > g['valuelim'][0]) and (tmpmax < g['valuelim'][1]):
                    indexes[index] = 1
                else:
                    if g['verbose'] == 'on':
                        print(f'Warning: event {index + 1} out of value limits')
            else:
                indexes[index] = 1
        else:
            if g['verbose'] == 'on':
                print(f'Warning: event {index + 1} out of data boundary')

        # Re-reference events
        if g['allevents'] is not None and g['allevents'].size > 0:
            posinit_re = pos0 + g['alleventrange'][0] * g['srate']
            posend_re = pos0 + g['alleventrange'][1] * g['srate']
            ae_pts = g['allevents'] * g['srate']
            # MATLAB: eventtrial = intersect_bc(find(ae_pts >= posinit_re), find(ae_pts < posend_re));
            # MISSING: intersect_bc. Using NumPy equivalent below.
            mask = (ae_pts >= posinit_re) & (ae_pts < posend_re)
            eventtrial = np.where(mask)[0]  # indices into allevents
            alleventout.append(eventtrial)
            alllatencyout.append(ae_pts[eventtrial] - pos0)

    newtime = np.array([reallim[0] / g['srate'], reallim[1] / g['srate']], dtype=float)

    keep = np.where(indexes == 1)[0]
    if keep.size == 0:
        # Keep shape-consistent empty outputs
        epochdat = epochdat[:, :, :0]
        indexes = keep
        alleventout = []
        alllatencyout = []
    else:
        epochdat = epochdat[:, :, keep]
        indexes = keep
        if len(alleventout) > 0:
            alleventout = [alleventout[i] for i in keep]
            alllatencyout = [alllatencyout[i] for i in keep]

    # Follow MATLAB exactly, even though this multiplication looks questionable.
    reallim = reallim * g['srate']

    return epochdat, newtime, indexes, alleventout, alllatencyout, reallim
