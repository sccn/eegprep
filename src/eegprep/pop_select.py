import numpy as np
import copy
from eegprep.eeg_lat2point import eeg_lat2point
from eegprep.eeg_point2lat import eeg_point2lat
from eegprep.eeg_decodechan import eeg_decodechan
from eegprep.eeg_eegrej import eeg_eegrej

def pop_select(EEG, **kwargs):
    """
    Python port of EEGLAB's pop_select for dict-based EEG.
    Assumptions:
      - EEG is a dict (e.g., EEG['chanlocs'][0]['X'] for channel coordinates).
      - eeg_decodechan(EEG, query, mode, labels=True/type=True) exists and returns int indices (0-based).
      - eeg_eegrej(EEG, bad_point_ranges) exists and returns an updated EEG after removing samples
        from continuous data. bad_point_ranges is an (N,2) array of [start,end] sample indices (1-based like EEGLAB).
    Returns
      EEG_out, com
    """

    # shallow options with MATLAB-compatible aliases
    g = {
        'time':        kwargs.get('time',        []),   # seconds; can be Nx2 for continuous
        'notime':      kwargs.get('notime',      []),   # seconds; legacy alias of rmtime
        'rmtime':      kwargs.get('rmtime',      []),   # seconds; removed
        'trial':       kwargs.get('trial',       None), # 1-based indices in MATLAB; we will accept 1-based and convert
        'notrial':     kwargs.get('notrial',     []),
        'rmtrial':     kwargs.get('rmtrial',     []),
        'point':       kwargs.get('point',       []),   # samples; accepts vectors or Nx2
        'nopoint':     kwargs.get('nopoint',     []),
        'rmpoint':     kwargs.get('rmpoint',     []),
        'channel':     kwargs.get('channel',     []),   # indices or names
        'nochannel':   kwargs.get('nochannel',   []),
        'rmchannel':   kwargs.get('rmchannel',   []),
        'chantype':    kwargs.get('chantype',    []),   # names
        'rmchantype':  kwargs.get('rmchantype',  []),
        'sort':        kwargs.get('sort',        None),
        'sorttrial':   kwargs.get('sorttrial',   'on'),
        'checkchans':  kwargs.get('checkchans',  'on'),
    }

    # alias normalization
    if g['rmtrial']:   g['notrial']  = g['rmtrial']
    if g['rmtime']:    g['notime']   = g['rmtime']
    if g['rmpoint']:   g['nopoint']  = g['rmpoint']
    if g['rmchannel']: g['nochannel']= g['rmchannel']

    # Core EEG fields with basic checks
    def _get(key, default=None):
        return EEG.get(key, default)

    trials  = int(_get('trials', 1))
    pnts    = int(_get('pnts'))
    nbchan  = int(_get('nbchan'))
    xmin    = float(_get('xmin'))
    xmax    = float(_get('xmax'))
    srate   = float(_get('srate'))
    data    = _get('data')  # expected shape (nbchan, pnts, trials) if epoched; (nbchan, pnts) if continuous

    if data is None:
        raise ValueError('EEG["data"] is required')

    epoched = trials > 1
    if not epoched and data.ndim == 3:
        # normalize continuous to (nbchan, pnts)
        data = data[:, :, 0]
        EEG['data'] = data

    # 1) Trial selection and sorting
    if g['trial'] is None:
        g['trial'] = list(range(1, trials + 1))  # 1-based like EEGLAB
    if g['sort'] is not None:
        g['sorttrial'] = 'on' if g['sort'] else 'off'

    # remove notrial
    trial_set = np.array(g['trial'], dtype=int)
    notrial_set = np.array(g['notrial'], dtype=int) if g['notrial'] else np.array([], dtype=int)

    if g['sorttrial'].lower() == 'on':
        keep = np.setdiff1d(trial_set, notrial_set, assume_unique=False)
        keep.sort()
        g['trial'] = keep.tolist()
        if len(g['trial']) == 0:
            fname = _get('filename', '<EEG>')
            raise ValueError(f'Error: dataset {fname} is empty')
    else:
        # preserve order; drop any listed in notrial and duplicates
        mask = ~np.isin(trial_set, notrial_set)
        trial_seq = trial_set[mask]
        # unique while preserving order
        _, idx = np.unique(trial_seq, return_index=True)
        g['trial'] = trial_seq[np.sort(idx)].tolist()

    if min(g['trial']) < 1 or max(g['trial']) > trials:
        raise ValueError('Wrong trial range')

    # 2) Channel selection by name or type, with mutual exclusion
    def _decode_list(x):
        # normalize scalar to list
        if x is None or x == []:
            return []
        if isinstance(x, (list, tuple, np.ndarray)):
            return list(x)
        return [x]

    chan_selected_flag = np.ones(nbchan, dtype=bool)  # default keep all

    # names win over types if provided
    if _decode_list(g['channel']) or _decode_list(g['nochannel']):
        if _decode_list(g['chantype']) or _decode_list(g['rmchantype']):
            raise ValueError('Select channels by name OR by type, not both')

        if _decode_list(g['channel']):
            inds, _ = eeg_decodechan(EEG, g['channel'], 'labels', True)
            chan_selected_flag[:] = False
            chan_selected_flag[np.array(inds, dtype=int)] = True

        if _decode_list(g['nochannel']):
            inds, _ = eeg_decodechan(EEG, g['nochannel'], 'labels', True)
            chan_selected_flag[np.array(inds, dtype=int)] = False

    else:
        # by type
        if _decode_list(g['chantype']):
            inds = eeg_decodechan(EEG, g['chantype'], 'type', True)
            chan_selected_flag[:] = False
            chan_selected_flag[np.array(inds, dtype=int)] = True

        if _decode_list(g['rmchantype']):
            inds = eeg_decodechan(EEG, g['rmchantype'], 'type', True)
            chan_selected_flag[np.array(inds, dtype=int)] = False

    g['channel'] = np.where(chan_selected_flag)[0].tolist()
    if len(g['channel']) == 0:
        raise ValueError('Empty channel selection')

    # normalize vector forms for point/nopoint to Nx2
    def _normalize_range_matrix(x):
        x = np.asarray(x) if isinstance(x, (list, tuple, np.ndarray)) else np.array([])
        if x.size == 0:
            return x.reshape(0, 2)
        x = x.astype(float)
        if x.ndim == 1:
            if x.size <= 2:
                return np.array(x).reshape(1, 2)
            # vector form → [first last]
            print('Warning: vector format for point/time range is deprecated')
            return np.array([x[0], x[-1]], dtype=float).reshape(1, 2)
        if x.shape[1] != 2:
            raise ValueError('Time/point range must contain exactly 2 columns')
        return x

    # points → time overrides time (as in MATLAB code)
    point_mat   = _normalize_range_matrix(g['point'])
    nopoint_mat = _normalize_range_matrix(g['nopoint'])

    if point_mat.size:
        tmat = np.zeros_like(point_mat, dtype=float)
        for i in range(point_mat.size):
            tmat.flat[i] = eeg_point2lat(point_mat.flat[i], 1, srate, [xmin, xmax])
        g['time'] = tmat
        g['notime'] = np.array([]).reshape(0, 2)

    if nopoint_mat.size:
        tmat = np.zeros_like(nopoint_mat, dtype=float)
        for i in range(nopoint_mat.size):
            tmat.flat[i] = eeg_point2lat(nopoint_mat.flat[i], 1, srate, [xmin, xmax])
        g['notime'] = tmat
        g['time'] = np.array([]).reshape(0, 2)

    time_mat   = _normalize_range_matrix(g['time'])
    notime_mat = _normalize_range_matrix(g['notime'])

    # constrain ranges to dataset bounds
    def _clip_time_matrix(mat):
        if mat.size == 0:
            return mat
        mat = mat.copy()
        mat[mat >  xmax] = xmax
        mat[mat <  xmin] = xmin
        return mat

    time_mat = _clip_time_matrix(time_mat)
    notime_mat = _clip_time_matrix(notime_mat)

    # notime → equivalent keep time for epoched only when boundary aligned
    if notime_mat.size:
        if not epoched:
            # continuous: we will pass notime to eeg_eegrej below
            pass
        else:
            # must touch epoch boundary
            if not (
                np.any(np.isclose(notime_mat[:, 0], xmin)) or
                np.any(np.isclose(notime_mat[:, 1], xmax))
            ):
                raise ValueError('Wrong notime range for epoched data; must include an epoch boundary')

            # convert notime to keep time when aligned to boundaries
            if np.any(np.isclose(notime_mat[:, 1], xmax)) and np.any(np.isclose(notime_mat[:, 0], xmin)):
                # removing whole epoch is handled via trial selection; here we keep as-is
                pass
            else:
                if np.any(np.isclose(notime_mat[:, 1], xmax)):
                    time_mat = np.array([[xmin, float(notime_mat[0, 0])]], dtype=float)
                elif np.any(np.isclose(notime_mat[:, 0], xmin)):
                    time_mat = np.array([[float(notime_mat[0, 1]), xmax]], dtype=float)
                else:
                    raise ValueError('Wrong notime range. Cannot remove a central slice from epoched data.')

    # 4) Informational prints (optional)
    if len(g['trial']) != trials:
        print(f"Removing {trials - len(g['trial'])} trial(s)...")
    if len(g['channel']) != nbchan:
        print(f"Removing {nbchan - len(g['channel'])} channel(s)...")

    # 5) Recompute event epoch indices and latencies when trials are dropped
    if len(g['trial']) != trials and EEG.get('event'):
        if not any('epoch' in ev for ev in EEG['event']):
            print('Pop_epoch warning: bad event format with epoch dataset, removing events')
            EEG['event'] = []
        else:
            keepevent = []
            for k, ev in enumerate(EEG['event']):
                if 'epoch' in ev:
                    # old epoch number → new index within kept list
                    try:
                        newindex = np.where(np.array(g['trial']) == int(ev['epoch']))[0]
                    except Exception:
                        newindex = np.array([], dtype=int)
                    if newindex.size:
                        keepevent.append(k)
                        if 'latency' in ev:
                            # adjust latency to new epoch position
                            ev['latency'] = (
                                float(ev['latency'])
                                - (int(ev['epoch']) - 1) * pnts
                                + (int(newindex[0]) ) * pnts
                            )
                        ev['epoch'] = int(newindex[0] + 1)  # back to 1-based for consistency
            diffevent = np.setdiff1d(np.arange(len(EEG['event'])), np.array(keepevent, dtype=int))
            if diffevent.size:
                print(f"Pop_select: removing {diffevent.size} unreferenced events")
                EEG['event'] = [EEG['event'][i] for i in range(len(EEG['event'])) if i in keepevent]

    # 6) Apply time selection
    if time_mat.size or notime_mat.size:
        if epoched:
            # epoched: crop epoch time window uniformly across epochs
            if time_mat.size == 0:
                # convert notime → keep time (already validated above)
                # if both edges present, nothing to crop
                if notime_mat.size:
                    if np.any(np.isclose(notime_mat[:, 0], xmin)) and np.any(np.isclose(notime_mat[:, 1], xmax)):
                        time_mat = np.array([[xmin, xmax]], dtype=float)
                    elif np.any(np.isclose(notime_mat[:, 1], xmax)):
                        time_mat = np.array([[xmin, float(notime_mat[0, 0])]], dtype=float)
                    elif np.any(np.isclose(notime_mat[:, 0], xmin)):
                        time_mat = np.array([[float(notime_mat[0, 1]), xmax]], dtype=float)
            # enforce single [tmin,tmax] for epoched
            if time_mat.shape[0] != 1:
                raise ValueError('Epoched data requires a single [tmin tmax] window')
            tmin, tmax = float(time_mat[0,0]), float(time_mat[0,1])
            # convert to points (1-based like EEGLAB)
            pts, _ = eeg_lat2point([tmin, tmax], [1, 1], srate, [xmin, xmax])
            a, b = int(pts[0]), int(pts[1])
            if a < 1: a = 1
            if b > pnts: b = pnts
            if b < a:
                raise ValueError('Invalid time window mapped to points')

            if data.ndim == 3:
                data = data[:, a-1:b, :]  # convert to 0-based
            else:
                # rare case: epoched flagged but data is 2D with trials meta
                data = data[:, a-1:b]
            EEG['data'] = data
            EEG['xmin'] = tmin
            EEG['xmax'] = tmax
            EEG['pnts'] = data.shape[1]
            pnts = EEG['pnts']

            # shift event latencies within each epoch window
            if EEG['event'] is not None and len(EEG['event']) > 0:
                newevents = []
                for ev in EEG['event']:
                    if 'epoch' in ev and 'latency' in ev:
                        e = copy.deepcopy(ev)
                        # within-epoch latency shift by (a-1) samples
                        e['latency'] = e['latency'] - (a - 1)
                        # keep only events that remain inside the cropped window
                        if 1 <= e['latency'] <= pnts * len(g['trial']):
                            newevents.append(e)
                EEG['event'] = newevents

            # erase epoch-level event fields
            if EEG['epoch'] is not None and len(EEG['epoch']) > 0:
                # remove fields that start with 'event'
                new_epoch = []
                for ep in EEG['epoch']:
                    new_ep = {k: v for k, v in ep.items() if not k.startswith('event')}
                    new_epoch.append(new_ep)
                EEG['epoch'] = new_epoch

        else:
            # continuous: build notime segments to reject if only keep time was provided
            if time_mat.size and not notime_mat.size:
                # convert keep time windows into notime complements
                # compose sequence along [xmin, xmax]
                bounds = []
                t_sorted = time_mat[np.argsort(time_mat[:,0])]
                cur = xmin
                for row in t_sorted:
                    t0, t1 = float(row[0]), float(row[1])
                    if t0 > cur:
                        bounds.append([cur, t0])
                    cur = max(cur, t1)
                if cur < xmax:
                    bounds.append([cur, xmax])
                notime_mat = np.array(bounds, dtype=float) if bounds else np.empty((0,2))

            # now reject notime_mat intervals from continuous data
            if notime_mat.size:
                # EEGLAB adjusts interior edges by +/- one sample; replicate
                adjusted = notime_mat.copy()
                for i in range(adjusted.shape[0]):
                    # shift interior boundaries off-sample
                    if adjusted[i,0] != xmin:
                        adjusted[i,0] += 1.0 / srate
                    if adjusted[i,1] != xmax:
                        adjusted[i,1] -= 1.0 / srate
                # map to 1-based sample indices
                nbtimes = adjusted.size
                pts, _ = eeg_lat2point(adjusted.reshape(-1), np.ones(nbtimes), srate, [xmin, xmax])
                pts = pts.reshape((-1, 2)).astype(int)
                # drop empty ranges
                keep_rows = (pts[:,1] - pts[:,0]) != 0
                pts = pts[keep_rows]
                if pts.size:
                    EEG = eeg_eegrej(EEG, pts)
                    data = EEG['data']
                    pnts = EEG['pnts']
                    xmin = EEG['xmin']
                    xmax = EEG['xmax']

    # 7) Apply channel and trial subsetting
    # respect memory-mapped constraints lightly: we subset in place
    chan_idx = np.array(g['channel'], dtype=int)
    trial_idx_0 = np.array(g['trial'], dtype=int) - 1  # to 0-based

    # erase dipfit if channels removed
    if len(chan_idx) != nbchan and EEG.get('dipfit') is not None:
        print('warning: erasing dipole information since channels have been removed')
        EEG['dipfit'] = []
        EEG['roi'] = []

    # data slicing
    data = EEG['data']
    if trials > 1 or data.ndim == 3:
        # epoched array expected
        EEG['data'] = data[chan_idx, :, :][:, :, trial_idx_0]
    else:
        EEG['data'] = data[chan_idx, :]

    # icaact
    if EEG['icaact'] is not None and len(EEG['icaact']) > 0:
        ia = EEG['icaact']
        if ia is not None and isinstance(ia, np.ndarray) and ia.ndim == 3:
            EEG['icaact'] = ia[:, :, trial_idx_0]

    # chanlocs bookkeeping
    if EEG['chanlocs'] is not None and len(EEG['chanlocs']) > 0:
        if 'chaninfo' in EEG and EEG['chaninfo'] is not None and len(EEG['chaninfo']) > 0:
            EEG['chaninfo'] = {}
        if 'removedchans' not in EEG['chaninfo'] or EEG['chaninfo']['removedchans'] is None:
            EEG['chaninfo']['removedchans'] = []
        try:
            removed = np.setdiff1d(np.arange(nbchan), chan_idx)
            for chan in EEG['chanlocs'][removed.tolist()]:
                EEG['chaninfo']['removedchans'].append(chan)
        except Exception:
            print('There was an issue storing removed channels in pop_select')
        EEG['chanlocs'] = [EEG['chanlocs'][i] for i in chan_idx.tolist()]

    # update sizes
    EEG['trials'] = len(trial_idx_0)
    # pnts already updated if time selection on epoched; keep consistent otherwise
    EEG['pnts'] = EEG['data'].shape[1]
    EEG['nbchan'] = len(chan_idx)

    # epoch metadata
    if EEG['epoch'] is not None and len(EEG['epoch']) > 0:
        EEG['epoch'] = [EEG['epoch'][i] for i in trial_idx_0.tolist()]

     # ICA channel bookkeeping
    if EEG.get('icachansind') is not None and len(EEG.get('icachansind')) > 0:
        rmch = np.setdiff1d(np.array(EEG['icachansind'], dtype=int), chan_idx)
        icachans = list(range(len(EEG['icachansind'])))
        for rc in rmch[::-1]:
            # remove component channel indices that were removed
            try:
                idx = int(np.where(np.array(EEG['icachansind']) == rc)[0][0])
                icachans.pop(idx)
            except Exception:
                pass

        # new mapping of icachansind to kept channel positions
        newinds = []
        chan_idx_list = chan_idx.tolist()
        for ch in EEG['icachansind']:
            if ch in chan_idx_list:
                newinds.append(chan_idx_list.index(ch))
        EEG['icachansind'] = newinds
    else:
        if EEG['icasphere'] is not None and len(EEG['icasphere']) > 0:  
            icachans = range(EEG['icasphere'].shape[1])
        else:
            icachans = 0

    # icawinv/icaweights/icasphere coherence if channels removed
    if EEG['icawinv'] is not None and len(EEG['icawinv']) > 0:
        icawinv = EEG['icawinv']
        if isinstance(icawinv, np.ndarray) and icawinv.size:
            flag_rmchan = (len(icachans) != icawinv.shape[0])
            if EEG.get('icaweights') is None or flag_rmchan:
                EEG['icawinv']    = icawinv[np.array(icachans, dtype=int), :]
                # recompute weights/sphere as in MATLAB
                iw = EEG['icawinv']
                EEG['icaweights'] = np.linalg.pinv(iw)
                EEG['icasphere']  = np.eye(EEG['icaweights'].shape[1])

    if EEG['specicaact'] is not None and len(EEG['specicaact']) > 0:
        EEG['specicaact'] = np.array([]) 
   # specdata/specicaact handling
    if EEG['specdata'] is not None and len(EEG['specdata']) > 0:
        EEG['specdata'] = np.array([])
    # single epoch → drop event.epoch and clear epoch list
    if EEG['trials'] == 1:
        if EEG['event'] is not None and len(EEG['event']) > 0:
            for ev in EEG['event']:
                if 'epoch' in ev:
                    ev.pop('epoch', None)
        EEG['epoch'] = []

    # reject, stats clean-up
    if EEG['reject'] is not None and isinstance(EEG['reject'], dict) and 'gcompreject' in EEG['reject'] and \
       len(g['channel']) == nbchan:
        tmp = EEG['reject']['gcompreject']
        EEG['reject'] = {}
        EEG['reject']['gcompreject'] = tmp
    else:
        EEG['reject'] = {}
    EEG['stats'] = {}
    EEG['reject']['rejmanual'] = []
    EEG['stats']['jp'] = []

    # event consistency check stub (depends on eeg_checkset in EEGLAB)
    # Here we simply ensure event latencies are within data bounds when possible.
    if EEG['event'] is not None and len(EEG['event']) > 0:
        total_pts = EEG['pnts'] * EEG['trials']
        cleaned = []
        for ev in EEG['event']:
            if 'latency' in ev:
                lat = float(ev['latency'])
                if 1 <= lat <= total_pts:
                    cleaned.append(ev)
            else:
                cleaned.append(ev)
        EEG['event'] = cleaned

    return EEG