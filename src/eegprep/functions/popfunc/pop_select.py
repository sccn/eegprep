import copy
import re
from typing import Any

import numpy as np

from eegprep.functions.adminfunc.eeg_checkset import eeg_checkset
from eegprep.functions.guifunc.inputgui import inputgui
from eegprep.functions.guifunc.spec import CallbackSpec, ControlSpec, DialogSpec
from eegprep.functions.popfunc.eeg_lat2point import eeg_lat2point
from eegprep.functions.popfunc.eeg_point2lat import eeg_point2lat
from eegprep.functions.popfunc.eeg_decodechan import eeg_decodechan
from eegprep.functions.popfunc.eeg_eegrej import eeg_eegrej


def pop_select(EEG, *args, gui=None, renderer=None, return_com=False, **kwargs):
    """Select EEG data using EEGLAB ``pop_select`` semantics."""
    options = _parse_key_value_args(args, kwargs)
    if not isinstance(EEG, list) and EEG.get("data") is None:
        raise ValueError('EEG["data"] is required')
    if gui is None:
        gui = not bool(options)
    if gui:
        gui_options = _run_gui(EEG[0] if isinstance(EEG, list) else EEG, renderer=renderer)
        if gui_options is None:
            return (EEG, "") if return_com else EEG
        options.update(gui_options)
        apply_options = _gui_options_for_apply(options)
    else:
        apply_options = options
    if isinstance(EEG, list):
        output = [pop_select(item, gui=False, **apply_options) for item in EEG]
        command = _history_command(options)
        return (output, command) if return_com else output
    output = _pop_select_apply(EEG, **apply_options)
    command = _history_command(options)
    return (output, command) if return_com else output


def _pop_select_apply(EEG, **kwargs):
    """Python port of EEGLAB's pop_select for dict-based EEG.

    Assumptions:
      - EEG is a dict (e.g., EEG['chanlocs'][0]['X'] for channel coordinates).
      - eeg_decodechan(EEG, query, mode, labels=True/type=True) exists and returns int indices (0-based).
      - eeg_eegrej(EEG, bad_point_ranges) exists and returns an updated EEG after removing samples
        from continuous data. bad_point_ranges is an (N,2) array of [start,end] sample indices (1-based like EEGLAB).

    Returns
    -------
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
    def _has_content(x):
        """Check if parameter has content (not None, not empty list/array)."""
        if x is None:
            return False
        if isinstance(x, (list, tuple)) and len(x) == 0:
            return False
        if isinstance(x, np.ndarray) and x.size == 0:
            return False
        return True

    # Track whether notime came directly from rmtime (to match MATLAB boundary adjustment logic)
    notime_from_rmtime = _has_content(g['rmtime'])

    if _has_content(g['rmtrial']):   g['notrial']  = g['rmtrial']
    if _has_content(g['rmtime']):    g['notime']   = g['rmtime']
    if _has_content(g['rmpoint']):   g['nopoint']  = g['rmpoint']
    if _has_content(g['rmchannel']): g['nochannel']= g['rmchannel']

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

    # if g['channel'] is a string, convert to list
    if isinstance(g['channel'], str):
        g['channel'] = [g['channel']]
    if isinstance(g['nochannel'], str):
        g['nochannel'] = [g['nochannel']]

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
            # show warning if not all channels are found and error if no channels are found
            if len(inds) != len(g['channel']):
                print(f"Warning: {len(g['channel'])-len(inds)} channels not found")
            if len(inds) == 0:
                raise ValueError(f"Channels not found: {g['channel']}")
            chan_selected_flag[:] = False
            chan_selected_flag[np.array(inds, dtype=int)] = True

        if _decode_list(g['nochannel']):
            inds, _ = eeg_decodechan(EEG, g['nochannel'], 'labels', True)
            chan_selected_flag[np.array(inds, dtype=int)] = False
            # show warning if not all channels are found and error if no channels are found
            if len(inds) != len(g['nochannel']):
                print(f"Warning: {len(g['nochannel'])-len(inds)} channels not found")

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
        points_flat = point_mat.reshape(-1)
        epochs_flat = np.ones(points_flat.shape, dtype=float)
        tflat = eeg_point2lat(points_flat, epochs_flat, srate, [xmin, xmax])
        g['time'] = np.asarray(tflat, dtype=float).reshape(point_mat.shape)
        g['notime'] = np.array([]).reshape(0, 2)

    if nopoint_mat.size:
        points_flat = nopoint_mat.reshape(-1)
        epochs_flat = np.ones(points_flat.shape, dtype=float)
        tflat = eeg_point2lat(points_flat, epochs_flat, srate, [xmin, xmax])
        g['notime'] = np.asarray(tflat, dtype=float).reshape(nopoint_mat.shape)
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
    if len(g['trial']) != trials and (EEG.get('event') is not None and len(EEG.get('event', [])) > 0):
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
            if _has_content(EEG.get('event')):
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
            if _has_content(EEG.get('epoch')):
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
                # EEGLAB only adjusts interior edges when notime was derived from time, not when it came from rmtime
                if notime_from_rmtime:
                    # Skip boundary adjustment when notime came directly from rmtime
                    adjusted = notime_mat.copy()
                else:
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
                pts = pts.reshape((-1, 2))
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
    ia = EEG.get('icaact')
    if _has_content(ia):
        if isinstance(ia, np.ndarray) and ia.ndim == 3:
            EEG['icaact'] = ia[:, :, trial_idx_0]

    # chanlocs bookkeeping
    chanlocs = EEG.get('chanlocs')
    if _has_content(chanlocs):
        chaninfo = EEG.get("chaninfo")
        if not isinstance(chaninfo, dict):
            chaninfo = {}
        removedchans = chaninfo.get("removedchans", [])
        if isinstance(removedchans, np.ndarray) and removedchans.size == 0:
            removedchans = []
        elif isinstance(removedchans, dict):
            removedchans = [removedchans]
        else:
            removedchans = list(removedchans or [])
        removed = np.setdiff1d(np.arange(nbchan), chan_idx)
        removedchans.extend(copy.deepcopy(chanlocs[int(index)]) for index in removed.tolist())
        chaninfo["removedchans"] = removedchans
        EEG["chaninfo"] = chaninfo
        EEG['chanlocs'] = [chanlocs[i] for i in chan_idx.tolist()]

    # update sizes
    EEG['trials'] = len(trial_idx_0)
    # pnts already updated if time selection on epoched; keep consistent otherwise
    EEG['pnts'] = EEG['data'].shape[1]
    EEG['nbchan'] = len(chan_idx)

    # epoch metadata
    if _has_content(EEG.get('epoch')):
        EEG['epoch'] = [EEG['epoch'][i] for i in trial_idx_0.tolist()]

     # ICA channel bookkeeping
    icachansind = EEG.get('icachansind')
    if _has_content(icachansind):
        rmch = np.setdiff1d(np.array(icachansind, dtype=int), chan_idx)
        icachans = list(range(len(icachansind)))
        for rc in rmch[::-1]:
            # remove component channel indices that were removed
            try:
                idx = int(np.where(np.array(icachansind) == rc)[0][0])
                icachans.pop(idx)
            except Exception:
                pass

        # new mapping of icachansind to kept channel positions
        newinds = []
        chan_idx_list = chan_idx.tolist()
        for ch in icachansind:
            if ch in chan_idx_list:
                newinds.append(chan_idx_list.index(ch))
        EEG['icachansind'] = newinds
    else:
        icasphere = EEG.get('icasphere')
        if _has_content(icasphere):
            icachans = range(icasphere.shape[1])
        else:
            icachans = 0

    # icawinv/icaweights/icasphere coherence if channels removed
    icawinv = EEG.get('icawinv')
    if _has_content(icawinv):
        if isinstance(icawinv, np.ndarray) and icawinv.size:
            flag_rmchan = (len(icachans) != icawinv.shape[0])
            if EEG.get('icaweights') is None or flag_rmchan:
                EEG['icawinv']    = icawinv[np.array(icachans, dtype=int), :]
                # recompute weights/sphere as in MATLAB
                iw = EEG['icawinv']
                EEG['icaweights'] = np.linalg.pinv(iw)
                EEG['icasphere']  = np.eye(EEG['icaweights'].shape[1])

    if _has_content(EEG.get('specicaact')):
        EEG['specicaact'] = np.array([])
   # specdata/specicaact handling
    if _has_content(EEG.get('specdata')):
        EEG['specdata'] = np.array([])
    # single epoch → drop event.epoch and clear epoch list
    if EEG['trials'] == 1:
        if _has_content(EEG.get('event')):
            for ev in EEG['event']:
                if 'epoch' in ev:
                    ev.pop('epoch', None)
        EEG['epoch'] = []

    # reject, stats clean-up
    if EEG.get('reject') is not None and isinstance(EEG.get('reject'), dict) and 'gcompreject' in EEG['reject'] and \
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
    if _has_content(EEG.get('event')):
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

    # Call eeg_checkset to ensure consistency after modifications
    EEG = eeg_checkset(EEG)

    return EEG


def pop_select_dialog_spec(EEG) -> DialogSpec:
    """Return the EEGLAB-like dialog spec for ``pop_select``."""
    chanlocs = list(EEG.get("chanlocs", []) or [])
    channel_labels = tuple(str(chan.get("labels", "")) for chan in chanlocs if isinstance(chan, dict))
    channel_types = tuple(
        value for value in dict.fromkeys(
            str(chan.get("type", "")) for chan in chanlocs if isinstance(chan, dict) and chan.get("type", "") != ""
        )
    )
    type_enabled = bool(channel_types)
    return DialogSpec(
        title="Select data -- pop_select()",
        function_name="pop_select",
        eeglab_source="functions/popfunc/pop_select.m",
        geometry=(
            (1, 1, 1),
            (1, 1, 0.25, 0.23, 0.51),
            (1, 1, 0.25, 0.23, 0.51),
            (1, 1, 0.25, 0.23, 0.51),
            (1, 1, 0.25, 0.23, 0.51),
            (1, 1, 0.25, 0.23, 0.51),
            (1,),
            (1, 1, 1),
        ),
        size=(695, 404),
        help_text="pophelp('pop_select')",
        controls=(
            ControlSpec("text", "Select data in:"),
            ControlSpec("text", "Input desired range"),
            ControlSpec("text", "on->remove these"),
            ControlSpec("text", "Time range [min max] (s)"),
            ControlSpec("edit", tag="time", value=""),
            ControlSpec("spacer"),
            ControlSpec("checkbox", "    ", tag="rmtime", value=False),
            ControlSpec("spacer"),
            ControlSpec("text", "Point range (ex: [1 10])"),
            ControlSpec("edit", tag="point", value=""),
            ControlSpec("spacer"),
            ControlSpec("checkbox", "    ", tag="rmpoint", value=False),
            ControlSpec("spacer"),
            ControlSpec("text", "Epoch range (ex: 3:2:10)"),
            ControlSpec("edit", tag="trial", value=""),
            ControlSpec("spacer"),
            ControlSpec("checkbox", "    ", tag="rmtrial", value=False),
            ControlSpec("spacer"),
            ControlSpec("text", "Channel(s)"),
            ControlSpec("edit", tag="chans", value=""),
            ControlSpec("spacer"),
            ControlSpec("checkbox", "    ", tag="rmchannel", value=False),
            ControlSpec(
                "pushbutton",
                "...",
                tag="chans_button",
                enabled=bool(channel_labels),
                callback=CallbackSpec(
                    "select_channels",
                    params={
                        "button": "chans_button",
                        "target": "chans",
                        "channels": channel_labels,
                    },
                    matlab_callback="pop_chansel(get(gcbf, 'userdata'), 'field', 'labels')",
                ),
            ),
            ControlSpec("text", "Channel type(s)"),
            ControlSpec("edit", tag="chantype", value=""),
            ControlSpec("spacer"),
            ControlSpec("checkbox", "    ", tag="rmchantype", value=False),
            ControlSpec(
                "pushbutton",
                "...",
                tag="chantype_button",
                enabled=type_enabled,
                callback=CallbackSpec(
                    "select_channels",
                    params={
                        "button": "chantype_button",
                        "target": "chantype",
                        "channels": channel_types,
                    },
                    matlab_callback="pop_chansel(get(gcbf, 'userdata'), 'field', 'type')",
                ),
            ),
            ControlSpec("spacer"),
            ControlSpec("spacer"),
            ControlSpec("pushbutton", "Scroll dataset", tag="scroll"),
            ControlSpec("spacer"),
        ),
    )


def _run_gui(EEG, renderer=None):
    spec = pop_select_dialog_spec(EEG)
    result = inputgui(spec, renderer=renderer)
    if result is None:
        return None
    options: dict[str, Any] = {}
    _add_range_option(options, result, "time", "rmtime")
    _add_range_option(options, result, "point", "rmpoint")
    _add_range_option(options, result, "trial", "rmtrial")
    _add_text_option(options, result, "chans", "rmchannel", keep_key="channel", remove_key="rmchannel")
    _add_text_option(options, result, "chantype", "rmchantype", keep_key="chantype", remove_key="rmchantype")
    return options or None


def _add_range_option(options, result, tag, remove_tag):
    text = str(result.get(tag, "") or "").strip()
    if not text:
        return
    key = remove_tag if result.get(remove_tag) else tag
    options[key] = _parse_numeric_text(text)


def _add_text_option(options, result, tag, remove_tag, *, keep_key, remove_key):
    text = str(result.get(tag, "") or "").strip()
    if not text:
        return
    key = remove_key if result.get(remove_tag) else keep_key
    options[key] = _parse_text_tokens(text)


def _gui_options_for_apply(options):
    apply_options = dict(options)
    for key in ("channel", "rmchannel"):
        values = apply_options.get(key)
        if values is None:
            continue
        if isinstance(values, (list, tuple)):
            apply_options[key] = [value - 1 if isinstance(value, int) else value for value in values]
        elif isinstance(values, int):
            apply_options[key] = values - 1
    return apply_options


def _parse_key_value_args(args, kwargs):
    if len(args) % 2:
        raise ValueError("Key/value arguments must be in pairs")
    options = dict(kwargs)
    for index in range(0, len(args), 2):
        key = args[index]
        if isinstance(key, bytes):
            key = key.decode("utf-8")
        if not isinstance(key, str):
            raise ValueError("Keys must be strings")
        options[key.lower()] = args[index + 1]
    return options


def _parse_numeric_text(text):
    values = []
    for value in re.split(r"[\s,]+", text.strip().strip("[]")):
        if not value:
            continue
        values.extend(_parse_numeric_token(value))
    if len(values) == 1:
        return values
    if all(value.is_integer() for value in values):
        return [int(value) for value in values]
    return values


def _parse_numeric_token(value):
    if ":" not in value:
        return [float(value)]
    parts = [float(part) for part in value.split(":") if part]
    if len(parts) == 2:
        start, stop = parts
        step = 1.0 if stop >= start else -1.0
    elif len(parts) == 3:
        start, step, stop = parts
        if step == 0:
            raise ValueError("Colon range step cannot be zero")
    else:
        raise ValueError("Colon ranges must use start:stop or start:step:stop")
    values = []
    current = start
    if step > 0:
        while current <= stop + np.finfo(float).eps:
            values.append(current)
            current += step
    else:
        while current >= stop - np.finfo(float).eps:
            values.append(current)
            current += step
    return values


def _parse_text_tokens(text):
    tokens = re.findall(r"'([^']*)'|\"([^\"]*)\"|([^,\s]+)", text.strip().strip("{}"))
    values = [next(part for part in token if part) for token in tokens]
    parsed = []
    for value in values:
        try:
            parsed.append(int(value))
        except ValueError:
            parsed.append(value)
    return parsed


def _history_command(options):
    if not options:
        return ""
    parts = []
    for key, value in options.items():
        parts.extend([f"'{key}'", _history_value(value)])
    return f"EEG = pop_select( EEG, {', '.join(parts)});"


def _history_value(value):
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, (list, tuple, np.ndarray)):
        if all(isinstance(item, str) for item in value):
            return "{" + " ".join(_history_value(item) for item in value) + "}"
        return "[" + " ".join(_history_value(item) for item in value) + "]"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)

if __name__ == '__main__':
    from eegprep.functions.popfunc.pop_loadset import pop_loadset
    EEG = pop_loadset('sample_data/eeglab_data.set')
    EEG2 = pop_select(EEG, channel=['FP1', 'FP2'])
    print(EEG2)
