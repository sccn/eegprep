"""EEG baseline removal utilities."""

import numpy as np
from typing import Iterable, List, Optional, Tuple

from eegprep.eeg_findboundaries import eeg_findboundaries
from .utils.misc import round_mat

def _normalize_pointrange(
    pointrange: Optional[Iterable], pnts: int
) -> np.ndarray:
    """
    Normalize MATLAB-like pointrange into a 0-based numpy index vector within [0, pnts-1].

    Accepts:
      - None or empty → full range
      - two-element iterable [start, end] inclusive (1-based or 0-based tolerated)
      - any iterable of indices → will be clipped and uniqued in ascending order
    """
    if pointrange is None:
        return np.arange(pnts, dtype=int)

    # convert to numpy array of ints/floats
    arr = np.asarray(list(pointrange)) if not isinstance(pointrange, slice) else None

    if isinstance(pointrange, slice):
        start = 0 if pointrange.start is None else int(pointrange.start)
        stop = pnts if pointrange.stop is None else int(pointrange.stop)
        step = 1 if pointrange.step is None else int(pointrange.step)
        idx = np.arange(start, stop, step, dtype=int)
    elif arr.ndim == 1 and arr.size == 0:
        idx = np.arange(pnts, dtype=int)
    elif arr.ndim == 1 and arr.size == 2:
        # tolerate 1-based inputs; convert to 0-based inclusive
        a = int(arr[0])
        b = int(arr[1])
        # if clearly 1-based, shift; otherwise assume 0-based
        if a >= 1 and b >= 1:
            a -= 1
            b -= 1
        if a < 0:
            a = 0
        if b >= pnts:
            b = pnts - 1
        if b < a:
            a, b = b, a
        idx = np.arange(a, b + 1, dtype=int)
    else:
        # arbitrary index list; clip and uniquify sorted
        idx = np.unique(arr.astype(int))
        idx = idx[(idx >= 0) & (idx < pnts)]

    return idx.astype(int)


def _indices_from_timerange(times: np.ndarray, timerange: Iterable[float]) -> np.ndarray:
    """Build 0-based indices from a millisecond timerange using EEG['times'] (ms)."""
    tr = np.asarray(list(timerange), dtype=float)
    if tr.size != 2:
        raise ValueError('timerange must contain 2 elements [min_ms, max_ms]')
    tmin, tmax = float(tr[0]), float(tr[1])
    if tmin < float(times[0]) or tmax > float(times[-1]):
        raise ValueError('pop_rmbase(): Bad time range')
    mask = (times >= tmin) & (times <= tmax)
    idx = np.where(mask)[0]
    if idx.size == 0:
        # fallback to nearest
        idx = np.array([np.argmin(np.abs(times - tmin))], dtype=int)
    return idx.astype(int)


def _subtract_mean_over_indices(data: np.ndarray, idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subtract mean over the provided indices from each channel for 2D data (chans x frames).

    Returns (data_out, means) where means is chans x 1.
    """
    if data.ndim != 2:
        raise ValueError('Expected 2D array (channels x frames)')
    if idx.size == 0:
        return data, np.zeros((data.shape[0], 1))
    means = np.nanmean(data[:, idx], axis=1, keepdims=True)
    return data - means, means


def pop_rmbase(
    EEG: dict,
    timerange: Optional[Iterable[float]] = None,
    pointrange: Optional[Iterable[int]] = None,
    chanlist: Optional[Iterable[int]] = None,
) -> dict:
    """
    POP_RMBASE - remove channel baseline means from an epoched or continuous EEG dataset.

    Parameters
    ----------
    EEG : dict
        EEGLAB-like EEG structure with keys: 'data', 'nbchan', 'pnts', 'trials', 'times', 'event'.
        Event latencies are 1-based indices (EEGLAB convention).
    timerange : [min_ms, max_ms] or None
        Baseline latency range in milliseconds; overrides pointrange when provided.
    pointrange : iterable of indices or [start, end]
        Baseline sample indices (0-based or 1-based tolerated). If None/empty, use whole epoch.
    chanlist : iterable of channel indices (0-based). If None, all channels are used.

    Returns
    -------
    EEG : dict
        Updated EEG structure with baseline removed. EEG['icaact'] is cleared.
    """
    if EEG is None or 'data' not in EEG or EEG['data'] is None or (hasattr(EEG['data'], 'size') and EEG['data'].size == 0):
        raise ValueError('pop_rmbase(): cannot remove baseline of an empty dataset')

    data = EEG['data']
    nbchan = int(EEG.get('nbchan', data.shape[0]))
    pnts = int(EEG.get('pnts', data.shape[1] if data.ndim >= 2 else 0))
    trials = int(EEG.get('trials', data.shape[2] if data.ndim == 3 else 1))

    if chanlist is None or (isinstance(chanlist, (list, tuple, np.ndarray)) and len(chanlist) == 0):
        chanlist = list(range(nbchan))
    else:
        chanlist = list(map(int, list(chanlist)))

    # Determine baseline indices
    if timerange is not None and len(list(timerange)) > 0:
        if 'times' not in EEG or EEG['times'] is None:
            raise ValueError('EEG["times"] is required when using timerange')
        pr = _indices_from_timerange(np.asarray(EEG['times'], dtype=float), timerange)
    elif pointrange is not None and len(list(pointrange)) > 0:
        pr = _normalize_pointrange(pointrange, pnts)
    else:
        pr = np.arange(pnts, dtype=int)

    # Epoched vs continuous handling
    epoched = trials > 1 or (data.ndim == 3 and data.shape[-1] > 1)

    # Ensure data dimensionality conforms
    if epoched:
        if data.ndim != 3:
            # normalize to (nbchan, pnts, trials)
            data = data.reshape((nbchan, pnts, trials))
            EEG['data'] = data
    else:
        if data.ndim == 3:
            data = data[:, :, 0]
            EEG['data'] = data

    # Remove baseline
    if not epoched:
        # Continuous data
        events = EEG.get('event', [])
        # Normalize events to list
        if isinstance(events, np.ndarray):
            try:
                events = events.tolist()
            except Exception:
                events = []
        use_boundaries = (
            isinstance(events, list)
            and len(events) > 0
            and isinstance(events[0], dict)
            and isinstance(events[0].get('type', None), str)
        )
        if use_boundaries:
            bidx = eeg_findboundaries(EEG=EEG)
            if len(bidx) == 0:
                # Manual check for boundaries - use this instead
                bidx = [i for i, ev in enumerate(events) if ev.get('type', '') == 'boundary']
        else:
            bidx = []

        if bidx:
            # MATLAB-compatible boundary processing
            # boundaries = round([ tmpevent(boundaries).latency ] -0.5-pointrange(1)+1);
            boundary_lats = []
            for i in bidx:
                try:
                    lat = float(events[i].get('latency', np.nan))
                    if not np.isnan(lat):
                        boundary_lats.append(lat)
                except Exception:
                    continue
            
            # Convert to MATLAB's boundary indices (relative to baseline start)
            # MATLAB formula: round(lat - 0.5 - pointrange(1) + 1)
            boundaries = []
            for lat in boundary_lats:
                boundary_idx = int(round_mat(lat - 0.5 - pr[0] + 1))
                boundaries.append(boundary_idx)
            
            # Filter boundaries to be within the baseline range
            # MATLAB: boundaries(boundaries>=pointrange(end)-pointrange(1)) = [];
            #         boundaries(boundaries<1) = [];
            baseline_len = pr[-1] - pr[0] + 1  # pointrange(end) - pointrange(1) + 1
            boundaries = [b for b in boundaries if 1 <= b < baseline_len]
            
            # Add start and end boundaries
            # MATLAB: boundaries = [0 boundaries pointrange(end)-pointrange(1)+1];
            boundaries = [0] + sorted(boundaries) + [baseline_len]
            
            # Process each segment
            for index in range(len(boundaries) - 1):
                # MATLAB: tmprange = [boundaries(index)+1:boundaries(index+1)];
                start_idx = boundaries[index] + 1  # 1-based in MATLAB
                end_idx = boundaries[index + 1]    # 1-based in MATLAB
                
                # Convert to 0-based Python indices within baseline range
                # pr[0] is 1-based MATLAB index, convert to 0-based Python index
                baseline_start_py = pr[0] - 1
                py_start = baseline_start_py + start_idx - 1  # start_idx is 1-based MATLAB
                py_end = baseline_start_py + end_idx - 1      # end_idx is 1-based MATLAB
                
                tmprange_len = end_idx - start_idx + 1
                
                if py_start < 0 or py_end < py_start:
                    continue
                    
                if tmprange_len > 1:
                    # Subtract mean of this segment
                    seg = EEG['data'][chanlist, py_start:py_end+1]
                    if seg.size > 0:
                        seg_means = np.nanmean(seg, axis=1, keepdims=True)
                        EEG['data'][chanlist, py_start:py_end+1] = seg - seg_means
                elif tmprange_len == 1:
                    EEG['data'][chanlist, py_start:py_end+1] = 0.0
        else:
            # No boundaries: subtract mean over baseline range from whole record
            EEG['data'][chanlist, :], _ = _subtract_mean_over_indices(EEG['data'][chanlist, :], pr)
    else:
        # Epoched data: for each channel, subtract mean over baseline indices per epoch
        # data shape: (nbchan, pnts, trials)
        for indc in chanlist:
            # mean across baseline points for each epoch → shape (trials,)
            m = np.nanmean(EEG['data'][indc, pr, :], axis=0)
            EEG['data'][indc, :, :] = EEG['data'][indc, :, :] - m[np.newaxis, :]

    # Clear ICA activations to remain consistent with EEGLAB behavior
    EEG['icaact'] = []

    return EEG


