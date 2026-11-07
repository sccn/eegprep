"""EEG channel interpolation utilities.

This module provides functions for interpolating bad channels in EEG data using various
methods including spherical spline interpolation.
"""

import numpy as np
from scipy.linalg import pinv
from scipy.interpolate import RBFInterpolator, griddata
from scipy.special import lpmv
from copy import deepcopy

from eegprep.functions.popfunc._chanutils import chanlocs_as_list


def eeg_interp(EEG, bad_chans, method='spherical', t_range=None, params=None, dtype=None):
    """Interpolate missing or bad EEG channels using spherical spline.

    interpolation.

    Parameters
    ----------
    EEG : dict
        EEG data structure with 'data', 'chanlocs', 'nbchan', etc.
    bad_chans : list, array-like, or list of dicts
        Channel names, 1-based channel indices, or channel-location dictionaries.
        Channel 1 is the first channel, as in ``pop_interp`` and EEGLAB.
        When channel-location dictionaries are provided, the function can
        return unchanged data for identical locations, append new channels when
        no existing locations overlap, or remap data when existing channels are
        a subset of the requested channel structure.
    method : str, optional
        Interpolation method ('spherical', 'sphericalKang', 'sphericalCRD',
        'sphericalfast', 'invdist'/'v4', or 'spacetime').
    t_range : tuple, optional
        Time range for interpolation
    params : tuple, optional
        Method-specific parameters
    dtype : str or dtype, optional
        Precision for the spherical-spline computation. The default follows
        the floating-point dtype of ``EEG['data']``, and uses ``float64`` when
        the data are not floating point. Pass ``"float32"`` for the
        lower-memory path.

    Returns
    -------
    EEG : dict
        Updated EEG structure with interpolated channels
    """
    bad_chans = _zero_based_channels(bad_chans, int(EEG["nbchan"]))
    EEG = deepcopy(EEG)
    if dtype is None:
        data_dtype = np.asarray(EEG['data']).dtype
        dtype = data_dtype if np.issubdtype(data_dtype, np.floating) else np.dtype('float64')
    # set defaults
    method = _normalise_method(method)
    if method not in ('spherical', 'sphericalKang', 'sphericalCRD', 'sphericalfast', 'invdist', 'v4', 'spacetime'):
        raise ValueError(f"Unknown method {method}")
    if t_range is None or np.size(t_range) == 0:
        t_range = (float(EEG['xmin']), float(EEG['xmax']))
    else:
        t_values = np.asarray(t_range, dtype=float).reshape(-1)
        if t_values.size != 2:
            raise ValueError(
                "eeg_interp: t_range must be [start, end] in seconds. "
                "Pass two numbers, or omit t_range to interpolate the whole recording."
            )
        t_range = (float(t_values[0]), float(t_values[1]))
    if params is None:
        if method == 'spherical':
            params = (0, 4, 7)
        elif method == 'sphericalKang':
            params = (1e-8, 3, 50)
        elif method == 'sphericalCRD':
            params = (1e-5, 4, 500)
    else:
        if len(params) != 3:
            raise ValueError("params must be length-3 tuple")
        method = 'spherical'

    original_data_shape = EEG['data'].shape

    locs = chanlocs_as_list(EEG['chanlocs'])
    EEG['chanlocs'] = locs
    if locs is None or len(locs) == 0:
        raise RuntimeError("Channel locations required for interpolation")
    if 'X' not in locs[0] or 'Y' not in locs[0] or 'Z' not in locs[0]:
        raise RuntimeError("Channel locations required for interpolation")

    # MATLAB-loaded channel-location structures are commonly object arrays.
    # Normalize them before distinguishing structures from numeric indices.
    candidate_chanlocs = chanlocs_as_list(bad_chans) if isinstance(bad_chans, (dict, list, tuple, np.ndarray)) else []
    if candidate_chanlocs and isinstance(candidate_chanlocs[0], dict):
        bad_chans = candidate_chanlocs

    location_channels = False
    if isinstance(bad_chans, list) and len(bad_chans) == 0:
        bad_idx = []
    elif (
        isinstance(bad_chans, list)
        and len(bad_chans) > 0
        and isinstance(bad_chans[0], dict)
        and 'labels' in bad_chans[0]
        and 'X' in bad_chans[0]
        and 'Y' in bad_chans[0]
        and 'Z' in bad_chans[0]
    ):
        location_channels = True
        EEG, bad_idx = _handle_chanloc_interpolation(EEG, bad_chans)
        locs = EEG['chanlocs']
    elif isinstance(bad_chans, list) and len(bad_chans) > 0 and isinstance(bad_chans[0], str):
        labels = [ch['labels'] for ch in locs]
        bad_idx = [labels.index(lbl) for lbl in bad_chans]
    else:
        bad_idx = sorted(set(bad_chans))

    # If no channels to interpolate, return as-is
    if len(bad_idx) == 0:
        return EEG

    good_idx = [i for i in range(EEG['nbchan']) if i not in bad_idx]
    empty_idx = [i for i in range(EEG['nbchan']) if (np.array_equal(locs[i]['X'], []) or np.isnan(locs[i]['X']))]
    good_idx = [i for i in good_idx if i not in empty_idx]
    bad_idx = [i for i in bad_idx if i not in empty_idx]

    # drop bad channels
    # data = EEG['data'].copy()
    # data = np.delete(data, bad_idx, axis=0)
    # EEG['data'] = data
    # EEG['nbchan'] = data.shape[0]

    def _norm(ch_ids):
        """Normalize channel coordinates to unit sphere.

        Parameters
        ----------
        ch_ids : list
            List of channel indices.

        Returns
        -------
        ndarray
            Normalized XYZ coordinates (3, n_channels).
        """
        xyz = np.vstack([[locs[i][c] for i in ch_ids] for c in ('X', 'Y', 'Z')])
        rad = np.linalg.norm(xyz, axis=0)
        return xyz / rad

    xyz_good = _norm(good_idx)
    xyz_bad = _norm(bad_idx)

    # reshape data to (n_chan, n_timepoints)
    d = EEG['data'].reshape(EEG['nbchan'], -1)

    original_bad_data = d[bad_idx, :].copy()

    if method in ('spherical', 'sphericalKang', 'sphericalCRD', 'sphericalfast'):
        bad_data = spheric_spline(
            xelec=xyz_good[0],
            yelec=xyz_good[1],
            zelec=xyz_good[2],
            xbad=xyz_bad[0],
            ybad=xyz_bad[1],
            zbad=xyz_bad[2],
            values=d[good_idx, :],
            params=params,
            dtype=dtype,
        )
    elif method in ('invdist', 'v4'):
        bad_data = _planar_v4_interpolate(locs, good_idx, bad_idx, d[good_idx, :])
    else:
        bad_data = _spacetime_interpolate(locs, good_idx, bad_idx, d[good_idx, :])

    bad_data = _restore_outside_time_range(bad_data, original_bad_data, EEG, t_range, location_channels)

    # assemble full data array
    full = np.zeros_like(d)
    full[good_idx, :] = d[good_idx, :]
    full[empty_idx, :] = d[empty_idx, :]
    full[bad_idx, :] = bad_data

    # Restore original data shape (2D for continuous, 3D for epoched)
    if len(original_data_shape) == 2:
        # Original was 2D continuous data
        EEG['data'] = full
    else:
        # Original was 3D epoched data or needs to be 3D
        EEG['data'] = full.reshape(EEG['nbchan'], EEG['pnts'], EEG['trials'])
    return EEG


def _sample_times_ms(EEG: dict, pnts: int) -> np.ndarray:
    if "times" not in EEG or EEG["times"] is None:
        raise ValueError(
            "eeg_interp: EEG.times is missing, so the time range cannot be applied. "
            "Run eeg_checkset so the dataset has one time value per sample."
        )
    values = np.asarray(EEG["times"], dtype=float).reshape(-1)
    if values.size != pnts:
        raise ValueError(
            f"eeg_interp: EEG.times has {values.size} values but EEG.pnts is {pnts}. "
            "Run eeg_checkset so times has one value per sample."
        )
    return values


def _restore_outside_time_range(bad_data, original_bad_data, EEG, t_range, location_channels):
    if location_channels:
        return bad_data
    start = float(t_range[0])
    end = float(t_range[1])
    if start == float(EEG["xmin"]) and end == float(EEG["xmax"]):
        return bad_data
    pnts = int(EEG["pnts"])
    times_ms = _sample_times_ms(EEG, pnts)
    outside = (times_ms < start * 1000.0) | (times_ms > end * 1000.0)
    if not np.any(outside):
        return bad_data
    restored = np.array(bad_data, dtype=float, copy=True).reshape(len(bad_data), pnts, -1)
    original = np.asarray(original_bad_data, dtype=float).reshape(len(bad_data), pnts, -1)
    restored[:, outside, :] = original[:, outside, :]
    return restored.reshape(len(bad_data), -1)


def _normalise_method(method):
    if not isinstance(method, str):
        return method
    method_lookup = {
        'spherical': 'spherical',
        'sphericalkang': 'sphericalKang',
        'sphericalcrd': 'sphericalCRD',
        'sphericalfast': 'sphericalfast',
        'invdist': 'invdist',
        'v4': 'v4',
        'spacetime': 'spacetime',
    }
    return method_lookup.get(method.lower(), method)


def _planar_v4_interpolate(locs, good_idx, bad_idx, values):
    """Use a thin-plate spline analogue of MATLAB griddata(..., 'v4')."""
    good_points = _planar_points(locs, good_idx)
    bad_points = _planar_points(locs, bad_idx)
    try:
        return RBFInterpolator(good_points, values, kernel='thin_plate_spline')(bad_points)
    except Exception:
        interpolated = griddata(good_points, values, bad_points, method='cubic')
        if np.isnan(interpolated).any():
            linear = griddata(good_points, values, bad_points, method='linear')
            interpolated = np.where(np.isnan(interpolated), linear, interpolated)
        if np.isnan(interpolated).any():
            nearest = griddata(good_points, values, bad_points, method='nearest')
            interpolated = np.where(np.isnan(interpolated), nearest, interpolated)
        return interpolated


def _spacetime_interpolate(locs, good_idx, bad_idx, values):
    """Match EEGLAB's nearest-neighbor space/time interpolation path."""
    good_points_2d = _planar_points(locs, good_idx)
    bad_points_2d = _planar_points(locs, bad_idx)
    n_time = values.shape[1]
    times = np.arange(1, n_time + 1)

    good_points = np.column_stack(
        [
            np.tile(good_points_2d[:, 0], n_time),
            np.tile(good_points_2d[:, 1], n_time),
            np.repeat(times, len(good_idx)),
        ]
    )
    bad_points = np.column_stack(
        [
            np.tile(bad_points_2d[:, 0], n_time),
            np.tile(bad_points_2d[:, 1], n_time),
            np.repeat(times, len(bad_idx)),
        ]
    )
    flattened = values.reshape(-1, order='F')
    interpolated = griddata(good_points, flattened, bad_points, method='nearest')
    return interpolated.reshape(len(bad_idx), n_time, order='F')


def _planar_points(locs, indices):
    points = []
    for index in indices:
        theta, radius = _theta_radius(locs[index])
        points.append((radius * np.sin(theta), radius * np.cos(theta)))
    return np.asarray(points, dtype=float)


def _theta_radius(chanloc):
    theta = chanloc.get('theta')
    radius = chanloc.get('radius')
    if not _is_empty_coordinate(theta) and not _is_empty_coordinate(radius):
        return float(np.deg2rad(float(theta))), float(radius)

    x = float(chanloc.get('X', np.nan))
    y = float(chanloc.get('Y', np.nan))
    if np.isnan(x) or np.isnan(y):
        raise RuntimeError("Channel theta/radius or X/Y locations required for planar interpolation")
    return float(np.arctan2(y, x)), float(np.sqrt(x * x + y * y))


def _is_empty_coordinate(value):
    if value is None:
        return True
    if isinstance(value, np.ndarray):
        return value.size == 0 or np.isnan(value).all()
    if isinstance(value, list):
        return len(value) == 0
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


def _handle_chanloc_interpolation(EEG, new_chanlocs):
    """Handle interpolation when bad_chans is provided as a list of chanloc.

    structures.

    Returns
    -------
    EEG : potentially modified EEG structure

    bad_idx : list of indices to interpolate
    """
    current_locs = EEG['chanlocs']
    current_labels = [ch['labels'] for ch in current_locs]
    new_labels = [ch['labels'] for ch in new_chanlocs]

    # Case 1: Identical chanlocs - return as-is
    if len(current_labels) == len(new_labels) and current_labels == new_labels:
        coords_match = True
        for i, (curr_ch, new_ch) in enumerate(zip(current_locs, new_chanlocs)):
            if curr_ch['X'] != new_ch['X'] or curr_ch['Y'] != new_ch['Y'] or curr_ch['Z'] != new_ch['Z']:
                coords_match = False
                break

        if coords_match:
            return EEG, []
        else:
            # Same labels but different coordinates - this is ambiguous, throw error
            raise ValueError(
                "Channel labels are identical but coordinates differ. "
                "This is ambiguous - use different channel labels or identical coordinates."
            )

    current_set = set(current_labels)
    new_set = set(new_labels)
    overlap = current_set.intersection(new_set)

    # Case 2: No overlap - append new channels
    if len(overlap) == 0:
        # Add new channels to data array (initialize with zeros)
        original_shape = EEG['data'].shape
        if len(original_shape) == 3:  # epoched data
            new_data = np.zeros((EEG['nbchan'] + len(new_chanlocs), original_shape[1], original_shape[2]))
            new_data[: EEG['nbchan'], :, :] = EEG['data']
        else:  # continuous data
            new_data = np.zeros((EEG['nbchan'] + len(new_chanlocs), original_shape[1]))
            new_data[: EEG['nbchan'], :] = EEG['data']

        EEG['data'] = new_data
        EEG['chanlocs'].extend(new_chanlocs)

        # The bad indices are the newly added channels
        bad_idx = list(range(EEG['nbchan'], EEG['nbchan'] + len(new_chanlocs)))
        EEG['nbchan'] = len(EEG['chanlocs'])

        return EEG, bad_idx

    # Case 3: Current channels are proper subset of new chanlocs
    elif current_set.issubset(new_set):
        old_to_new_idx = {}
        for i, label in enumerate(current_labels):
            new_idx = new_labels.index(label)
            old_to_new_idx[i] = new_idx

        original_shape = EEG['data'].shape
        if len(original_shape) == 3:  # epoched data
            new_data = np.zeros((len(new_chanlocs), original_shape[1], original_shape[2]))
            for old_idx, new_idx in old_to_new_idx.items():
                new_data[new_idx, :, :] = EEG['data'][old_idx, :, :]
        else:  # continuous data
            new_data = np.zeros((len(new_chanlocs), original_shape[1]))
            for old_idx, new_idx in old_to_new_idx.items():
                new_data[new_idx, :] = EEG['data'][old_idx, :]

        EEG['data'] = new_data
        EEG['chanlocs'] = new_chanlocs.copy()
        EEG['nbchan'] = len(new_chanlocs)

        # Handle ICA channel indices update (equivalent to MATLAB lines 174-189)
        if EEG.get('icasphere') is not None and hasattr(EEG['icasphere'], '__len__') and len(EEG['icasphere']) > 0:
            if (
                EEG.get('icachansind') is not None
                and hasattr(EEG['icachansind'], '__len__')
                and len(EEG['icachansind']) > 0
            ):
                if hasattr(EEG['icachansind'], 'tolist'):
                    icachansind = EEG['icachansind'].tolist()
                else:
                    icachansind = list(EEG['icachansind'])

                # Create sort index equivalent to MATLAB's [~, sorti] = sort(neworder)
                # This maps from old position to new position in the sorted order
                updated_icachansind = []
                for old_ica_idx in icachansind:
                    if old_ica_idx in old_to_new_idx:
                        new_pos = old_to_new_idx[old_ica_idx]
                        updated_icachansind.append(new_pos)

                EEG['icachansind'] = updated_icachansind

                if 'chaninfo' not in EEG:
                    EEG['chaninfo'] = {}
                EEG['chaninfo']['icachansind'] = updated_icachansind

        # Bad indices are all positions that don't have existing data
        existing_new_indices = set(old_to_new_idx.values())
        bad_idx = [i for i in range(len(new_chanlocs)) if i not in existing_new_indices]

        return EEG, bad_idx

    else:
        # Partial overlap case - not clearly specified in requirements
        # Default to treating new_chanlocs as the channels to interpolate
        bad_idx = []
        for i, new_ch in enumerate(new_chanlocs):
            if new_ch['labels'] in current_labels:
                bad_idx.append(current_labels.index(new_ch['labels']))

        return EEG, bad_idx


def spheric_spline(xelec, yelec, zelec, xbad, ybad, zbad, values, params, dtype='float32'):
    """Perform spherical spline interpolation.

    Parameters
    ----------
    xelec, yelec, zelec : array-like
        Coordinates of good electrodes.
    xbad, ybad, zbad : array-like
        Coordinates of bad electrodes to interpolate.
    values : ndarray
        Data values at good electrodes.
    params : tuple
        Interpolation parameters (lambda, m, maxn).
    dtype : str or dtype, optional
        Data type for computation.

    Returns
    -------
    ndarray
        Interpolated values at bad electrode positions.
    """
    dtype = np.dtype(dtype)

    # values: (n_good, n_points)
    Gelec = computeg(xelec, yelec, zelec, xelec, yelec, zelec, params)
    Gsph = computeg(xbad, ybad, zbad, xelec, yelec, zelec, params)

    # Match MATLAB: mean across all values (not just axis=1)
    # mean across the first dimension
    meanvalues = values.mean(axis=0, dtype=dtype)  # scalar mean across all dimensions
    values = values.astype(dtype)
    values = values - meanvalues  # subtract scalar mean

    # Add zero row like MATLAB
    values = np.vstack([values, np.zeros((1, values.shape[1]))])

    lam = params[0]
    A = np.vstack([Gelec + np.eye(Gelec.shape[0]) * lam, np.ones((1, Gelec.shape[0]))])
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        # Matches MATLAB numerically; NumPy may warn inside finite pseudo-inverse matmuls.
        C = pinv(A) @ values
        allres = Gsph @ C
    # Add mean back like MATLAB: repmat(meanvalues, [size(allres,1) 1])
    allres = allres + meanvalues
    return allres


def computeg(x, y, z, xelec, yelec, zelec, params):
    """Compute spherical spline basis functions.

    Parameters
    ----------
    x, y, z : array-like
        Coordinates of points to evaluate.
    xelec, yelec, zelec : array-like
        Coordinates of electrode positions.
    params : tuple
        Parameters (lambda, m, maxn).

    Returns
    -------
    ndarray
        Basis function values.
    """
    # x,y,z are points to interpolate; xelec,... electrode locations
    X = x.ravel()[:, None]
    Y = y.ravel()[:, None]
    Z = z.ravel()[:, None]
    E = 1 - np.sqrt((X - xelec[None, :]) ** 2 + (Y - yelec[None, :]) ** 2 + (Z - zelec[None, :]) ** 2)

    m, maxn = params[1], int(params[2])
    g = np.zeros((E.shape[0], E.shape[1]))
    for n in range(1, maxn + 1):
        Pn = lpmv(0, n, E)  # shape (E.shape)
        g += ((2 * n + 1) / (n**m * (n + 1) ** m)) * Pn

    return g / (4 * np.pi)


def _zero_based_channels(bad_elec, nbchan):
    if isinstance(bad_elec, dict):
        return bad_elec
    if isinstance(bad_elec, np.ndarray) and bad_elec.dtype == object:
        bad_elec = bad_elec.reshape(-1).tolist()
    if isinstance(bad_elec, (list, tuple)) and bad_elec and isinstance(bad_elec[0], (dict, str)):
        return list(bad_elec) if isinstance(bad_elec, tuple) else bad_elec
    values = np.asarray([] if bad_elec is None else bad_elec)
    if values.size == 0:
        return []
    if values.dtype.kind not in {"i", "u", "f"}:
        return bad_elec
    flat = values.astype(float, copy=False).reshape(-1)
    if np.any(~np.isfinite(flat)) or np.any(flat != np.trunc(flat)):
        raise ValueError(
            f"eeg_interp: channel indices must be whole numbers from 1 to {nbchan}. Channel 1 is the first channel."
        )
    indices = flat.astype(int)
    outside = indices[(indices < 1) | (indices > nbchan)]
    if outside.size:
        raise ValueError(
            f"eeg_interp: channel index {int(outside[0])} is outside 1..{nbchan}. Channel 1 is the first channel."
        )
    return (indices - 1).tolist()

