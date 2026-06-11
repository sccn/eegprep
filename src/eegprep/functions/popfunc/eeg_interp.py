"""EEG channel interpolation utilities.

This module provides functions for interpolating bad channels in EEG data using various
methods including spherical spline interpolation.
"""

import numpy as np
from scipy.linalg import pinv
from scipy.interpolate import RBFInterpolator, griddata
from scipy.special import lpmv
from copy import deepcopy


def eeg_interp(EEG, bad_chans, method='spherical', t_range=None, params=None, dtype='float32'):
    """Interpolate missing or bad EEG channels using spherical spline.

    interpolation.

    Parameters
    ----------
    EEG : dict
        EEG data structure with 'data', 'chanlocs', 'nbchan', etc.
    bad_chans : list, array-like, or list of dicts
        Channel names, channel indices, or channel-location dictionaries.
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
        Precision for the interpolation computation. Use ``"float32"`` to
        match MATLAB-oriented workflows with lower memory use, or
        ``"float64"`` for full precision.

    Returns
    -------
    EEG : dict
        Updated EEG structure with interpolated channels
    """
    EEG = deepcopy(EEG)
    # set defaults
    method = _normalise_method(method)
    if method not in ('spherical', 'sphericalKang', 'sphericalCRD', 'sphericalfast', 'invdist', 'v4', 'spacetime'):
        raise ValueError(f"Unknown method {method}")
    if t_range is None:
        t_range = (EEG['xmin'], EEG['xmax'])
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

    # if bad chans is numerical, subtract 1 to make it 0-based
    # if isinstance(bad_chans, list) and isinstance(bad_chans[0], int):
    #     bad_chans = [i-1 for i in bad_chans]

    # Store original data shape to preserve it at the end
    original_data_shape = EEG['data'].shape

    # ensure channel locations present
    locs = EEG['chanlocs']
    # check if locs is null or empty
    if locs is None or len(locs) == 0:
        raise RuntimeError("Channel locations required for interpolation")
    if 'X' not in locs[0] or 'Y' not in locs[0] or 'Z' not in locs[0]:
        raise RuntimeError("Channel locations required for interpolation")

    # convert bad_chans from labels to indices if needed
    # Handle empty lists first
    if isinstance(bad_chans, list) and len(bad_chans) == 0:
        bad_idx = []
    # Check if bad_chans is a list of chanloc structures
    elif (
        isinstance(bad_chans, list)
        and len(bad_chans) > 0
        and isinstance(bad_chans[0], dict)
        and 'labels' in bad_chans[0]
        and 'X' in bad_chans[0]
        and 'Y' in bad_chans[0]
        and 'Z' in bad_chans[0]
    ):
        # Handle the new chanloc structure case
        EEG, bad_idx = _handle_chanloc_interpolation(EEG, bad_chans)
        # Update local variables that may have changed
        locs = EEG['chanlocs']
    elif isinstance(bad_chans, list) and len(bad_chans) > 0 and isinstance(bad_chans[0], str):
        labels = [ch['labels'] for ch in locs]
        bad_idx = [labels.index(lbl) for lbl in bad_chans]
    else:
        bad_idx = sorted(bad_chans)

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

    # extract Cartesian positions and normalize to unit sphere
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

    # Save original bad channel data before interpolation
    original_bad_data = d[bad_idx, :].copy()

    # compute interpolated signals for bad channels
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

    # restore original time range if needed
    if t_range != (EEG['xmin'], EEG['xmax']):
        t_start, t_end = t_range
        # Only apply for continuous data (trials=1), not epoched data (trials>1)
        # MATLAB's length(size(tmpdata))==2 is true for continuous data because
        # MATLAB drops trailing singleton dimensions
        if EEG['trials'] == 1 and 'srate' in EEG:
            # MATLAB convention: For continuous data, xmin is set to 0 by eeg_checkset,
            # so time values are interpreted as absolute sample indices using time*srate
            # times_2b_ignored = [1:floor(t_start*srate), floor(t_end*srate):floor(xmax*srate)]
            # (MATLAB 1-based indexing)
            # But empirically, MATLAB uses pnts-1 instead of floor(xmax*srate) for the upper bound

            # Calculate sample indices using MATLAB convention (time*srate, not (time-xmin)*srate)
            idx_start = int(np.floor(t_start * EEG['srate']))
            idx_end = int(np.floor(t_end * EEG['srate']))
            # Use pnts-1 as upper bound instead of floor(xmax*srate)
            idx_upper = EEG['pnts'] - 1

            # Build list of time indices to ignore (outside requested range)
            # In MATLAB: [1:idx_start, idx_end:pnts] where pnts is the last index
            # In Python (0-based): [0:max(0,idx_start), idx_end-1:pnts-2]
            # MATLAB keeps indices [idx_end-1 : idx_upper-1] as original
            times_to_ignore = []
            if idx_start > 0:
                times_to_ignore.extend(range(0, idx_start))
            # MATLAB keeps indices [idx_end-1 : idx_upper-1] as original
            # So we restore these indices
            if idx_end - 1 < idx_upper:
                times_to_ignore.extend(range(idx_end - 1, idx_upper))

            # Restore original data for bad channels at these time points
            if len(times_to_ignore) > 0:
                bad_data[:, times_to_ignore] = original_bad_data[:, times_to_ignore]

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
        # Check if the coordinate data is also identical
        coords_match = True
        for i, (curr_ch, new_ch) in enumerate(zip(current_locs, new_chanlocs)):
            if curr_ch['X'] != new_ch['X'] or curr_ch['Y'] != new_ch['Y'] or curr_ch['Z'] != new_ch['Z']:
                coords_match = False
                break

        if coords_match:
            # Return empty bad_idx since no interpolation needed
            return EEG, []
        else:
            # Same labels but different coordinates - this is ambiguous, throw error
            raise ValueError(
                "Channel labels are identical but coordinates differ. "
                "This is ambiguous - use different channel labels or identical coordinates."
            )

    # Check overlap between current and new labels
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

        # Update EEG structure
        EEG['data'] = new_data
        EEG['chanlocs'].extend(new_chanlocs)

        # The bad indices are the newly added channels
        bad_idx = list(range(EEG['nbchan'], EEG['nbchan'] + len(new_chanlocs)))
        EEG['nbchan'] = len(EEG['chanlocs'])

        return EEG, bad_idx

    # Case 3: Current channels are proper subset of new chanlocs
    elif current_set.issubset(new_set):
        # Create mapping from current channels to new positions
        old_to_new_idx = {}
        for i, label in enumerate(current_labels):
            new_idx = new_labels.index(label)
            old_to_new_idx[i] = new_idx

        # Create new data array with size matching new chanlocs
        original_shape = EEG['data'].shape
        if len(original_shape) == 3:  # epoched data
            new_data = np.zeros((len(new_chanlocs), original_shape[1], original_shape[2]))
            # Map existing data to correct positions
            for old_idx, new_idx in old_to_new_idx.items():
                new_data[new_idx, :, :] = EEG['data'][old_idx, :, :]
        else:  # continuous data
            new_data = np.zeros((len(new_chanlocs), original_shape[1]))
            # Map existing data to correct positions
            for old_idx, new_idx in old_to_new_idx.items():
                new_data[new_idx, :] = EEG['data'][old_idx, :]

        # Update EEG structure
        EEG['data'] = new_data
        EEG['chanlocs'] = new_chanlocs.copy()
        EEG['nbchan'] = len(new_chanlocs)

        # Handle ICA channel indices update (equivalent to MATLAB lines 174-189)
        if EEG.get('icasphere') is not None and hasattr(EEG['icasphere'], '__len__') and len(EEG['icasphere']) > 0:
            # Update icachansind if it exists and is not empty
            if (
                EEG.get('icachansind') is not None
                and hasattr(EEG['icachansind'], '__len__')
                and len(EEG['icachansind']) > 0
            ):
                # Convert icachansind to list if it's a numpy array for easier manipulation
                if hasattr(EEG['icachansind'], 'tolist'):
                    icachansind = EEG['icachansind'].tolist()
                else:
                    icachansind = list(EEG['icachansind'])

                # Create sort index equivalent to MATLAB's [~, sorti] = sort(neworder)
                # This maps from old position to new position in the sorted order
                updated_icachansind = []
                for old_ica_idx in icachansind:
                    # Find where this old channel index went in the new structure
                    if old_ica_idx in old_to_new_idx:
                        new_pos = old_to_new_idx[old_ica_idx]
                        updated_icachansind.append(new_pos)

                # Update both EEG.icachansind and EEG.chaninfo.icachansind
                EEG['icachansind'] = updated_icachansind

                # Ensure chaninfo exists and update icachansind there too
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
        # Find which of the new_chanlocs exist in current structure
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
