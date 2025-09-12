
# to do, look at line 83 and 84 and try to see if the MATLAB array output match. Run code side by side.

# EEG = pop_loadset('data/eeglab_data_tmp.set');
# EEG = eeg_interp(EEG, [1, 2, 3], 'spherical'); % or EEG = eeg_interp(EEG, {'Fp1' 'Fp2' 'F7'}, 'spherical');
# pop_save(EEG, 'data/eeglab_data_tmp_out_matlab.set');

import numpy as np
from scipy.linalg import pinv
from scipy.special import lpmv
from eegprep.eeg_compare import eeg_compare
from copy import deepcopy
import os

# absolute path for all files in data folder
data_path = '/Users/arno/Python/eegprep/data/' #os.path.abspath('data/')

def eeg_interp(EEG, bad_chans, method='spherical', t_range=None, params=None):
    """
    Interpolate missing or bad EEG channels using spherical spline interpolation.
    
    Parameters:
    -----------
    EEG : dict
        EEG data structure with 'data', 'chanlocs', 'nbchan', etc.
    bad_chans : list, array-like, or list of dicts
        Can be one of:
        - List of channel names (strings): e.g., ['Fp1', 'Fp2']
        - List of channel indices (integers): e.g., [0, 1, 2]
        - List of chanloc structures (dicts): e.g., [{'labels': 'T7', 'X': 0.8, 'Y': 0.0, 'Z': 0.6}, ...]
          When chanloc structures are provided, the function supports three modes:
          1. If chanlocs are identical to EEG['chanlocs'], returns data unchanged
          2. If no overlap with existing channels, appends new channels and interpolates them
          3. If existing channels are a subset, remaps data to new channel structure
    method : str, optional
        Interpolation method ('spherical', 'sphericalKang', 'sphericalCRD', 'sphericalfast')
    t_range : tuple, optional
        Time range for interpolation
    params : tuple, optional
        Method-specific parameters
        
    Returns:
    --------
    EEG : dict
        Updated EEG structure with interpolated channels
    """
    EEG = deepcopy(EEG)
    # set defaults
    if method not in ('spherical','sphericalKang','sphericalCRD','sphericalfast'):
        raise ValueError(f"Unknown method {method}")
    if t_range is None:
        t_range = (EEG['xmin'], EEG['xmax'])
    if params is None:
        if method=='spherical':
            params = (0,4,7)
        elif method=='sphericalKang':
            params = (1e-8,3,50)
        elif method=='sphericalCRD':
            params = (1e-5,4,500)
    else:
        if len(params)!=3:
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
    elif (isinstance(bad_chans, list) and len(bad_chans) > 0 and 
          isinstance(bad_chans[0], dict) and 
          'labels' in bad_chans[0] and 'X' in bad_chans[0] and 
          'Y' in bad_chans[0] and 'Z' in bad_chans[0]):
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
    empty_idx = [i for i in range(EEG['nbchan']) if np.isnan(locs[i]['X'])]
    good_idx = [i for i in good_idx if not np.isnan(locs[i]['X'])]

    # drop bad channels
    # data = EEG['data'].copy()
    # data = np.delete(data, bad_idx, axis=0)
    # EEG['data'] = data
    # EEG['nbchan'] = data.shape[0]

    # extract Cartesian positions and normalize to unit sphere
    def _norm(ch_ids):
        xyz = np.vstack([ [locs[i][c] for i in ch_ids] for c in ('X','Y','Z') ])
        rad = np.linalg.norm(xyz, axis=0)
        return xyz / rad

    xyz_good = _norm(good_idx)
    xyz_bad  = _norm(bad_idx)

    # reshape data to (n_chan, n_timepoints)
    d = EEG['data'].reshape(EEG['nbchan'], -1)

    # compute interpolated signals for bad channels
    bad_data = spheric_spline(
        xelec=xyz_good[0], yelec=xyz_good[1], zelec=xyz_good[2],
        xbad =xyz_bad[0],  ybad =xyz_bad[1],  zbad =xyz_bad[2],
        values=d[good_idx,:],
        params=params
    )

    # restore original time range if needed
    if t_range != (EEG['xmin'], EEG['xmax']):
        start, end = t_range
        ts = np.arange(EEG['nbchan']) # dummy
        # here you would mask out-of-range portions as in MATLAB

    # assemble full data array
    full = np.zeros_like(d)
    full[good_idx,:] = d[good_idx,:]
    full[empty_idx,:] = d[empty_idx,:]
    full[bad_idx,:]  = bad_data

    # Restore original data shape (2D for continuous, 3D for epoched)
    if len(original_data_shape) == 2:
        # Original was 2D continuous data
        EEG['data'] = full
    else:
        # Original was 3D epoched data or needs to be 3D
        EEG['data'] = full.reshape(EEG['nbchan'], EEG['pnts'], EEG['trials'])
    return EEG

def _handle_chanloc_interpolation(EEG, new_chanlocs):
    """
    Handle interpolation when bad_chans is provided as a list of chanloc structures.
    
    Returns:
        EEG: potentially modified EEG structure
        bad_idx: list of indices to interpolate
    """
    current_locs = EEG['chanlocs']
    current_labels = [ch['labels'] for ch in current_locs]
    new_labels = [ch['labels'] for ch in new_chanlocs]
    
    # Case 1: Identical chanlocs - return as-is
    if len(current_labels) == len(new_labels) and current_labels == new_labels:
        # Check if the coordinate data is also identical
        coords_match = True
        for i, (curr_ch, new_ch) in enumerate(zip(current_locs, new_chanlocs)):
            if (curr_ch['X'] != new_ch['X'] or 
                curr_ch['Y'] != new_ch['Y'] or 
                curr_ch['Z'] != new_ch['Z']):
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
            new_data[:EEG['nbchan'], :, :] = EEG['data']
        else:  # continuous data
            new_data = np.zeros((EEG['nbchan'] + len(new_chanlocs), original_shape[1]))
            new_data[:EEG['nbchan'], :] = EEG['data']
        
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

def spheric_spline(xelec, yelec, zelec, xbad, ybad, zbad, values, params):
    # values: (n_good, n_points)
    Gelec = computeg(xelec, yelec, zelec, xelec, yelec, zelec, params)
    Gsph  = computeg(xbad,  ybad,  zbad,  xelec, yelec, zelec, params)

    # Match MATLAB: mean across all values (not just axis=1)
    # mean across the first dimension
    meanvalues = values.mean(axis=0, dtype=np.float32)  # scalar mean across all dimensions
    values = values.astype(np.float32)
    values = values - meanvalues  # subtract scalar mean
    
    # Add zero row like MATLAB
    values = np.vstack([values, np.zeros((1, values.shape[1]))])

    lam = params[0]
    A   = np.vstack([Gelec + np.eye(Gelec.shape[0])*lam,
                     np.ones((1, Gelec.shape[0]))])
    C   = pinv(A) @ values # some minor differences with MATLAB in the pinv implementation

    allres = Gsph @ C
    # Add mean back like MATLAB: repmat(meanvalues, [size(allres,1) 1])
    allres = allres + meanvalues
    return allres

def computeg(x, y, z, xelec, yelec, zelec, params):
    # x,y,z are points to interpolate; xelec,... electrode locations
    X = x.ravel()[:,None]; Y = y.ravel()[:,None]; Z = z.ravel()[:,None]
    E = 1 - np.sqrt((X - xelec[None,:])**2 + (Y - yelec[None,:])**2 + (Z - zelec[None,:])**2)

    m, maxn = params[1], int(params[2])
    g = np.zeros((E.shape[0], E.shape[1]))
    for n in range(1, maxn+1):
        Pn = lpmv(0, n, E)  # shape (E.shape)
        g += ((2*n+1)/(n**m*(n+1)**m)) * Pn

    return g/(4*np.pi)

# Test functions moved to tests/test_eeg_interp.py

def test_chanloc_interpolation():
    """
    Example usage of the new chanloc interpolation functionality.
    This demonstrates the three different cases.
    """
    
    # Create a sample EEG structure
    EEG = {
        'data': np.random.randn(4, 100, 1),  # 4 channels, 100 time points, 1 trial
        'nbchan': 4,
        'pnts': 100,
        'trials': 1,
        'srate': 500,
        'xmin': 0,
        'xmax': 0.2,
        'chanlocs': [
            {'labels': 'Fp1', 'X': 0.1, 'Y': 0.8, 'Z': 0.6},
            {'labels': 'Fp2', 'X': -0.1, 'Y': 0.8, 'Z': 0.6},
            {'labels': 'F3', 'X': 0.4, 'Y': 0.6, 'Z': 0.7},
            {'labels': 'F4', 'X': -0.4, 'Y': 0.6, 'Z': 0.7},
        ]
    }
    
    print("Original EEG structure:")
    print(f"Data shape: {EEG['data'].shape}")
    print(f"Number of channels: {EEG['nbchan']}")
    print(f"Channel labels: {[ch['labels'] for ch in EEG['chanlocs']]}")
    
    # Case 1: Identical chanlocs (should return unchanged)
    identical_chanlocs = EEG['chanlocs'].copy()
    result1 = eeg_interp(EEG.copy(), identical_chanlocs)
    print(f"\nCase 1 - Identical chanlocs:")
    print(f"Data shape unchanged: {result1['data'].shape == EEG['data'].shape}")
    print(f"Data is identical: {np.array_equal(result1['data'], EEG['data'])}")
    
    # Case 2: No overlap (should append new channels)
    new_chanlocs = [
        {'labels': 'T7', 'X': 0.8, 'Y': 0.0, 'Z': 0.6},
        {'labels': 'T8', 'X': -0.8, 'Y': 0.0, 'Z': 0.6},
    ]
    result2 = eeg_interp(EEG.copy(), new_chanlocs)
    print(f"\nCase 2 - No overlap (append new channels):")
    print(f"Original channels: {EEG['nbchan']}, After: {result2['nbchan']}")
    print(f"Data shape: {EEG['data'].shape} -> {result2['data'].shape}")
    print(f"New channel labels: {[ch['labels'] for ch in result2['chanlocs']]}")
    
    # Case 3: Existing channels are proper subset (should remap to new structure)
    superset_chanlocs = [
        {'labels': 'Fp1', 'X': 0.1, 'Y': 0.8, 'Z': 0.6},
        {'labels': 'Fp2', 'X': -0.1, 'Y': 0.8, 'Z': 0.6},
        {'labels': 'F3', 'X': 0.4, 'Y': 0.6, 'Z': 0.7},
        {'labels': 'F4', 'X': -0.4, 'Y': 0.6, 'Z': 0.7},
        {'labels': 'C3', 'X': 0.6, 'Y': 0.0, 'Z': 0.8},
        {'labels': 'C4', 'X': -0.6, 'Y': 0.0, 'Z': 0.8},
    ]
    result3 = eeg_interp(EEG.copy(), superset_chanlocs)
    print(f"\nCase 3 - Existing subset of new structure:")
    print(f"Original channels: {EEG['nbchan']}, After: {result3['nbchan']}")
    print(f"Data shape: {EEG['data'].shape} -> {result3['data'].shape}")
    print(f"Final channel labels: {[ch['labels'] for ch in result3['chanlocs']]}")
    
    return result1, result2, result3

# Uncomment to run the test
# if __name__ == '__main__':
#     test_chanloc_interpolation()
