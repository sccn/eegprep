
# to do, look at line 83 and 84 and try to see if the MATLAB array output match. Run code side by side.

# EEG = pop_loadset('data/eeglab_data_tmp.set');
# EEG = eeg_interp(EEG, [1, 2, 3], 'spherical');
# pop_save(EEG, 'data/eeglab_data_tmp_out_matlab.set');

import numpy as np
from scipy.linalg import pinv
from scipy.special import lpmv
from eegprep.eeg_compare import eeg_compare
import os

# absolute path for all files in data folder
data_path = '/Users/arno/Python/eegprep/data/' #os.path.abspath('data/')

def eeg_interp(EEG, bad_chans, method='spherical', t_range=None, params=None):
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

    # ensure channel locations present
    locs = EEG['chanlocs']
    # check if locs is null or empty
    if locs is None or len(locs) == 0:
        raise RuntimeError("Channel locations required for interpolation")
    if 'X' not in locs[0] or 'Y' not in locs[0] or 'Z' not in locs[0]:
        raise RuntimeError("Channel locations required for interpolation")

    # convert bad_chans from labels to indices if needed
    if isinstance(bad_chans, list) and isinstance(bad_chans[0], str):
        labels = [ch['labels'] for ch in locs]
        bad_idx = [labels.index(lbl) for lbl in bad_chans]
    else:
        bad_idx = sorted(bad_chans)

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

    EEG['data'] = full.reshape(EEG['nbchan'], EEG['pnts'], EEG['trials'])
    return EEG

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

def test_spheric_spline():
    import numpy as np
    from scipy.io import loadmat, savemat

    # generate random electrode positions on the unit sphere
    rng = np.random.default_rng(0)
    n_good, n_bad, n_pts = 10, 2, 100
    xyz = rng.normal(size=(3, n_good))
    xyz /= np.linalg.norm(xyz, axis=0)
    xbad = rng.normal(size=(3, n_bad))
    xbad /= np.linalg.norm(xbad, axis=0)

    # random “good” channel data
    values = rng.standard_normal((n_good, n_pts))
    
    # write to MATLAB file
    mat = {
        'xelec': xyz[0],
        'yelec': xyz[1],
        'zelec': xyz[2],
        'xbad': xbad[0],
        'ybad': xbad[1],
        'zbad': xbad[2],    
        'values': values,
        'params': (0.0, 4.0, 7.0)
    }
    savemat(os.path.join(data_path, 'test_spheric_spline.mat'), mat)

    # compute in Python
    py_res = spheric_spline(
        xelec=xyz[0], yelec=xyz[1], zelec=xyz[2],
        xbad=xbad[0], ybad=xbad[1], zbad=xbad[2],
        values=values, params=(0, 4, 7)
    )

    # # load MATLAB result (assumed saved as `mat_res` in test.mat)
    mat_data = loadmat(os.path.join(data_path, 'test_spheric_spline_results.mat'))
    mat_res = mat_data['allres']  # Assuming the MATLAB result is saved as 'mat_res'
    
    # # compare
    diff = np.abs(py_res - mat_res)
    
    # do a proper max abs and rel difference
    max_abs_diff = np.max(np.abs(py_res - mat_res))
    max_rel_diff = np.max(np.abs(py_res - mat_res) / np.abs(mat_res))
    print(f"Max absolute difference: {max_abs_diff}")
    print(f"Max relative difference: {max_rel_diff}")    

def test_computeg():
    import numpy as np
    from scipy.io import loadmat, savemat
    # test computeg
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    z = np.linspace(0, 1, 100)
    xelec = np.linspace(0, 1, 10)
    yelec = np.linspace(0, 1, 10)
    zelec = np.linspace(0, 1, 10)
    params = (0.0, 4.0, 7.0)
    
    # save to mat file
    mat = {
        'x': x,
        'y': y,
        'z': z,
        'xelec': xelec,
        'yelec': yelec,
        'zelec': zelec,
        'params': params
    }
    savemat(os.path.join(data_path, 'test_computeg.mat'), mat)
    
    # compute in Python
    g = computeg(x, y, z, xelec, yelec, zelec, params)
    print("g.shape python:", g.shape)
    
    # load MATLAB result
    mat_data = loadmat(os.path.join(data_path, 'test_computeg_results.mat'))
    mat_res = mat_data['g']
    print("g.shape matlab:", mat_res.shape)

    # compare
    diff = np.abs(g - mat_res)
    
    # do a proper max abs and rel difference
    max_abs_diff = np.max(np.abs(g - mat_res))
    max_rel_diff = np.max(np.abs(g - mat_res) / np.abs(mat_res))
    print(f"Max absolute difference: {max_abs_diff}")
    print(f"Max relative difference: {max_rel_diff}")
    
def test_eeg_interp():
    # test eeg_interp
    from eegprep import pop_loadset
    # EEG = pop_loadset('../data/eeglab_data_tmp.set')
    EEG = pop_loadset(os.path.join(data_path, 'eeglab_data_tmp.set'))
    EEG = eeg_interp(EEG, [0, 1, 2], method='spherical')
    EEG2 = pop_loadset(os.path.join(data_path, 'eeglab_data_tmp_out_matlab.set'));
    # eeg_compare(EEG, EEG2)

    # compare data fields
    EEG['data'] = EEG['data'].reshape(EEG['nbchan'], -1)
    print('EEG[data] shape Python:', EEG['data'].shape)
    print('EEG2[data] shape MATLAB:', EEG2['data'].shape)
    
    print('np.allclose(EEG[data], EEG2[data]):', np.allclose(EEG['data'], EEG2['data']))
    
    # get max abs diff, index, and value at the max diff
    max_abs_diff = np.max(np.abs(EEG['data'] - EEG2['data']))
    max_idx = np.argmax(np.abs(EEG['data'] - EEG2['data']))
    max_coords = np.unravel_index(max_idx, EEG['data'].shape)
    print('Max abs diff:', max_abs_diff, 'value:', EEG['data'][max_coords])
    
    # find non-zero values before computing relative difference
    non_zero_idx = np.where(EEG['data'] != 0)
    rel_diff = np.abs(EEG['data'][non_zero_idx] - EEG2['data'][non_zero_idx]) / np.abs(EEG['data'][non_zero_idx])
    max_rel_diff = np.max(rel_diff)
    max_rel_idx_in_nonzero = np.argmax(rel_diff)
    max_rel_coords = (non_zero_idx[0][max_rel_idx_in_nonzero], non_zero_idx[1][max_rel_idx_in_nonzero])
    print('Max rel diff:', max_rel_diff, 'value:', EEG['data'][max_rel_coords])
    
if __name__ == '__main__':
    print("\nRunning test_computeg")
    test_computeg()
    print("\nRunning test_spheric_spline")
    test_spheric_spline()
    print("\nRunning test_eeg_interp")
    test_eeg_interp()
    
    
