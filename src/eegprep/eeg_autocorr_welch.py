"""
EEG autocorrelation computation using Welch method.

This module provides functions for computing autocorrelation of EEG ICA components
using the Welch method for spectral estimation.
"""

import numpy as np
from scipy.signal import resample_poly
import random
from .pop_loadset import pop_loadset
import numpy as np
from numpy.fft import fft, ifft

def eeg_autocorr_welch(EEG, pct_data=100):
    """
    Compute autocorrelation of EEG ICA components using Welch method.
    
    Parameters
    ----------
    EEG : dict
        EEG data structure with 'icaweights', 'icaact', 'pnts', 'srate' fields.
    pct_data : float, optional
        Percentage of data to use. Default 100.
        
    Returns
    -------
    ndarray
        Autocorrelation array.
    """
    # clean input cutoff freq
    if pct_data is None or pct_data == 0:
        pct_data = 100
    
    # setup constants
    ncomp = EEG['icaweights'].shape[0]
    n_points = min(EEG['pnts'], EEG['srate'] * 3)
    nfft = 2**(int(np.log2(n_points * 2 - 1)) + 1)
    cutoff = (EEG['pnts'] // n_points) * n_points
    index = np.add.outer(np.ceil(np.arange(0, cutoff - n_points + 1, n_points // 2)).astype(int), np.arange(n_points)).astype(int)
    index = index.T
    
    # separate data segments
    if pct_data != 100:
        random.seed(0)
        n_seg = index.shape[0] * EEG['trials']
        subset = random.sample(range(n_seg), int(np.ceil(n_seg * pct_data / 100)))
        random.seed()  # restore normal random behavior
        temp = np.reshape(EEG['icaact'][:, index, :], (ncomp, *index.shape, EEG['trials']))
        segments = temp[:, :, subset]
    else:
        segments = np.reshape(EEG['icaact'][:, index, :], (ncomp, *index.shape, EEG['trials']))
    
    # calc autocorrelation
    ac = np.zeros((ncomp, nfft))
    for it in range(ncomp):
        fftpow = np.mean(np.abs(fft(segments[it, :, :], nfft, axis=0))**2, axis=1)
        ac[it, :] = np.real(ifft(fftpow, axis=0)).T
    
    # normalizefft
    if EEG['pnts'] < EEG['srate']:
        ac = np.concatenate([ac[:, :EEG['pnts']] / (ac[:, 0][:, np.newaxis] * np.arange(n_points, 0, -1) / n_points), 
                             np.zeros((ncomp, int(EEG['srate']) - n_points + 1))], axis=1)
    else:
        ac = ac[:, :int(EEG['srate']) + 1] / (ac[:, 0][:, np.newaxis] * np.concatenate((np.arange(n_points, n_points - int(EEG['srate']), -1), np.array([max(1, n_points - int(EEG['srate']))]))) / n_points)
    
    # resample to 1 second at 100 samples/sec
    ac = resample_poly(ac.T, up=100, down=EEG['srate']).T
    ac = ac[:, 1:101]
      
    return ac

def test_eeg_autocorr_welch():
    """Test function for eeg_autocorr_welch."""
    eeglab_file_path = './eeglab_data_with_ica_tmp.set'
    EEG = pop_loadset(eeglab_file_path)
    
    psdmed = eeg_autocorr_welch(EEG, 100)
    
    # print information about psdmed
    # print(psdmed.shape)
    # print(psdmed)

# test_eeg_autocorr_welch()
