import numpy as np
from numpy.fft import fft
from scipy.signal.windows import hamming

def eeg_rpsd(EEG, nfreqs=None, pct_data=100):
    # clean input cutoff freq
    nyquist = EEG['srate'] // 2
    if nfreqs is None or nfreqs > nyquist:
        nfreqs = nyquist
    
    # setup constants
    ncomp = EEG['icaweights'].shape[0]
    
    # Hamming window
    n_points = min(EEG['pnts'], EEG['srate'])
    m = n_points
    isOddLength = m % 2
    if isOddLength:
        x = np.arange(0, (m - 1) / 2 + 1) / (m - 1)
    else:
        x = np.arange(0, m / 2) / (m - 1)
    
    a = 0.54
    window = a - (1 - a) * np.cos(2 * np.pi * x)
    if isOddLength:
        window = np.concatenate([window, window[-2::-1]])
    else:
        window = np.concatenate([window, window[::-1]])
    
    cutoff = (EEG['pnts'] // n_points) * n_points
    index = np.add.outer(np.arange(0, cutoff - n_points//2, n_points // 2), np.arange(0, n_points)).astype(int).transpose()
    
    np.random.seed(0)  # rng('default') in MATLAB
    n_seg = index.shape[1] * EEG['trials']
    subset = np.random.permutation(n_seg)[:int(n_seg * pct_data / 100)]
    
    # calculate windowed spectrums
    psdmed = np.zeros((ncomp, nfreqs))
    for it in range(ncomp):
        temp = np.reshape(EEG['icaact'][it, index, :], (1, index.shape[0], index.shape[1] * EEG['trials']))
        temp = temp[:, :, subset] * window[:, np.newaxis]
        temp = fft(temp, n_points, axis=1)
        temp = np.abs(temp) ** 2
        temp = temp[:, 1:nfreqs + 1, :] * 2 / (EEG['srate'] * np.sum(window ** 2))
        if nfreqs == nyquist:
            temp[:, -1, :] /= 2
        psdmed[it, :] = 20 * np.log10(np.median(temp, axis=2))
    
    return psdmed

def test_eeg_rpsd():
    EEG = {
        'srate': 256,
        'icaweights': np.random.randn(10, 256),
        'pnts': 1000,
        'trials': 5,
        'icaact': np.random.randn(10, 1000, 5)
    }
    
    psdmed = eeg_rpsd(EEG, 100)
    assert psdmed.shape == (10, 100)
    assert np.all(np.isfinite(psdmed))

# test_eeg_rpsd()