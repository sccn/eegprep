import numpy as np
from scipy.signal import resample
from numpy.fft import fft, ifft

def eeg_autocorr(EEG, pct_data=None):
    if pct_data is None:
        pct_data = 100

    # convert EEG['icaact'] to single precision
    EEG['icaact'] = EEG['icaact'].astype(np.float32)

    ncomp = EEG['icaact'].shape[0]
    nfft = 2**np.ceil(np.log2(2 * EEG['pnts'] - 1)).astype(int)

    # Calculate autocorrelation
    c = np.zeros((ncomp, nfft))
    for it in range(ncomp):
        comp = EEG['icaact'][it,:].reshape(-1)
        Xtmp = fft(comp, nfft)
        Xtmp = Xtmp.astype(np.complex64) # matches MATLAB's single precision since python convert to double precision by default
        X = np.abs(Xtmp)**2
        c[it, :] = np.real(ifft(X))

    # Adjust the size of the autocorrelation to match sampling rate
    if EEG['pnts'] < EEG['srate']:
        ac = np.hstack([c[:, :EEG['pnts']], np.zeros((ncomp, EEG['srate'] - EEG['pnts'] + 1))])
    else:
        ac = c[:, :EEG['srate'] + 1]

    # Normalize by the 0-tap of the autocorrelation
    ac /= ac[:, [0]]

    # Resample to 1 second at 100 samples/sec

    # print the size of the second dim of ac
    num_samples = round(ac.shape[1] * 100 / EEG['srate'])  # Calculate the number of samples for the new rate
    resamp = resample(ac.T, num_samples).T

    # Remove the first column
    resamp = resamp[:, 1:]
    
    return resamp

def test_eeg_autocorr():
    EEG = {
        'srate': 256,
        'icaweights': np.random.randn(10, 256),
        'pnts': 1000,
        'trials': 5,
        'icaact': np.random.randn(10, 1000, 5)
    }
    
    psdmed = eeg_autocorr(EEG, 100)
    
    # print information about psdmed
    print(psdmed.shape)
    
    #print(psdmed)
    
    
    #assert psdmed.shape == (10, 100)
    #assert np.all(np.isfinite(psdmed))

test_eeg_autocorr()