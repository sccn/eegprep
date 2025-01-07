import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, next_fast_len
from scipy.signal import resample_poly
from .pop_loadset import pop_loadset
from .pop_reref import pop_reref

def eeg_autocorr_fftw(EEG, pct_data=100):
    
    # FFT length
    nfft = next_fast_len(2 * EEG['pnts'] - 1)
    
    # Initialize autocorrelation array
    ncomp = EEG['icaact'].shape[0]
    ac = np.zeros((ncomp, nfft))
    
    # Calculate autocorrelation using FFT
    for it in range(EEG['icaact'].shape[0]):
        # Apply FFT
        X = fft(EEG['icaact'][it, :, :], n=nfft, axis=0)
        # Compute the mean of the power spectrum
        ac[it, :] = np.mean(np.abs(X)**2, axis=1)
    
    # Inverse FFT to get autocorrelation
    ac = ifft(ac, axis=1)

    # make sure the data is in real
    ac = np.real(ac)
    
    # Adjust the size of autocorrelation array
    if EEG['pnts'] < EEG['srate']:
        # ac = np.hstack(     [ac[:, :EEG['pnts']], np.zeros((ncomp      , EEG['srate'] - EEG['pnts'] + 1))])
        ac = np.concatenate((ac[:, :EEG['pnts']], np.zeros((ac.shape[0], EEG['srate'] - EEG['pnts'] + 1))), axis=1)
    else:
        ac = ac[:, :int(EEG['srate']) + 1]
    
    # Normalize by 0-lag autocorrelation
    ac = ac / ac[:, 0][:, np.newaxis]
    
    # resample to 1 second at 100 samples/sec
    ac = resample_poly(ac.T, up=100, down=EEG['srate']).T
    ac = ac[:, 1:101]

    return ac
    
    
def test_eeg_autocorr_fftw():
    EEG = {
        'srate': 256,
        'icaweights': np.random.randn(10, 256),
        'pnts': 1000,
        'trials': 5,
        'icaact': np.random.randn(10, 1000, 5)
    }
    EEG = pop_loadset('/System/Volumes/Data/data/data/STUDIES/STERN/S01/Memorize.set')
    
    # reshape the last two dimensions of EEG['icaact']
    # EEG['icaact'] = EEG['icaact'].reshape(EEG['icaact'].shape[0], -1)
    
    # convert EEG['icaact'] to double precision
    
    psdmed = eeg_autocorr_fftw(EEG, 100)
    
    # print information about psdmed
    print(psdmed.shape)
    print(psdmed)
    
# test_eeg_autocorr_fftw()