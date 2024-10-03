import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, next_fast_len
from scipy.signal import resample

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
        ac = ac[:, :EEG['srate'] + 1]
    
    # Normalize by 0-lag autocorrelation
    ac = ac / ac[:, 0][:, np.newaxis]
    
    num_samples = round(ac.shape[1] * 100 / EEG['srate'])  # Calculate the number of samples for the new rate
    resamp = resample(ac.T, num_samples).T

    # Remove the first column
    resamp = resamp[:, 1:]
    
    return resamp
    
    
def test_eeg_autocorr_fftw():
    EEG = {
        'srate': 256,
        'icaweights': np.random.randn(10, 256),
        'pnts': 1000,
        'trials': 5,
        'icaact': np.random.randn(10, 1000, 5)
    }
    
    psdmed = eeg_autocorr_fftw(EEG, 100)
    
    # print information about psdmed
    print(psdmed.shape)
    print(psdmed)
    
# test_eeg_autocorr_fftw()